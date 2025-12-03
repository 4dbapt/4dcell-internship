'''
Imports
'''
import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tkinter.filedialog import askdirectory
from scipy.signal import savgol_filter, find_peaks

np.set_printoptions(legacy='1.25')
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})


'''
Function definitions
'''
'''
Sorts peaks and troughs and checks if they match.
Parameters:
peaks (list or array-like): Indices of the peaks in the signal.
all_troughs (list or array-like): Indices of all troughs in the signal.
Returns:
tuple: Peaks, pretroughs, troughs, passs, message.
Peaks, pretroughs and troughs are lists of indices.
Passs is a boolean indicating if the signal should be skipped.
Message is a string with a reason for skipping the signal.
'''
def sort_peaks_troughs(peaks, all_troughs):

    filtered_troughs = []
    last_troughs = []  # list to store last min after each peak
    first_troughs = all_troughs[all_troughs < peaks[0]]   # First, check for troughs before the first peak 

    if len(first_troughs) > 0:
        last_troughs.append(first_troughs[-1])  # add the closest trough to first peak to last_troughs

    for i in range(len(peaks)):
        # find all troughs after this peak and before next peak
        if i == len(peaks) - 1: next_troughs = all_troughs[all_troughs > peaks[i]]
        else: next_troughs = all_troughs[(all_troughs > peaks[i]) & (all_troughs < peaks[i+1])]
        if len(next_troughs) > 0:     # if there are multiple troughs after a peak or before first peak, keep only the closest ones
            filtered_troughs.append(next_troughs[0])
            last_troughs.append(next_troughs[-1]) # keep the last one for last_troughs list

    troughs = np.array(filtered_troughs)
    pretroughs = np.array(last_troughs)

    if pretroughs[0] > peaks[0]: # Ensure the first beat starts with a pretrough
        peaks = peaks[1:]
    if len(peaks) < 2:
        message = 'One or less peaks found. Onto the next signal.'
        return None, None, None, True, message

    if peaks[-1] > troughs[-1]: # Ensure the last peak ends with a trough
        peaks = peaks[:-1]
    if len(peaks) < 2:
        message = 'One or less peaks found. Onto the next signal.'
        return None, None, None, True, message

    if pretroughs[-1] > peaks[-1]: # Ensure the last pretrough is before the last peak
        pretroughs = pretroughs[:-1]
    if troughs[0] < peaks[0]: # Ensure the first trough is after the first peak
        troughs = troughs[1:]

    if len(peaks) != len(troughs) or len(peaks) != len(pretroughs):     # if each peak doesn't have a pretrough before and a trough after, pass
        message = 'Peaks, troughs and pretroughs do not match. Onto the next signal.'
        return None, None, None, True, message
    
    return peaks, pretroughs, troughs, False, None


"""
Computes the maximum slopes of the signal between pretroughs and peaks.
Parameters:
time (list or array-like): The time points corresponding to the signal values.
signal (list or array-like): The signal values.
peaks (list or array-like): Indices of the peaks in the signal.
pretroughs (list or array-like): Indices of the pretroughs in the signal.
Returns:
list: A list of maximum slopes for each peak-pretrough pair.
"""
def compute_max_slopes(time, signal, peaks, pretroughs):
    rise_max_slopes = []
    for i in range(len(peaks)):
        peak_idx = peaks[i]
        pretrough_idx = pretroughs[i]
        max_slope = 0
        for j in range(pretrough_idx, peak_idx):
            rise = signal[j+1] - signal[j]
            rise_time = time[j+1] - time[j]
            slope = rise / rise_time
            if slope > max_slope:
                max_slope = slope
        rise_max_slopes.append(max_slope)
    return rise_max_slopes


'''
Function to compute APD values for each action potential.
Parameters:
i (int): Index of the signal.
apd_percents (list): List of APD percentages to compute.
peaks (list): List of peak indices.
signal (DataFrame): DataFrame containing the signal data.

pretroughs (list): List of pretrough indices.
troughs (list): List of trough indices.
Returns:
tuple: APD off, APD off x, APD off y, APD, APD on x, APD on y.
APD off, APD off x, APD off y are dictionaries containing the APD values and corresponding times for each APD percentage.
APD, APD on x, APD on y are dictionaries containing the APD values and corresponding times for each APD percentage.
'''
def get_apds(i, apd_percents, peaks, signal, pretroughs, troughs):
    # Create lists to store all APD values
    apd_off_x = {f'apd{p}off_x': [] for p in apd_percents}
    apd_off_y = {f'apd{p}off_y': [] for p in apd_percents}
    apd = {f'apd{p}': [] for p in apd_percents}
    apd_off = {f'apd{p}off': [] for p in apd_percents}
    apd_on_x = {f'apd{p}on_x': [] for p in apd_percents}  
    apd_on_y = {f'apd{p}on_y': [] for p in apd_percents}

    for i in range(len(peaks)):
        peak_value = signal['Mean_norm'][peaks[i]]
        trough_value = signal['Mean_norm'][troughs[i]]
        pretrough_value = signal['Mean_norm'][pretroughs[i]]
                
        # Calculate target values for each APD percentage
        targets_off = {f'target_{p}': peak_value - (p/100)*(peak_value-trough_value) for p in apd_percents}
        targets_on = {f'target_{p}_on': pretrough_value - (p/100)*(pretrough_value-peak_value) for p in apd_percents}
                
        for p in apd_percents:
            crossings_off = []
            crossings_on = []
            target_off = targets_off[f'target_{p}']
            target_on = targets_on[f'target_{p}_on']
                    
            for j in range(peaks[i], troughs[i]): # find all crossings between peak and trough
                if (signal['Mean_norm'][j-1] >= target_off >= signal['Mean_norm'][j]) or (signal['Mean_norm'][j-1] <= target_off <= signal['Mean_norm'][j]):
                    crossings_off.append(j)
                    
            for j in range(pretroughs[i], peaks[i]): # find all crossings between pretrough and peak
                if (signal['Mean_norm'][j-1] <= target_on <= signal['Mean_norm'][j]) or (signal['Mean_norm'][j-1] >= target_on >= signal['Mean_norm'][j]):
                    crossings_on.append(j)
                    
            # Store the results if crossings were found
            if crossings_off:
                last_idx = crossings_off[-1]
                off_duration = signal['Time (s)'][last_idx] - signal['Time (s)'][peaks[i]]
                apd_off[f'apd{p}off'].append(off_duration)
                apd[f'apd{p}'].append(signal['Time (s)'][last_idx] - signal['Time (s)'][crossings_on[0]])
                apd_off_y[f'apd{p}off_y'].append(signal['Mean_norm'][last_idx])
                apd_off_x[f'apd{p}off_x'].append(signal['Time (s)'][last_idx])
                    
            if crossings_on:
                first_idx = crossings_on[0]
                apd_on_x[f'apd{p}on_x'].append(signal['Time (s)'][first_idx])
                apd_on_y[f'apd{p}on_y'].append(signal['Mean_norm'][first_idx])

    return apd_off, apd_off_x, apd_off_y, apd, apd_on_x, apd_on_y
#___________________________________________________________________________________________________________________________________________________________________________________#


'''
Load Files storing raw data
'''
EXPOSURE = 1/0.033   # in milliseconds    
#TODO: maybe ask user or find in metadata ??

# Selecting folder to analyse (folder containing all conditions)
folder = askdirectory(title='Select the folder containing all the conditions to analyse')

conditions = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('Results')] # get only folders in folder except Results folder
if len(conditions)==0 :
    print('WARNING: No conditions found in folder. Exiting...')
    exit()

'''
Loop through all folders/conditions in folder
'''
for i, condition in enumerate(conditions):
    print("Analyzing condition: "+ condition)
    path = f'{folder}/{condition}'
    files = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file = os.path.join(path, file)
            files.append(file) # list of all files in folder condition

    # Create Results folder if it doesn't exist
    if not os.path.exists(f'{folder}/Results'):
        os.makedirs(f'{folder}/Results')
    if not os.path.exists(f'{folder}/Results/{condition}'):
        os.makedirs(f'{folder}/Results/{condition}')

    # Initializing dataframe to store results
    apd_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    columns = ['Name', 'Period (s)', 'Frequence (Hz)', 'Interpeak std (s)', 'Action potential (s)', 'Rise time (s)', 'Max rise slope (a.u./s)', 'Amplitude (a.u.)', 'Amplitude std (a.u.)', 'Triangulation (apd50/apd90)']
    for p in apd_percents:
        columns.append(f'APD{p} mean (s)')
        columns.append(f'APD{p}off mean (s)')
    results_allrings = pd.DataFrame(index=range(len(files)), columns=columns)

    '''
    Loop through all files in folder/condition and analyses them
    '''
    for f in range(len(files)):
        name = os.path.basename(files[f]).split('.')[0]  # whole file name without extension
        name = name.split('_', 1)[1] # TODO: remove this after modification in java code, right now it removes "rawresults_" in front of name
        name_condition = name.rsplit('_', 1)[0] # name and condition without well and ring number
        print('Processing file: ', name)


        ''' 1. Load and plot raw data '''
        raw_data = pd.read_csv(files[f])
        mean_column = next(col for col in raw_data.columns if col.startswith('Mean')) # find column with 'Mean' in its name
        raw_data = raw_data[[mean_column]]  
        raw_data.columns = ['Mean']
        raw_data['Time (s)'] = ((raw_data.index+1)*EXPOSURE)/1000 # convert time index to seconds

        plt.figure(figsize=(15, 4)) # plot raw data
        plt.plot(raw_data['Time (s)'], raw_data['Mean'])
        plt.xlabel('Time (s)')
        plt.ylabel('Mean')
        plt.title('Raw plot '+name) 
        plt.show()


        ''' 2. Find preliminary peaks and period, 
            and skip signal if one or less peaks found '''
        peaks, _ = find_peaks(raw_data['Mean'], prominence=1, distance=5)  # fairly high prominence to find clear peaks, distance is very low
        if len(peaks) < 2:
            print('One or less peaks found. Onto the next signal.')
            continue
        preliminary_period = np.diff(peaks).mean() * EXPOSURE/1000


        ''' 3. Interpolate, smooth signal and subtract baseline '''
        n_points = 5000  # points in interpolated signal (a bit arbitrary maybe)
        interp_func = interp1d(raw_data['Time (s)'], raw_data['Mean'], kind='quadratic')
        time_interp = np.linspace(raw_data['Time (s)'].min(), raw_data['Time (s)'].max(), n_points)   # create a new time array with more points
        mean_interp = interp_func(time_interp) # interpolation
        signal = pd.DataFrame() # store all interpolated signals in new dataframe (they have different lengths)
        signal['Mean'] = mean_interp
        signal['Time (s)'] = time_interp

        window = int(n_points/100)
         # window size for savgol filter is arbitrary too... #TODO: should depend of prelim preiod? ask Magali
        signal['Mean_smooth_savgol'] = savgol_filter(signal['Mean'], window, polyorder = 3)

        fit = np.polyfit(signal['Time (s)'], signal['Mean_smooth_savgol'], 1)  # fit a linear function to the smoothed signal
        signal['linear_fit'] = fit[0]*signal['Time (s)'] + fit[1]  # linear fit signal
        signal['Mean_BS'] = signal['Mean_smooth_savgol'] - signal['linear_fit'] + signal['linear_fit'].mean() # baseline subtraction + mean addition


        ''' 4. Find peaks, troughs and pretroughs 
            and exclude some if they don't match '''    
        min = signal['Mean_BS'].min()
        max = signal['Mean_BS'].max()
        max_peak = (max-min)*0.75 + min # peaks and troughs must be in the last of first 25% of the signal respectivaly
        max_trough = (max-min)*0.25 + min
        distance = int(preliminary_period * n_points/signal['Time (s)'].max()/2.5) # distance between peaks is more than half the period in points

        all_peaks, _ = find_peaks(signal['Mean_BS'], distance = distance, height = (max_peak, None), prominence=0.03) # prominence is lower heres as te signal is smoothed
        all_troughs, _ = find_peaks(-signal['Mean_BS'], height = (-max_trough, None), prominence=0.01) 

        peaks, pretroughs, troughs, passs, message = sort_peaks_troughs(all_peaks, all_troughs)
        if passs: # pass if conditions in fucntion are not satisfied
            print(message)
            continue

        print("Peaks: ", peaks)
        print("Pretroughs: ", pretroughs)
        print("Troughs: ", troughs)


        ''' 5. Normalize signal '''     # only done here because troughs are needed to compute baseline
        baseline = signal['Mean_BS'][troughs].mean()
        signal['Mean_norm'] = (signal['Mean_BS'] - baseline)/baseline


        ''' 6. Compute main metrics here '''
        amplitudes = np.array(signal['Mean_norm'][peaks]) - np.array(signal['Mean_norm'][troughs])
        mean_amplitude = np.mean(amplitudes)
        std_amplitude = np.std(amplitudes)
        interpeak_distance = pd.Series(signal['Time (s)'][peaks]).diff().dropna()
        period = interpeak_distance.mean() 
        interpeak_std = interpeak_distance.std() # will be None if only 2 peaks
        rise_times =  np.array(signal['Time (s)'][peaks]) - np.array(signal['Time (s)'][pretroughs])
        rise_time = rise_times.mean()
        rise_max_slopes = compute_max_slopes(signal['Time (s)'], signal['Mean_norm'], peaks, pretroughs)
        rise_max_slope = np.array(rise_max_slopes).mean()
        action_potentials = np.array(signal['Time (s)'][troughs]) - np.array(signal['Time (s)'][pretroughs]) # time between pretrough and trough = apd100off
        action_potential = action_potentials.mean()


        ''' 7. Compute and plot APD values '''
        apd_off, apd_off_x, apd_off_y, apd, apd_on_x, apd_on_y = get_apds(i, apd_percents, peaks, signal, pretroughs, troughs) # compute APD values for each action potential

        apd_on_x = {f'apd{100-p}on_x': apd_on_x[f'apd{p}on_x'] for p in apd_percents} # turn around values for APDon values
        apd_on_y = {f'apd{100-p}on_y': apd_on_y[f'apd{p}on_y'] for p in apd_percents}

        # Plot the results on whole signal
        plt.figure(figsize=(13, 3.5))
        plt.plot(signal['Time (s)'], signal['Mean_norm'], label='Mean Signal')
        plt.scatter(signal['Time (s)'][peaks], signal['Mean_norm'][peaks], color='red', label='Peaks')
        plt.scatter(signal['Time (s)'][troughs], signal['Mean_norm'][troughs], color='green', label='Ends of AP')
        plt.scatter(signal['Time (s)'][pretroughs], signal['Mean_norm'][pretroughs], color='blue', label='Starts of AP')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(apd_percents))) # plot APD points with different colors
        for (p, color) in zip(apd_percents, colors):
            plt.scatter(apd_off_x[f'apd{p}off_x'], apd_off_y[f'apd{p}off_y'], 
                        color=color, label=f'APD{p}', marker='o', alpha=0.7)
            plt.scatter(apd_on_x[f'apd{p}on_x'], apd_on_y[f'apd{p}on_y'], 
                        color=color, marker='d', alpha=0.7)
        plt.title("Signal with APD values")
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F0 (a.u.)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{folder}/Results/{condition}/{name}_plot.png', facecolor = 'w', bbox_inches="tight", pad_inches=0.3)
        plt.show()

        # Plot second beat 
        end_trough = troughs[1]
        pretrough = pretroughs[1]
        plt.figure(figsize=(6, 4))
        plt.plot(signal['Time (s)'][pretrough:end_trough], signal['Mean_norm'][pretrough:end_trough], label='Signal')
        # Add peak and trough points
        peak_in_range = peaks[(peaks > pretrough) & (peaks < end_trough)]
        plt.scatter(signal['Time (s)'][peak_in_range], signal['Mean_norm'][peak_in_range], color='red', label='Peak')
        plt.scatter(signal['Time (s)'][[end_trough]], signal['Mean_norm'][[end_trough]], color='green', label='End of peak')
        plt.scatter(signal['Time (s)'][pretrough], signal['Mean_norm'][pretrough], color='blue', label='Start of peak')
        for p, color in zip(apd_percents, colors): # add APD markers
            # Find APD points in this beat range
            apd_off_x_array = np.array(apd_off_x[f'apd{p}off_x'])
            apd_off_y_array = np.array(apd_off_y[f'apd{p}off_y'])
            mask = (apd_off_x_array >= signal['Time (s)'][pretrough]) & (apd_off_x_array <= signal['Time (s)'][end_trough])
            if any(mask):
                plt.scatter(apd_off_x_array[mask], apd_off_y_array[mask], color=color, label=f'APD{p}', alpha=0.7, marker='o')
            apd_on_x_array = np.array(apd_on_x[f'apd{p}on_x'])
            apd_on_y_array = np.array(apd_on_y[f'apd{p}on_y'])
            mask_on = (apd_on_x_array >= signal['Time (s)'][pretrough]) & (apd_on_x_array <= signal['Time (s)'][end_trough])
            if any(mask_on):
                plt.scatter(apd_on_x_array[mask_on], apd_on_y_array[mask_on], color=color, alpha=0.7, marker='d')
        plt.title('Single Beat with APD Values')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F0 (a.u.)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{folder}/Results/{condition}/{name}_singlebeat_plot.png', facecolor = 'w', bbox_inches="tight", pad_inches=0.3)
        plt.show()


        ''' 8. Save signal and APD values '''
        choice = input("Save this signal? Press Enter to save, or type anything else to skip: ")
        if choice == '':
            # add param means to results_dict
            results_dict = {
                'Name': name,
                'Period (s)': period,
                'Frequence (Hz)': 1/period,
                'Interpeak std (s)': interpeak_std,
                'Action potential (s)': action_potential,
                'Rise time (s)': rise_time,
                'Max rise slope (a.u./s)': rise_max_slope,
                'Amplitude (a.u.)': mean_amplitude,
                'Amplitude std (a.u.)': std_amplitude,
                'Triangulation (apd50/apd90)': np.mean(apd['apd50'])/np.mean(apd['apd90'])}
            for p in apd_percents:  # add APD metrics using apd_percents list
                results_dict[f'APD{p} mean (s)'] = np.mean(apd[f'apd{p}'])
                results_dict[f'APD{p}off mean (s)'] = np.mean(apd_off[f'apd{p}off'])

            results_allrings.iloc[f] = pd.Series(results_dict)  # add results dict as a line of results_allrings dataframe without using append

            apds = pd.DataFrame({**apd, **apd_off, **apd_off_x, **apd_off_y, **apd_on_x, **apd_on_y})
            apds.insert(0, 'Name', name)
            apds.insert(1, 'Pretroughs', pretroughs)
            apds.insert(2, 'Pretroughs_x', np.array(signal['Time (s)'][pretroughs]))
            apds.insert(3, 'Pretroughs_y', np.array(signal['Mean_norm'][pretroughs]))
            apds.insert(4, 'Peaks', peaks)
            apds.insert(5, 'Peaks_x', np.array(signal['Time (s)'][peaks]))
            apds.insert(6, 'Peaks_y', np.array(signal['Mean_norm'][peaks]))
            apds.insert(7, 'Troughs', troughs)
            apds.insert(8, 'Troughs_x', np.array(signal['Time (s)'][troughs]))
            apds.insert(9, 'Troughs_y', np.array(signal['Mean_norm'][troughs]))
            apds.to_csv(f'{folder}/Results/{condition}/{name}_apd.csv', index=False) # save apd as CSVs

            signal_to_save = signal[['Time (s)', 'Mean', 'Mean_norm']]
            signal_to_save.columns = ['Time (s)', 'Mean', 'Mean_norm_smooth (a.u.)'] 
            signal_to_save.insert(0, 'Name', name)
            signal_to_save.to_csv(f'{folder}/Results/{condition}/{name}_signal.csv', index=False) # save the signal as CSVs

    # Save params as one csv for whole condition
    results_allrings.to_csv(f'{folder}/Results/{condition}/{name_condition}_param_means.csv', index=False)

    print('All signals from condition '+condition+' processed!\n\n\n')

print('All conditions processed!')