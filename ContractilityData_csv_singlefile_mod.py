import pandas as pd
import numpy as np
import os
from tkinter import filedialog

def normalize_and_average_contractility(file_path, output_folder=None):
    """
    Normalize and average cardiac contractility dose response data from a single CSV file.
    Each tissue is normalized by its own DMSO control value.
    
    Parameters:
    file_path (str): Path to the input CSV file
    output_folder (str): Optional output folder path. If None, saves in same directory as input file.
    
    Returns:
    tuple: (normalized_data, averaged_data) - DataFrames with normalized and averaged results
    """
    
    # Load the data
    data = pd.read_excel(file_path)
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.dirname(file_path)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find the name column (assumes it starts with 'Name')
    name_column = next((col for col in data.columns if col.startswith('Name')), None)
    if name_column is None:
        raise ValueError("No column starting with 'Video name' found in the data")
    
    # Extract well and concentration information from names (matching original code logic)
    names = data[name_column].values.flatten()
    wells = []
    concentrations = []
    unique_concentrations = []
    
    # Extract wells and concentrations following the original code pattern
    for name in names:
        # Extract concentration and well following original logic
        C = name.split('_')[-4]  # Concentration part
        W = name.split('_')[-3]  # Well part


        Well=W.split('.')[0] #remove '.mp4'
        
        conc_value = C[-1]  # Last character of concentration part
        wells.append(Well)
        
        # Handle DMSO vs drug concentrations
        if 'DMSO' in name:
            concentrations.append(-1)
        elif 'Basal' in name:
            concentrations.append(0)
        else:
            concentrations.append(conc_value)
            if conc_value not in unique_concentrations:
                unique_concentrations.append(conc_value)
    
    # Add well and concentration columns to the dataframe
    RingID=next((col for col in data.columns if col.startswith('Ring ID')), None)
    data['Concentration'] = concentrations
    data['Well'] = wells
    if RingID !=None:
        data['Tissue']=wells+data[RingID]
    else:
        data['Tissue']=wells
    

    # Identify tissues that have both DMSO and drug data
    unique_tissues = data['Tissue'].unique()
    tissues_with_complete_data = []
    dmso_only_tissues = []
    drug_only_tissues = []
    
    for tissue in unique_tissues:
        tissue_data = data[data['Tissue'] == tissue]
        has_dmso = (tissue_data['Concentration'] == -1).any()
        has_drug = (tissue_data['Concentration'] != -1).any()
        
        if has_dmso and has_drug:
            tissues_with_complete_data.append(tissue)
        elif has_dmso and not has_drug:
            dmso_only_tissues.append(tissue)
        elif not has_dmso and has_drug:
            drug_only_tissues.append(tissue)
    
    print(f"Tissues with both DMSO and drug data: {len(tissues_with_complete_data)}")
    print(f"Tissues with only DMSO (will be removed): {len(dmso_only_tissues)}")
    print(f"Tissues with only drug data (will be removed): {len(drug_only_tissues)}")
    
    if dmso_only_tissues:
        print(f"Removing DMSO-only tissues: {dmso_only_tissues}")
    if drug_only_tissues:
        print(f"Removing drug-only tissues: {drug_only_tissues}")
    
    # Filter data to keep only tissues with both DMSO and drug data
    filtered_data = data[data['Tissue'].isin(tissues_with_complete_data)].copy()
    
    if len(filtered_data) == 0:
        raise ValueError("No tissues found with both DMSO control and drug concentration data")
    



    # Define columns to exclude from normalization
    exclude_cols = ['Total video time', 'SNR', 'Initial Youngs Modulus', 'Days in culture', 
                   'Well', 'Concentration', name_column, RingID]
    
    # Get numeric columns for normalization
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    normalize_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    # Create copy for normalized data
    normalized_data = filtered_data.copy()
    
    # Normalize each remaining well by its DMSO control
    for tissue in tissues_with_complete_data:
        # Get DMSO control for this tissue
        dmso_rows = (normalized_data['Tissue'] == tissue) & (normalized_data['Concentration'] == -1)
        tissue_rows = normalized_data['Tissue'] == tissue
        
        # Calculate DMSO mean for each parameter
        for col in normalize_columns:
            dmso_mean = normalized_data.loc[dmso_rows, col].mean()
            
            if dmso_mean != 0 and not np.isnan(dmso_mean):
                # Normalize all rows for this tissue by the DMSO mean
                normalized_data.loc[tissue_rows, col] = normalized_data.loc[tissue_rows, col] / dmso_mean
            else:
                normalized_data.loc[tissue_rows, col] = np.nan
    
    # Create averaged data table (following original code pattern)
    # Get unique concentrations for averaging
    unique_concs = [c for c in normalized_data['Concentration'].unique() if pd.notna(c)]
    
    # Separate numeric and string concentrations for proper sorting
    numeric_concs = []
    string_concs = []
    
    for c in unique_concs:
        if isinstance(c, (int, float)):
            numeric_concs.append(c)
        else:
            try:
                # Try to convert string to number
                numeric_concs.append(float(c))
            except (ValueError, TypeError):
                string_concs.append(c)
    
    # Sort numeric concentrations and keep string ones separate
    all_unique_concentrations = sorted(numeric_concs) + sorted(string_concs)
    
    # Create columns for mean, SD, and N
    mean_sd_columns = []
    for col in numeric_columns:
        mean_sd_columns.extend([f'Mean_{col}', f'SD_{col}', f'N_{col}'])
    
    averaged_data = pd.DataFrame(columns=['Concentration'] + mean_sd_columns)
    
    # Calculate statistics for each concentration (including DMSO at -1)
    for idx, concentration in enumerate(all_unique_concentrations):
        averaged_data.loc[idx, 'Concentration'] = concentration
        
        conc_rows = normalized_data['Concentration'] == concentration
        
        for col in numeric_columns:
            values = normalized_data.loc[conc_rows, col].dropna()
            
            averaged_data.loc[idx, f'Mean_{col}'] = values.mean() if len(values) > 0 else np.nan
            averaged_data.loc[idx, f'SD_{col}'] = values.std() if len(values) > 0 else np.nan
            averaged_data.loc[idx, f'N_{col}'] = len(values)
    
    # Generate output filenames
    input_filename = os.path.splitext(os.path.basename(file_path))[0]
    normalized_output = os.path.join(output_folder, f'{input_filename}_NormalizedDMSO.csv')
    averaged_output = os.path.join(output_folder, f'{input_filename}_NormalizedDMSO_MeanSDN.csv')
    
    # Save results
    normalized_data.to_csv(normalized_output, index=False)
    averaged_data.to_csv(averaged_output, index=False)
    
    print(f"Normalized data saved to: {normalized_output}")
    print(f"Averaged data saved to: {averaged_output}")
    print(f"Processed {len(tissues_with_complete_data)} tissues with complete data")
    
    return normalized_data, averaged_data


# USE
filename = filedialog.askopenfilename(title="Select a File")
normalized_df, averaged_df = normalize_and_average_contractility(filename)


# If you want to process multiple files in a folder:
def process_multiple_files(folder_path, file_pattern='param_means.csv'):
    """
    Process multiple CSV files in a folder
    """
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and file_pattern in file and 'DMSO' not in file:
            file_path = os.path.join(folder_path, file)
            print(f"\nProcessing: {file}")
            try:
                normalize_and_average_contractility(file_path, f'{folder_path}/Analysis')
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

# Example for batch processing:
# process_multiple_files('/path/to/your/folder')