"""
Complete Cardiac Contractility Analysis Pipeline
Combines: Video Review → CSV Filtering → Drug Splitting → DMSO Normalization → Statistics & Plots
"""

import os
import shutil
import cv2
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tkinter import filedialog, Tk, simpledialog
from scipy.stats import shapiro, wilcoxon, sem
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: VIDEO REVIEW AND CSV FILTERING (First Code)
# ============================================================================

def select_folder(title="Select output folder"):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def get_corresponding_csv(video_path, csv_root):
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    for file in os.listdir(csv_root):
        if file.startswith(prefix) and file.endswith("_analysis_aggregated.csv"):
            return os.path.join(csv_root, file)
    return None

def review_video(video_path):
    print(f"\n🎬 Reviewing {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    bad_rings = set()

    print("➡️ Controls:")
    print("   - Press A–I to toggle bad rings.")
    print("   - ENTER to confirm selection.")
    print("   - ESC to skip this video.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        overlay = frame.copy()
        text = f"Bad rings: {', '.join(sorted(bad_rings)) if bad_rings else 'None'}"
        cv2.putText(overlay, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        cv2.imshow("Video Review", overlay)

        key = cv2.waitKey(60) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            bad_rings.clear()
            break
        elif 97 <= key <= 105:  # keys a–i
            ring = chr(key).upper()
            if ring in bad_rings:
                bad_rings.remove(ring)
            else:
                bad_rings.add(ring)

    cap.release()
    cv2.destroyAllWindows()
    return bad_rings

def filter_csv(csv_path, bad_rings, filtered_root):
    df = pd.read_csv(csv_path)
    filtered_df = df[~df["Ring ID"].isin(bad_rings)]

    base_name = os.path.basename(csv_path).replace(".csv", "_filtered.csv")
    filtered_path = os.path.join(filtered_root, base_name)

    filtered_df.to_csv(filtered_path, index=False)

    removed = len(df) - len(filtered_df)
    print(f"✅ {base_name} → removed {removed} rows ({len(filtered_df)} kept)")

    return filtered_path, removed

def copy_unfiltered_to_filtered(csv_path, filtered_root):
    base_name = os.path.basename(csv_path).replace(".csv", "_filtered.csv")
    destination = os.path.join(filtered_root, base_name)
    shutil.copy(csv_path, destination)
    print(f"👌 No bad rings → copied unmodified CSV to {destination}")
    return destination

def create_global_filtered_csv(filtered_folder, removed_total):
    output_name = f"global_results_summary_filtered_REMOVED_{removed_total}.csv"

    filtered_files = [
        os.path.join(filtered_folder, f)
        for f in os.listdir(filtered_folder)
        if f.endswith("_analysis_aggregated_filtered.csv")
    ]

    if not filtered_files:
        print("⚠️ No filtered CSVs found to create global summary.")
        return None

    dfs = [pd.read_csv(f) for f in filtered_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_folder = os.path.join(filtered_folder, "combined")
    os.makedirs(combined_folder, exist_ok=True)

    output_path = os.path.join(combined_folder, output_name)
    combined_df.to_csv(output_path, index=False)

    print(f"\n🎉 Combined global summary saved: {output_path}")
    print(f"📊 Total removed rings across all videos: {removed_total}")

    return output_path

def step1_video_review_and_filter(folder):
    """Step 1: Review videos and filter CSVs"""
    print("\n" + "="*80)
    print("STEP 1: VIDEO REVIEW AND CSV FILTERING")
    print("="*80)
    
    videos_folder = os.path.join(folder, "videos")
    csv_folder = os.path.join(folder, "CSV")

    if not os.path.exists(videos_folder) or not os.path.exists(csv_folder):
        print("❌ Invalid folder structure. Expecting 'videos/' and 'CSV/'.")
        return None

    filtered_csv_folder = os.path.join(folder, "CSV_filtered")
    os.makedirs(filtered_csv_folder, exist_ok=True)

    videos = [os.path.join(videos_folder, v) for v in os.listdir(videos_folder) if v.endswith(".mp4")]
    if not videos:
        print("❌ No .mp4 videos found.")
        return None

    total_removed_rings = 0

    for video_path in videos:
        csv_path = get_corresponding_csv(video_path, csv_folder)
        if not csv_path:
            print(f"⚠️ No matching CSV found for {os.path.basename(video_path)}")
            continue

        bad_rings = review_video(video_path)

        if bad_rings:
            _, removed = filter_csv(csv_path, bad_rings, filtered_csv_folder)
            total_removed_rings += removed
        else:
            copy_unfiltered_to_filtered(csv_path, filtered_csv_folder)

    global_csv_path = create_global_filtered_csv(filtered_csv_folder, total_removed_rings)
    
    return global_csv_path

# ============================================================================
# STEP 2: NORMALIZE BY DMSO (Second Code) - MODIFIED
# ============================================================================

def normalize_and_average_contractility(file_path, output_folder=None):
    """Step 2: Normalize global CSV by DMSO controls"""
    
    print(f"\n{'='*60}")
    print(f"Normalizing: {Path(file_path).name}")
    print(f"{'='*60}")
    
    # Convert CSV to XLSX if needed
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        new_path = os.path.splitext(file_path)[0] + ".xlsx"
        df.to_excel(new_path, index=False)
        file_path = new_path
    
    data = pd.read_excel(file_path)
    
    if output_folder is None:
        output_folder = os.path.dirname(file_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    name_column = next((col for col in data.columns if col.startswith('Video')), None)
    if name_column is None:
        raise ValueError("No column starting with 'Video name' found")
    
    names = data[name_column].values.flatten()
    wells = []
    concentrations = []
    unique_concentrations = []
    
    for name in names:
        C = name.split('_')[-4]
        W = name.split('_')[-3]
        conc_value = C[-1]
        wells.append(W)
        
        if 'DMSO' in name:
            concentrations.append(-1)
        elif 'Basal' in name:
            concentrations.append(0)
        else:
            concentrations.append(conc_value)
            if conc_value not in unique_concentrations:
                unique_concentrations.append(conc_value)
    
    RingID = next((col for col in data.columns if col.startswith('Ring ID')), None)
    data['Concentration'] = concentrations
    data['Well'] = wells
    if RingID != None:
        data['Tissue'] = wells + data[RingID]
    else:
        data['Tissue'] = wells
    
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
    
    if len(tissues_with_complete_data) == 0:
        raise ValueError("No tissues found with both DMSO control and drug concentration data")
    
    filtered_data = data[data['Tissue'].isin(tissues_with_complete_data)].copy()
    
    exclude_cols = ['Total video time (s)', "Initial Young's Modulus", 'Days in culture', 
                   'Well', 'Concentration', name_column, RingID, 'Resize Factor X', 'Resize Factor Y']
    
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    normalize_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    normalized_data = filtered_data.copy()
    
    for tissue in tissues_with_complete_data:
        dmso_rows = (normalized_data['Tissue'] == tissue) & (normalized_data['Concentration'] == -1)
        tissue_rows = normalized_data['Tissue'] == tissue
        
        for col in normalize_columns:
            dmso_mean = normalized_data.loc[dmso_rows, col].mean()
            
            if dmso_mean != 0 and not np.isnan(dmso_mean):
                normalized_data.loc[tissue_rows, col] = normalized_data.loc[tissue_rows, col] / dmso_mean
            else:
                normalized_data.loc[tissue_rows, col] = np.nan
    
    # MODIFICATION: Only create the normalized output, not the MeanSDN CSV
    input_filename = os.path.splitext(os.path.basename(file_path))[0]
    normalized_output = os.path.join(output_folder, f'{input_filename}_NormalizedDMSO.xlsx')
    
    normalized_data.to_excel(normalized_output, index=False)
    
    print(f"✓ Normalized data saved to: {normalized_output}")
    
    return normalized_output

# ============================================================================
# STEP 3: SPLIT BY DRUG (Third Code) - MODIFIED
# ============================================================================

def extract_drug_name(video_name):
    """Extract drug name from video name"""
    parts = video_name.split('_')
    if len(parts) >= 4:
        drug_field = parts[-4]
        match = re.match(r'^([A-Za-z]+)', drug_field)
        if match:
            return match.group(1)
    return None

def step3_split_by_drug(xlsx_path, global_csv_path):
    """Step 3: Split both normalized and non-normalized data by drug"""
    print("\n" + "="*80)
    print("STEP 3: SPLITTING DATA BY DRUG")
    print("="*80)
    
    # Load normalized data
    norm_df = pd.read_excel(xlsx_path)
    norm_df['Drug'] = norm_df['Video name'].apply(extract_drug_name)
    
    # Load non-normalized data from global CSV
    global_df = pd.read_csv(global_csv_path)
    global_df['Drug'] = global_df['Video name'].apply(extract_drug_name)
    
    # Get unique drugs (excluding DMSO)
    drugs = [d for d in norm_df['Drug'].dropna().unique() if d.upper() != 'DMSO']
    
    input_dir = Path(xlsx_path).parent
    output_dir = input_dir / "by_drug"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output folder: {output_dir}")
    print(f"Found {len(drugs)} unique drugs: {', '.join(drugs)}")
    
    # Get DMSO rows for both datasets
    def get_dmso_rows(df):
        dmso_mask = df['Video name'].str.contains(r'DMSO\d*', case=False, na=False, regex=True)
        return df[dmso_mask].copy()
    
    norm_dmso_rows = get_dmso_rows(norm_df)
    global_dmso_rows = get_dmso_rows(global_df)
    
    print(f"Found {len(norm_dmso_rows)} DMSO control rows")
    
    drug_files = []
    for drug in drugs:
        # Create drug folder
        drug_folder = output_dir / drug
        drug_folder.mkdir(exist_ok=True)
        
        drug_pattern = rf'{drug}\d*'
        
        # Process normalized data
        norm_drug_mask = norm_df['Video name'].str.contains(drug_pattern, case=False, na=False, regex=True)
        norm_drug_rows = norm_df[norm_drug_mask].copy()
        norm_filtered_df = pd.concat([norm_dmso_rows, norm_drug_rows], ignore_index=True)
        norm_filtered_df = norm_filtered_df.drop('Drug', axis=1)
        
        # Process non-normalized data
        global_drug_mask = global_df['Video name'].str.contains(drug_pattern, case=False, na=False, regex=True)
        global_drug_rows = global_df[global_drug_mask].copy()
        global_filtered_df = pd.concat([global_dmso_rows, global_drug_rows], ignore_index=True)
        global_filtered_df = global_filtered_df.drop('Drug', axis=1)
        
        if not norm_filtered_df.empty and len(norm_drug_rows) > 0:
            # Save normalized versions
            norm_output = drug_folder / f"{drug}_NormalizedDMSO.xlsx"
            norm_filtered_df.to_excel(norm_output, index=False)
            
            # Save non-normalized version
            non_norm_output = drug_folder / f"{drug}_non_normalised.xlsx"
            global_filtered_df.to_excel(non_norm_output, index=False)
            
            drug_files.append({
                'drug': drug,
                'normalized': norm_output,
                'non_normalized': non_norm_output,
                'folder': drug_folder
            })
            
            print(f"Created for {drug}:")
            print(f"  - {norm_output.name} → {len(norm_filtered_df)} rows "
                  f"({len(norm_drug_rows)} drug + {len(norm_dmso_rows)} DMSO)")
            print(f"  - {non_norm_output.name} → {len(global_filtered_df)} rows "
                  f"({len(global_drug_rows)} drug + {len(global_dmso_rows)} DMSO)")
            print(f"  - Files saved in: {drug_folder}")
    
    print(f"\n✓ All drug files saved to: {output_dir}")
    return drug_files

# ============================================================================
# STEP 4: STATISTICS AND PLOTS (Fourth Code) - MODIFIED
# ============================================================================

def extract_concentration_from_video_name(video_name):
    """Extract concentration from video name"""
    if 'DMSO' in video_name:
        return -1
    
    parts = video_name.split('_')
    if len(parts) >= 4:
        drug_field = parts[-4]
        match = re.search(r'\d+', drug_field)
        if match:
            return int(match.group())
    return 0

def get_concentration_mapping(drug_name, conc_labels):
    """Ask user for actual concentration values for each concentration label"""
    print(f"\nPlease enter actual concentration values for {drug_name}:")
    print(f"Concentration labels found: {conc_labels}")
    
    concentration_map = {}
    for conc_label in conc_labels:
        # Skip DMSO (Conc_-1)
        if conc_label == "-1":
            concentration_map[conc_label] = "DMSO"
            continue
            
        # Ask for actual concentration value
        while True:
            try:
                value = simpledialog.askstring(
                    "Concentration Input",
                    f"Enter actual concentration for '{conc_label}' in {drug_name} (e.g., 0.1, 1, 10):\n"
                    f"Leave empty for '{conc_label}', type 'skip' to skip this concentration:"
                )
                
                if value is None:  # User cancelled
                    return None
                elif value.lower() == 'skip':
                    print(f"  Skipping concentration {conc_label}")
                    concentration_map[conc_label] = None
                    break
                elif value == '':
                    # Use the label itself
                    concentration_map[conc_label] = conc_label
                    print(f"  Using label '{conc_label}' for concentration")
                    break
                else:
                    # Try to convert to float
                    try:
                        float_val = float(value)
                        concentration_map[conc_label] = value
                        print(f"  Set concentration {conc_label} = {value}")
                        break
                    except ValueError:
                        # If not a number, use as is
                        concentration_map[conc_label] = value
                        print(f"  Set concentration {conc_label} = '{value}'")
                        break
            except Exception as e:
                print(f"Error getting concentration: {e}")
                concentration_map[conc_label] = conc_label
    
    return concentration_map

def create_plots_and_stats(reorganized_file_path, drug_name, drug_folder):
    """Create plots and enhanced statistics for a reorganized file"""
    print(f"\n{'='*80}")
    print(f"Creating plots for: {drug_name}")
    print(f"{'='*80}")
    
    all_sheets = pd.read_excel(reorganized_file_path, sheet_name=None)
    print(f"Found {len(all_sheets)} sheets (parameters) to plot")
    
    # Get concentration mapping from user
    # First, extract all concentration labels from the data
    sample_sheet = list(all_sheets.values())[0]
    conc_cols = [c for c in sample_sheet.columns if c.startswith("Conc_") and c != "Conc_-1"]
    conc_labels = [c.replace("Conc_", "") for c in conc_cols]
    
    # Add DMSO label
    all_conc_labels = ["-1"] + conc_labels
    
    # Ask user for concentration mapping
    root = Tk()
    root.withdraw()
    concentration_map = get_concentration_mapping(drug_name, all_conc_labels)
    root.destroy()
    
    if concentration_map is None:
        print("User cancelled concentration input. Skipping plots for this drug.")
        return None, None
    
    # Create Wilcoxon stats file in drug folder
    stats_output = drug_folder / "Wilcoxon_Stats.xlsx"
    writer = pd.ExcelWriter(stats_output, engine='openpyxl')
    
    for sheet_name, df in all_sheets.items():
        print(f"\n  Processing: {sheet_name}")
        
        required_cols = ["Conc_-1"]
        conc_cols = [c for c in df.columns if c.startswith("Conc_") and c != "Conc_-1"]
        
        if "Conc_-1" not in df.columns or len(conc_cols) == 0:
            print(f"    ⚠ SKIPPED: Missing baseline or concentration columns")
            continue
        
        baseline = df["Conc_-1"]
        stats_rows = []
        means = []
        sems_plot = []
        pvals_plot = []
        n_samples = []
        actual_concentrations = []
        
        for conc_col in conc_cols:
            conc_label = conc_col.replace("Conc_", "")
            actual_conc = concentration_map.get(conc_label, conc_label)
            
            if actual_conc is None:
                print(f"    Skipping {conc_col} (user requested skip)")
                continue
                
            paired_data = pd.DataFrame({
                "baseline": df["Conc_-1"],
                "test": df[conc_col]
            }).dropna()
            
            x = paired_data["baseline"].values
            y = paired_data["test"].values
            n = len(paired_data)
            
            if n < 2:
                stats_rows.append([conc_col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, n, ""])
                means.append(np.nan)
                sems_plot.append(np.nan)
                pvals_plot.append(None)
                n_samples.append(n)
                actual_concentrations.append(actual_conc)
                continue
            
            try:
                diff = y - x
                if np.all(diff == 0):
                    stat, pval = np.nan, 1.0
                else:
                    stat, pval = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
                
            except Exception as e:
                print(f"      ⚠ ERROR: {e}")
                stats_rows.append([conc_col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, n, "ERROR"])
                means.append(np.nan)
                sems_plot.append(np.nan)
                pvals_plot.append(None)
                n_samples.append(n)
                actual_concentrations.append(actual_conc)
                continue
            
            valid_idx = x > 0
            if np.sum(valid_idx) == 0:
                mean_fold_change = np.nan
                sem_fc = np.nan
            else:
                fold_changes = y[valid_idx] / x[valid_idx]
                mean_fold_change = np.mean(fold_changes)
                sem_fc = sem(fold_changes)
            
            significance = ("****" if pval < 0.0001 else 
                          "***" if pval < 0.001 else 
                          "**" if pval < 0.01 else 
                          "*" if pval < 0.05 else "")
            
            stats_rows.append([conc_col, "Wilcoxon", stat, pval, 
                             mean_fold_change, sem_fc, n, significance])
            
            means.append(mean_fold_change)
            sems_plot.append(sem_fc)
            pvals_plot.append(pval)
            n_samples.append(n)
            actual_concentrations.append(actual_conc)
        
        if stats_rows:
            df_stats = pd.DataFrame(stats_rows, 
                                   columns=["Concentration", "Test", "Statistic", "p-value",
                                          "Mean_FC", "SEM_FC", "N", "Significance"])
            
            # Add actual concentration column
            df_stats["Actual_Concentration"] = actual_concentrations
            
            excel_sheet_name = str(sheet_name)[:31]
            df_stats.to_excel(writer, sheet_name=excel_sheet_name, index=False)
        
        # Create plot with actual concentrations
        if not all(np.isnan(means)):
            # Filter out skipped concentrations
            valid_indices = [i for i, conc in enumerate(actual_concentrations) if conc is not None]
            if not valid_indices:
                continue
                
            filtered_means = [means[i] for i in valid_indices]
            filtered_sems = [sems_plot[i] for i in valid_indices]
            filtered_pvals = [pvals_plot[i] for i in valid_indices]
            filtered_ns = [n_samples[i] for i in valid_indices]
            filtered_concs = [actual_concentrations[i] for i in valid_indices]
            
            means_arr = np.array(filtered_means)
            sems_arr = np.array(filtered_sems)
            yerr = sems_arr
            
            x_pos = np.arange(len(filtered_concs))
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(x_pos, means_arr, yerr=yerr, fmt='o-', capsize=6, 
                        linewidth=2.5, markersize=9, color='steelblue', 
                        ecolor='gray', alpha=0.9)
            
            valid_means = means_arr[~np.isnan(means_arr)]
            if len(valid_means) > 0:
                valid_sems = sems_arr[~np.isnan(means_arr)]
                ymax = np.max(valid_means + valid_sems)
                ymin = np.min(valid_means - valid_sems)
                y_range = ymax - ymin if ymax != ymin else abs(ymax) if ymax != 0 else 1
            else:
                ymax, ymin, y_range = 1, 0, 1
            
            for i, p in enumerate(filtered_pvals):
                if p is not None and p < 0.05 and not np.isnan(means_arr[i]):
                    stars = ("****" if p < 0.0001 else 
                           "***" if p < 0.001 else 
                           "**" if p < 0.01 else "*")
                    y_offset = 0.08 * y_range
                    plt.text(x_pos[i], means_arr[i] + sems_arr[i] + y_offset, stars,
                           ha="center", fontsize=16, fontweight='bold', color='red')
            
            for i, n in enumerate(filtered_ns):
                if n > 0:
                    plt.text(x_pos[i], ymin - 0.12 * y_range, f"n={n}",
                           ha="center", fontsize=9, color='dimgray')
            
            plt.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.4)
            
            # Use actual concentrations as x-axis labels
            plt.xticks(x_pos, filtered_concs, fontsize=11)
            plt.xlabel("Concentration", fontsize=12, fontweight='bold')
            plt.ylabel(f"Fold Change {sheet_name}\n(vs DMSO)", fontsize=12, fontweight='bold')
            plt.title(f"{drug_name} - {sheet_name}\nWilcoxon Paired Test: DMSO vs Drug", 
                     fontsize=13, fontweight='bold', pad=15)
            plt.grid(alpha=0.3, linestyle=':', linewidth=0.5)
            plt.tight_layout()
            
            plot_filename = f"{str(sheet_name).replace(' ', '_').replace('/', '_')}.png"
            plot_path = drug_folder / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    writer.close()
    print(f"\n✓ Wilcoxon stats saved to: {stats_output}")
    print(f"✓ All plots saved in: {drug_folder}")
    return drug_folder, stats_output

def process_single_normalized_file(file_info):
    """Process a single normalized XLSX file and generate reorganized and stats files"""
    file_path = file_info['normalized']
    drug_name = file_info['drug']
    drug_folder = file_info['folder']
    
    print(f"\n{'='*80}")
    print(f"Processing: {drug_name}")
    print(f"{'='*80}")

    aggregated_data = pd.read_excel(file_path)

    if 'Concentration' not in aggregated_data.columns:
        aggregated_data['Concentration'] = aggregated_data['Video name'].apply(extract_concentration_from_video_name)

    UniqueTissues = aggregated_data['Tissue'].unique()
    Conc = sorted(aggregated_data['Concentration'].unique())

    metadata_columns = ['Video name', 'Ring ID', 'Resize Factor X', 'Resize Factor Y', 
                       'Total video time (s)', 'Days in culture',
                       'Internal radius (pixels)', 'External radius (pixels)',
                       'Concentration', 'Well', 'Tissue']

    Aggregated_Columns = [col for col in aggregated_data.columns if col not in metadata_columns]

    # Save reorganized file in drug folder
    output_path = drug_folder / f"{drug_name}_NormalizedDMSO_reorganized.xlsx"

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for col in Aggregated_Columns:
            Sheet_columns = ['Tissue']
            for c in Conc:
                Sheet_columns.append(f'Conc_{c}')

            rows_data = []
            for tissue in UniqueTissues:
                tissue_data = aggregated_data[aggregated_data['Tissue'] == tissue]
                
                if len(tissue_data) < 1:
                    continue

                row_data = {'Tissue': tissue}

                for c in Conc:
                    conc_data = tissue_data[tissue_data['Concentration'] == c]
                    if len(conc_data) > 0:
                        param_value = conc_data[col].iloc[0]
                    else:
                        param_value = np.nan
                    row_data[f'Conc_{c}'] = param_value

                has_dmso = not pd.isna(row_data.get('Conc_-1', np.nan))
                
                has_other_conc = False
                for c in Conc:
                    if c != -1:
                        if not pd.isna(row_data.get(f'Conc_{c}', np.nan)):
                            has_other_conc = True
                            break
                
                if has_dmso and has_other_conc:
                    rows_data.append(row_data)

            Parameter_sheet = pd.DataFrame(rows_data)

            sheet_name = col.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('*', '_')
            sheet_name = sheet_name.replace('[', '_').replace(']', '_').replace(':', '_')
            sheet_name = sheet_name.replace('?', '_').replace('|', '_').replace('(', '_').replace(')', '_')[:31]

            Parameter_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Reorganized data saved to: {output_path}")

    # Create plots and stats in drug folder
    result = create_plots_and_stats(output_path, drug_name, drug_folder)
    if result is None:
        print(f"⚠️ Skipping plots and stats for {drug_name} due to user cancellation")
        return file_info
    
    drug_folder, stats_output = result
    file_info['reorganized'] = output_path
    return file_info

def create_combined_wilcoxon_stats(drug_files, output_dir):
    """Create combined Wilcoxon stats file with data from all drugs concatenated laterally"""
    print(f"\n{'='*80}")
    print("CREATING COMBINED WILCOXON STATS FILE")
    print(f"{'='*80}")
    
    # Get all stats files
    stats_files = []
    for file_info in drug_files:
        if 'folder' in file_info:
            stats_file = file_info['folder'] / "Wilcoxon_Stats.xlsx"
            if stats_file.exists():
                stats_files.append((file_info['drug'], stats_file))
    
    if not stats_files:
        print("No Wilcoxon stats files found to combine")
        return None
    
    # Read all stats files
    all_stats = {}
    for drug_name, stats_file in stats_files:
        drug_stats = pd.read_excel(stats_file, sheet_name=None)
        for sheet_name, df in drug_stats.items():
            if sheet_name not in all_stats:
                all_stats[sheet_name] = []
            all_stats[sheet_name].append((drug_name, df))
    
    # Create combined workbook
    combined_output = output_dir / "Combined_Wilcoxon_Stats.xlsx"
    
    # Use context manager for ExcelWriter
    with pd.ExcelWriter(combined_output, engine='openpyxl') as writer:
        for sheet_name, drug_dfs in all_stats.items():
            print(f"\nProcessing sheet: {sheet_name}")
            
            # Start with an empty dataframe
            combined_df = pd.DataFrame()
            
            # Keep track of actual concentrations for each drug
            drug_concentrations = {}
            
            for drug_name, df in drug_dfs:
                # Filter out rows with errors
                df_clean = df[df['Significance'] != 'ERROR'].copy()
                
                # Extract actual concentrations if available
                if 'Actual_Concentration' in df_clean.columns:
                    # Create mapping of concentration labels to actual values
                    conc_mapping = {}
                    for _, row in df_clean.iterrows():
                        conc_label = row['Concentration'].replace('Conc_', '')
                        actual_conc = row['Actual_Concentration']
                        conc_mapping[conc_label] = actual_conc
                    drug_concentrations[drug_name] = conc_mapping
                
                # Add drug name prefix to columns - EXCLUDE Significance and Actual_Concentration columns
                drug_prefix = f"{drug_name}_"
                # Only include Mean_FC, SEM_FC, and N columns (not Significance, Statistic, p-value, Test)
                df_clean = df_clean[['Concentration', 'Mean_FC', 'SEM_FC', 'N']]
                df_clean.columns = [drug_prefix + col if col != 'Concentration' else col for col in df_clean.columns]
                
                # Merge dataframes
                if combined_df.empty:
                    combined_df = df_clean
                else:
                    # Ensure we have all concentrations
                    all_concs = set(combined_df['Concentration']).union(set(df_clean['Concentration']))
                    combined_df = pd.merge(combined_df, df_clean, on='Concentration', how='outer')
            
            # Sort by concentration
            combined_df['Concentration'] = combined_df['Concentration'].astype(str)
            conc_order = []
            for conc in combined_df['Concentration']:
                if conc == 'Conc_-1':
                    conc_order.append(-2)  # Place before -1
                elif conc.startswith('Conc_'):
                    try:
                        conc_order.append(int(conc.replace('Conc_', '')))
                    except ValueError:
                        conc_order.append(999)  # Place at end
                else:
                    conc_order.append(999)
            
            combined_df['sort_order'] = conc_order
            combined_df = combined_df.sort_values('sort_order')
            combined_df = combined_df.drop('sort_order', axis=1)
            
            # Add actual concentration mapping table below the main table
            # Create a DataFrame for actual concentrations
            all_concentrations = sorted(set(combined_df['Concentration']))
            actual_conc_df = pd.DataFrame({'Concentration': all_concentrations})
            
            # Add actual concentration for each drug
            for drug_name in drug_concentrations:
                conc_mapping = drug_concentrations[drug_name]
                actual_values = []
                for conc in all_concentrations:
                    conc_label = conc.replace('Conc_', '')
                    actual_values.append(conc_mapping.get(conc_label, conc_label))
                actual_conc_df[f'{drug_name}_Actual_Conc'] = actual_values
            
            # Combine the main table and actual concentration table
            # First, write the main table
            sheet_name_trunc = str(sheet_name)[:31]
            combined_df.to_excel(writer, sheet_name=sheet_name_trunc, index=False)
            
            # Then write the actual concentration table starting 2 rows below the main table
            start_row = len(combined_df) + 3  # Leave 2 empty rows
            actual_conc_df.to_excel(writer, sheet_name=sheet_name_trunc, 
                                   startrow=start_row, index=False)
            
            print(f"  Added data for {len(drug_dfs)} drugs, {len(combined_df)} concentrations")
            print(f"  Added actual concentration table with {len(actual_conc_df)} rows")
    
    print(f"\n✓ Combined Wilcoxon stats saved to: {combined_output}")
    return combined_output

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline combining all steps"""
    
    print("\n" + "="*80)
    print("CARDIAC CONTRACTILITY ANALYSIS - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Review videos and filter CSVs by bad rings")
    print("  2. Normalize filtered data by DMSO controls")
    print("  3. Split normalized and non-normalized data by drug")
    print("  4. Generate statistics and plots for each drug (will ask for concentration values)")
    print("  5. Create combined Wilcoxon stats file with actual concentrations")
    print("="*80)
    
    # Select main folder
    folder = select_folder("Select the output folder containing videos/ and CSV/")
    if not folder:
        print("No folder selected. Exiting.")
        return
    
    # STEP 1: Video review and filtering
    global_csv_path = step1_video_review_and_filter(folder)
    if not global_csv_path:
        print("❌ Step 1 failed. Exiting.")
        return
    
    # STEP 2: Normalize the global CSV by DMSO
    print("\n" + "="*80)
    print("STEP 2: NORMALIZING GLOBAL CSV BY DMSO")
    print("="*80)
    
    try:
        # MODIFIED: Don't create MeanSDN CSV
        normalized_output = normalize_and_average_contractility(global_csv_path)
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 3: Split both normalized and non-normalized data by drug
    print("\n" + "="*80)
    print("STEP 3: SPLITTING DATA BY DRUG")
    print("="*80)
    
    drug_files_info = step3_split_by_drug(normalized_output, global_csv_path)
    if not drug_files_info:
        print("❌ Step 3 failed. No drug files created. Exiting.")
        return
    
    print("\n" + "="*80)
    print("STEP 4: ANALYZING EACH DRUG")
    print("="*80)
    print("NOTE: You will be asked to provide actual concentration values for each drug.")
    print("For each concentration label (like '1', '2', '3'), enter the actual concentration")
    print("(e.g., '0.1', '1', '10' μM). Leave empty to use the label itself.")
    print("="*80)
    
    # STEP 4: Analyze each drug file
    processed_drugs = []
    for file_info in drug_files_info:
        try:
            print(f"\n{'='*80}")
            print(f"Processing drug: {file_info['drug']}")
            print(f"{'='*80}")
            
            # Create statistics and plots
            result = process_single_normalized_file(file_info)
            if result:  # Only append if not skipped
                processed_drugs.append(result)
            
        except Exception as e:
            print(f"❌ ERROR processing {file_info['drug']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # STEP 5: Create combined Wilcoxon stats
    print("\n" + "="*80)
    print("STEP 5: CREATING COMBINED STATISTICS")
    print("="*80)
    
    output_dir = Path(normalized_output).parent / "by_drug"
    combined_stats = create_combined_wilcoxon_stats(processed_drugs, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nProcessed {len(processed_drugs)} drugs successfully:")
    
    for result in processed_drugs:
        print(f"\n  ✓ {result['drug']}:")
        print(f"    - Folder: {result['folder']}")
        print(f"    - Normalized: {result['normalized'].name}")
        print(f"    - Non-normalized: {result.get('non_normalized', 'Not found').name if 'non_normalized' in result else 'Not found'}")
        print(f"    - Reorganized: {result.get('reorganized', 'Not found').name if 'reorganized' in result else 'Not found'}")
        print(f"    - Plots and stats saved in drug folder")
    
    if combined_stats:
        print(f"\n  ✓ Combined Wilcoxon stats: {combined_stats.name}")
        print(f"    - Includes actual concentration table at the bottom of each sheet")
    
    print("\n" + "="*80)
    print("All analysis complete! Check the output folders for results.")
    print("="*80)

if __name__ == "__main__":
    main()