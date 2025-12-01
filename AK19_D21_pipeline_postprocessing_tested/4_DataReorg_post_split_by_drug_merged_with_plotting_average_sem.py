##fourth##

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon, sem
import statsmodels.stats.multitest as multitest
from pathlib import Path
import matplotlib.pyplot as plt

def extract_concentration_from_video_name(video_name):
    """
    Extract concentration from video name.
    Assumes DMSO is concentration -1 (baseline)
    and other drugs have numbered concentrations
    """
    if 'DMSO' in video_name:
        return -1
    
    parts = video_name.split('_')
    if len(parts) >= 4:
        drug_field = parts[-4]
        import re
        match = re.search(r'\d+', drug_field)
        if match:
            return int(match.group())
    return 0

def select_input_folder():
    """Open folder dialog to select input folder with XLSX files"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory(
        title="Select folder containing drug XLSX files"
    )
    root.destroy()
    return folder_path

def create_plots_and_stats(reorganized_file_path, drug_name):
    """
    Create plots and enhanced statistics for a reorganized file
    """
    print(f"\n{'='*80}")
    print(f"Creating plots for: {drug_name}")
    print(f"{'='*80}")
    
    # Create drug-specific folder for plots
    parent_dir = Path(reorganized_file_path).parent
    drug_folder = parent_dir / drug_name
    drug_folder.mkdir(exist_ok=True)
    
    # Read all sheets from reorganized file
    all_sheets = pd.read_excel(reorganized_file_path, sheet_name=None)
    print(f"Found {len(all_sheets)} sheets (parameters) to plot")
    
    # Create Excel writer for Wilcoxon stats
    stats_output = drug_folder / "Wilcoxon_Stats.xlsx"
    writer = pd.ExcelWriter(stats_output, engine='xlsxwriter')
    
    # Process each sheet (parameter)
    for sheet_name, df in all_sheets.items():
        print(f"\n  Processing: {sheet_name}")
        
        # Check for required columns
        required_cols = ["Conc_-1"]
        conc_cols = [c for c in df.columns if c.startswith("Conc_") and c != "Conc_-1"]
        
        if "Conc_-1" not in df.columns or len(conc_cols) == 0:
            print(f"    ⚠ SKIPPED: Missing baseline or concentration columns")
            continue
        
        print(f"    Rows: {len(df)}")
        print(f"    Concentrations: {conc_cols}")
        
        # Print data availability
        for col in ["Conc_-1"] + conc_cols:
            n_valid = df[col].notna().sum()
            n_missing = df[col].isna().sum()
            print(f"      {col}: {n_valid} valid, {n_missing} missing")
        
        # Calculate statistics
        baseline = df["Conc_-1"]
        stats_rows = []
        means = []
        sems_plot = []
        pvals_plot = []
        n_samples = []
        
        for conc_col in conc_cols:
            # Create paired data (remove rows where either value is missing)
            paired_data = pd.DataFrame({
                "baseline": df["Conc_-1"],
                "test": df[conc_col]
            }).dropna()
            
            x = paired_data["baseline"].values
            y = paired_data["test"].values
            n = len(paired_data)
            
            print(f"\n    {conc_col}:")
            print(f"      Valid pairs: {n}")
            
            if n < 2:
                print(f"      ⚠ SKIPPED (insufficient data)")
                stats_rows.append([conc_col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, n, ""])
                means.append(np.nan)
                sems_plot.append(np.nan)
                pvals_plot.append(None)
                n_samples.append(n)
                continue
            
            # Wilcoxon paired test
            try:
                diff = y - x
                if np.all(diff == 0):
                    print(f"      ⚠ WARNING: All differences are zero")
                    stat, pval = np.nan, 1.0
                else:
                    stat, pval = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
                
                print(f"      Statistic: {stat:.4f}")
                print(f"      p-value: {pval:.6f}")
                
            except Exception as e:
                print(f"      ⚠ ERROR: {e}")
                stats_rows.append([conc_col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, n, "ERROR"])
                means.append(np.nan)
                sems_plot.append(np.nan)
                pvals_plot.append(None)
                n_samples.append(n)
                continue
            
            # Calculate fold change
            valid_idx = x > 0
            if np.sum(valid_idx) == 0:
                print(f"      ⚠ WARNING: No valid baseline for fold change")
                mean_fold_change = np.nan
                sem_fc = np.nan
            else:
                fold_changes = y[valid_idx] / x[valid_idx]
                mean_fold_change = np.mean(fold_changes)
                sem_fc = sem(fold_changes)
            
            print(f"      Mean Fold Change: {mean_fold_change:.4f}")
            print(f"      SEM: {sem_fc:.4f}")
            
            # Significance stars
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
        
        # Save stats to Excel
        if stats_rows:
            df_stats = pd.DataFrame(stats_rows, 
                                   columns=["Concentration", "Test", "Statistic", "p-value",
                                          "Mean_FC", "SEM_FC", "N", "Significance"])
            
            excel_sheet_name = str(sheet_name)[:31]
            df_stats.to_excel(writer, sheet_name=excel_sheet_name, index=False)
            print(f"    ✓ Stats saved to sheet: {excel_sheet_name}")
        
        # Create plot
        if not all(np.isnan(means)):
            means_arr = np.array(means)
            sems_arr = np.array(sems_plot)
            yerr = sems_arr
            
            x_pos = np.arange(len(conc_cols))
            
            plt.figure(figsize=(8, 6))
            plt.errorbar(x_pos, means_arr, yerr=yerr, fmt='o-', capsize=6, 
                        linewidth=2.5, markersize=9, color='steelblue', 
                        ecolor='gray', alpha=0.9)
            
            # Add significance stars
            valid_means = means_arr[~np.isnan(means_arr)]
            if len(valid_means) > 0:
                valid_sems = sems_arr[~np.isnan(means_arr)]
                ymax = np.max(valid_means + valid_sems)
                ymin = np.min(valid_means - valid_sems)
                y_range = ymax - ymin if ymax != ymin else abs(ymax) if ymax != 0 else 1
            else:
                ymax, ymin, y_range = 1, 0, 1
            
            for i, p in enumerate(pvals_plot):
                if p is not None and p < 0.05 and not np.isnan(means_arr[i]):
                    stars = ("****" if p < 0.0001 else 
                           "***" if p < 0.001 else 
                           "**" if p < 0.01 else "*")
                    y_offset = 0.08 * y_range
                    plt.text(x_pos[i], means_arr[i] + sems_arr[i] + y_offset, stars,
                           ha="center", fontsize=16, fontweight='bold', color='red')
            
            # Add sample sizes
            for i, n in enumerate(n_samples):
                if n > 0:
                    plt.text(x_pos[i], ymin - 0.12 * y_range, f"n={n}",
                           ha="center", fontsize=9, color='dimgray')
            
            # Reference line at 1 (no change)
            plt.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.4)
            
            # Labels
            conc_labels = [c.replace("Conc_", "") for c in conc_cols]
            plt.xticks(x_pos, conc_labels, fontsize=11)
            plt.xlabel("Concentration", fontsize=12, fontweight='bold')
            plt.ylabel(f"Fold Change {sheet_name}\n(vs DMSO)", fontsize=12, fontweight='bold')
            plt.title(f"{drug_name} - {sheet_name}\nWilcoxon Paired Test: DMSO vs Drug", 
                     fontsize=13, fontweight='bold', pad=15)
            plt.grid(alpha=0.3, linestyle=':', linewidth=0.5)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{str(sheet_name).replace(' ', '_').replace('/', '_')}.png"
            plot_path = drug_folder / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Plot saved: {plot_filename}")
        else:
            print(f"    ⚠ No plot generated (no valid data)")
    
    writer.close()
    print(f"\n✓ All plots and stats saved in: {drug_folder}")
    return drug_folder

def process_single_file(file_path):
    """Process a single XLSX file and generate reorganized and stats files"""

    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    # Load data
    aggregated_data = pd.read_excel(file_path)

    # Extract concentration if not already present
    if 'Concentration' not in aggregated_data.columns:
        aggregated_data['Concentration'] = aggregated_data['Video name'].apply(extract_concentration_from_video_name)

    # Get unique tissues
    UniqueTissues = aggregated_data['Tissue'].unique()
    print(f"Found {len(UniqueTissues)} unique tissues")

    # Get sorted concentrations
    Conc = sorted(aggregated_data['Concentration'].unique())
    print(f"Concentrations found: {Conc}")

    # Define metadata columns to exclude from analysis
    metadata_columns = ['Video name', 'Ring ID', 'Resize Factor X', 'Resize Factor Y', 
                       'Total video time (s)', 'Days in culture',
                       'Internal radius (pixels)', 'External radius (pixels)',
                       'Concentration', 'Well', 'Tissue']

    # Get all parameter columns to analyze
    Aggregated_Columns = [col for col in aggregated_data.columns if col not in metadata_columns]
    print(f"Parameters to analyze: {len(Aggregated_Columns)}")

    # Output files in same folder
    parent_dir = Path(file_path).parent
    drug_name = Path(file_path).stem
    output_path = parent_dir / (drug_name + '_reorganized.xlsx')
    print(f"Reorganized output: {output_path}")

    # Create reorganized file - ONE SHEET PER PARAMETER
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for col in Aggregated_Columns:
            print(f"\nProcessing parameter: {col}")

            # Create sheet columns: Tissue, then one column per concentration
            Sheet_columns = ['Tissue']
            for c in Conc:
                Sheet_columns.append(f'Conc_{c}')

            # Process each tissue
            rows_data = []
            for tissue in UniqueTissues:
                tissue_data = aggregated_data[aggregated_data['Tissue'] == tissue]
                
                if len(tissue_data) < 1:
                    continue

                # Create row data starting with tissue name
                row_data = {'Tissue': tissue}

                # Get parameter value for each concentration
                for c in Conc:
                    conc_data = tissue_data[tissue_data['Concentration'] == c]
                    if len(conc_data) > 0:
                        param_value = conc_data[col].iloc[0]
                    else:
                        param_value = np.nan
                    row_data[f'Conc_{c}'] = param_value

                # Filter criteria:
                # 1) Check if there's a DMSO value (Conc_-1)
                has_dmso = not pd.isna(row_data.get('Conc_-1', np.nan))
                
                # 2) Check if there's at least one non-DMSO concentration value
                has_other_conc = False
                for c in Conc:
                    if c != -1:  # Skip DMSO
                        if not pd.isna(row_data.get(f'Conc_{c}', np.nan)):
                            has_other_conc = True
                            break
                
                # Only include if both conditions are met
                if has_dmso and has_other_conc:
                    rows_data.append(row_data)

            # Create dataframe
            Parameter_sheet = pd.DataFrame(rows_data)

            # Clean sheet name for Excel (max 31 chars)
            sheet_name = col.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('*', '_')
            sheet_name = sheet_name.replace('[', '_').replace(']', '_').replace(':', '_')
            sheet_name = sheet_name.replace('?', '_').replace('|', '_').replace('(', '_').replace(')', '_')[:31]

            # Write to Excel
            Parameter_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Sheet '{sheet_name}' created with {len(Parameter_sheet)} rows (tissues)")

    print(f"\nReorganized data saved to: {output_path}")

    # Create statistics file (basic version)
    StatPath = parent_dir / (drug_name + '_Stats.xlsx')
    print(f"Stats output: {StatPath}")

    with pd.ExcelWriter(StatPath, engine='openpyxl') as writer:
        for col in Aggregated_Columns:

            # Clean sheet name
            sheet_name = col.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('*', '_')
            sheet_name = sheet_name.replace('[', '_').replace(']', '_').replace(':', '_')
            sheet_name = sheet_name.replace('?', '_').replace('|', '_').replace('(', '_').replace(')', '_')[:31]

            try:
                # Read the sheet we just created
                param_data = pd.read_excel(output_path, sheet_name=sheet_name)
                
                # Get concentration columns (excluding baseline -1)
                conc_columns = [c for c in param_data.columns if c.startswith('Conc_') and c != 'Conc_-1']

                if 'Conc_-1' not in param_data.columns or len(conc_columns) == 0:
                    print(f"Skipping {sheet_name}: no baseline or concentration data")
                    continue

                # Prepare results dataframe
                results = pd.DataFrame(columns=['Concentration', 'Test', 'Stat', 'p-value', 'Significance'])

                concentration = []
                tests = []
                statval = []
                pval = []
                significance = []

                # Perform Wilcoxon test for each concentration vs baseline
                for conc in conc_columns:
                    # Get paired data (same tissues in both columns)
                    aligned_data = param_data[[conc, 'Conc_-1']].dropna()

                    if len(aligned_data) >= 3:  # Need at least 3 pairs
                        stat, p_value = wilcoxon(aligned_data[conc], aligned_data['Conc_-1'])
                        concentration.append(conc)
                        tests.append('Wilcoxon')
                        statval.append(stat)
                        pval.append(p_value)
                        significance.append(p_value < 0.05)
                        print(f"{conc} Wilcoxon: stat={stat:.4f}, p={p_value:.4f}, Significant={p_value < 0.05}")

                results['Concentration'] = concentration
                results['Test'] = tests
                results['Stat'] = statval
                results['p-value'] = pval
                results['Significance'] = significance

                # Write to Excel
                results.to_excel(writer, sheet_name=sheet_name, index=False)

            except Exception as e:
                print(f"Error processing {sheet_name}: {e}")
                continue

    print(f"\nStatistics saved to: {StatPath}")
    
    # Now create plots and enhanced stats
    drug_folder = create_plots_and_stats(output_path, drug_name)
    
    print(f"{'='*80}\n")
    
    return output_path, drug_folder

def main():
    """Main function to process all XLSX files in selected folder"""
    print("Please select the folder containing drug XLSX files...")
    folder_path = select_input_folder()

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    # Find all XLSX files
    xlsx_files = list(Path(folder_path).glob("*.xlsx"))
    
    # Filter out already processed files
    xlsx_files = [f for f in xlsx_files if '_reorganized' not in f.name and '_Stats' not in f.name]

    if not xlsx_files:
        print(f"No XLSX files found in {folder_path}")
        return

    print(f"\nFound {len(xlsx_files)} XLSX files to process:")
    for f in xlsx_files:
        print(f"  - {f.name}")

    # Process each file
    processed_drugs = []
    for file_path in xlsx_files:
        try:
            output_path, drug_folder = process_single_file(str(file_path))
            processed_drugs.append((file_path.stem, drug_folder))
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("ALL FILES PROCESSED SUCCESSFULLY!")
    print("="*80)
    print(f"\nProcessed {len(processed_drugs)} drugs:")
    for drug_name, drug_folder in processed_drugs:
        print(f"  ✓ {drug_name} → {drug_folder}")
    print("="*80)

if __name__ == "__main__":
    main()