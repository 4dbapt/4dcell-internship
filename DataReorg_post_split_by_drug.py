import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon
import statsmodels.stats.multitest as multitest
from pathlib import Path

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
    output_path = parent_dir / (Path(file_path).stem + '_reorganized.xlsx')
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
                        # Take the mean if multiple values (shouldn't happen, but safety)
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

    # Create statistics file
    StatPath = parent_dir / (Path(file_path).stem + '_Stats.xlsx')
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
    print(f"{'='*80}\n")

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
    for file_path in xlsx_files:
        try:
            process_single_file(str(file_path))
        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("ALL FILES PROCESSED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()