# Imports
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd 
from scipy.stats import shapiro, wilcoxon
import statsmodels.stats.multitest as multitest

# ------------------------------------------------------------
# Load data (CSV or Excel)
# ------------------------------------------------------------
file_path = filedialog.askopenfilename(
    title="Select aggregated CSV or Excel file",
    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
)

# Determine extension and load
ext = os.path.splitext(file_path)[1].lower()

if ext == ".csv":
    aggregated_data = pd.read_csv(file_path)
else:
    aggregated_data = pd.read_excel(file_path)

# Output paths
folder = os.path.dirname(file_path)
base = os.path.splitext(os.path.basename(file_path))[0]

output_path = os.path.join(folder, base + "_reorganized.xlsx")
StatPath = os.path.join(folder, base + "_Stats.xlsx")

# ------------------------------------------------------------
# Extract Exp and ExpWell from Video name
# Video name example: AK16_120625_D23_5x obj_Iso2_D5.mp4
# Exp = first token before "_", e.g., "AK16"
# Well = from column "Well"
# ------------------------------------------------------------

video_names = aggregated_data["Video name"].astype(str)

ExpID = [v.split("_")[0] for v in video_names]       # e.g., AK16
ExpWellID = [f"{exp}_{well}" for exp, well in zip(ExpID, aggregated_data["Well"])]

aggregated_data["Exp"] = ExpID
aggregated_data["ExpWell"] = ExpWellID

UniqueExpWell = np.unique(ExpWellID)
Aggregated_Columns = aggregated_data.columns

# ------------------------------------------------------------
# Find sorted concentrations
# ------------------------------------------------------------
Conc = sorted(aggregated_data['Concentration'].unique())
print(f"Found concentrations: {Conc}")

# Determine baseline concentration (first/lowest concentration)
baseline_conc = Conc[0] if len(Conc) > 0 else None
baseline_col_name = f"Conc_{baseline_conc}"
print(f"Using baseline concentration: {baseline_conc} (column: {baseline_col_name})")

# ------------------------------------------------------------
# Create reorganized Excel file
# ------------------------------------------------------------
print("\n=== Creating reorganized file ===")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

    for col in Aggregated_Columns:

        if col in ['Video name', 'Exp', 'Well', 'ExpWell', 'Concentration', 'Tissue', 'Ring ID']:
            continue

        print(f"\nProcessing {col}")

        # Sheet columns
        Sheet_columns = ['Video name', 'Exp', 'Well', 'Tissue']
        col_name = col.replace(' ', '_')

        for c in Conc:
            Sheet_columns.append(f'Conc_{c}')

        rows_data = []

        # For each Exp-Well
        for exp_well in UniqueExpWell:

            subset = aggregated_data[aggregated_data["ExpWell"] == exp_well]
            if len(subset) < 1:
                continue

            # Get unique tissues for this exp_well
            unique_tissues = subset["Tissue"].unique()
            
            # For each tissue
            for tissue in unique_tissues:
                tissue_subset = subset[subset["Tissue"] == tissue]
                
                # Metadata - use video name from baseline concentration if available
                baseline_video = tissue_subset[tissue_subset["Concentration"] == baseline_conc]
                if len(baseline_video) > 0:
                    video_name = baseline_video["Video name"].iloc[0]
                else:
                    # If no baseline, use first available video name
                    video_name = tissue_subset["Video name"].iloc[0]
                
                exp = tissue_subset["Exp"].iloc[0]
                well = tissue_subset["Well"].iloc[0]

                row = {
                    "Video name": video_name,
                    "Exp": exp,
                    "Well": well,
                    "Tissue": tissue
                }

                # Parameter values for each concentration
                for c in Conc:
                    sub2 = tissue_subset[tissue_subset["Concentration"] == c]
                    row[f"Conc_{c}"] = sub2[col].iloc[0] if len(sub2) > 0 else np.nan

                rows_data.append(row)

        # Make dataframe
        Parameter_sheet = pd.DataFrame(rows_data)

        # Fix sheet name for Excel
        sheet_name = (
            col_name.replace('/', '_')
                    .replace('\\', '_')
                    .replace('*', '_')
                    .replace('[', '_')
                    .replace(']', '_')
                    .replace(':', '_')
                    .replace('?', '_')
                    .replace('|', '_')
        )[:31]

        Parameter_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  Created sheet '{sheet_name}' with {len(Parameter_sheet)} rows")

print(f"\nReorganized file written to:\n{output_path}")


# ------------------------------------------------------------
# Create statistics Excel file
# ------------------------------------------------------------
print("\n=== Creating statistics file ===")

# Check if we have enough concentrations for statistics
if len(Conc) < 2:
    print(f"WARNING: Only {len(Conc)} concentration(s) found. Need at least 2 for statistics.")
    print("Skipping statistics file creation.")
else:
    stats_sheets_created = 0

    with pd.ExcelWriter(StatPath, engine='openpyxl') as writer:

        for col in Aggregated_Columns:

            if col in ['Video name', 'Exp', 'Well', 'ExpWell', 'Concentration', 'Tissue', 'Ring ID']:
                continue

            # Sheet name used before
            col_name = col.replace(' ', '_')
            sheet_name = (
                col_name.replace('/', '_')
                        .replace('\\', '_')
                        .replace('*', '_')
                        .replace('[', '_')
                        .replace(']', '_')
                        .replace(':', '_')
                        .replace('?', '_')
                        .replace('|', '_')
            )[:31]

            print(f"\nProcessing stats for {sheet_name}")

            # Load reorganized sheet
            try:
                param_data = pd.read_excel(output_path, sheet_name=sheet_name)
            except Exception as e:
                print(f"  ERROR: Could not read sheet '{sheet_name}': {e}")
                continue

            # Columns like Conc_1.0, Conc_2.0, ...
            stats_columns = [c for c in param_data.columns if c.startswith("Conc_")]
            
            if len(stats_columns) == 0:
                print(f"  WARNING: No concentration columns found for {sheet_name}")
                continue

            stats_data = param_data[stats_columns]

            # Check for baseline
            if baseline_col_name not in stats_data.columns:
                print(f"  WARNING: No baseline {baseline_col_name} for {col}, skipping stats")
                continue

            print(f"  Shapiro tests:")
            NormalGlobal = True

            # Normality per concentration
            for conc, values in stats_data.items():
                clean_values = values.dropna()
                if len(clean_values) < 3:
                    print(f"    {conc}: Not enough data points (n={len(clean_values)})")
                    continue
                stat, p_value = shapiro(clean_values)
                if p_value < 0.05:
                    NormalGlobal = False
                print(f"    {conc}: W={stat:.4f}, p={p_value:.4f}")

            print(f"  NormalGlobal = {NormalGlobal}")

            # Statistical results
            concentrations = []
            tests = []
            statval = []
            pval = []
            signif = []

            # Wilcoxon vs baseline
            baseline = param_data[baseline_col_name]

            for conc in stats_columns:
                if conc == baseline_col_name:
                    continue

                try:
                    stat, p = wilcoxon(param_data[conc], baseline, nan_policy='omit')
                    
                    concentrations.append(conc)
                    tests.append("Wilcoxon")
                    statval.append(stat)
                    pval.append(p)
                    signif.append(p < 0.05)

                    print(f"    {conc} Wilcoxon: stat={stat}, p={p:.4f}")
                except Exception as e:
                    print(f"    ERROR with {conc}: {e}")
                    continue

            if len(concentrations) == 0:
                print(f"  WARNING: No statistics computed for {sheet_name}")
                continue

            results = pd.DataFrame({
                'Concentration': concentrations,
                'Test': tests,
                'Stat': statval,
                'p-value': pval,
                'Significance': signif
            })

            results.to_excel(writer, sheet_name=sheet_name, index=False)
            stats_sheets_created += 1
            print(f"  Created stats sheet '{sheet_name}'")

    if stats_sheets_created == 0:
        print("\nWARNING: No statistics sheets were created.")
        print("The statistics file may be empty or invalid.")
        print("Common causes:")
        print(f"  - Missing baseline concentration ({baseline_col_name})")
        print("  - Insufficient data points")
        print("  - All columns were skipped")
        # Delete the empty stats file
        if os.path.exists(StatPath):
            os.remove(StatPath)
            print(f"\nDeleted empty statistics file: {StatPath}")
    else:
        print(f"\nStatistics file written to:\n{StatPath}")
        print(f"Total sheets created: {stats_sheets_created}")

print("\n=== Processing complete ===")