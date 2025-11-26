import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
import os
from tkinter import Tk, filedialog

# ------------------ INPUT FILES ------------------
stats_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Input\Dis_AK06-AK07_Pooled_Trié_Stats.xlsx"
data_file  = r"Z:\RnD\Baptiste\Example analyses\Contractility\Input\Dis_AK06-AK07_Pooled_Trié.xlsx"

# ------------------ LOAD DATA ------------------
df_data = pd.read_excel(data_file)

# ------------------ EXTRACT Dis GROUP ------------------
def extract_dis_number(name):
    if pd.isna(name):
        return None
    match = re.search(r"dis\s*([1-4])", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

df_data["Dis_group"] = df_data["Name"].apply(extract_dis_number)
df_data = df_data[df_data["Dis_group"].notna()]
df_data["Dis_group"] = df_data["Dis_group"].astype(int)

# ------------------ LOAD STATS ------------------
stats_sheets = pd.read_excel(stats_file, sheet_name=None)

ignore_cols = ["Total video time", "Initial Youngs Modulus", "Days in culture"]

def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""  # plus de "ns"

# ------------------ ASK FOR SAVE FOLDER ------------------
root = Tk()
root.withdraw()
save_folder = filedialog.askdirectory(title="Select folder to save plots")
root.destroy()

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ------------------ LOOP OVER METRICS ------------------
for metric_raw, df_stats in stats_sheets.items():

    metric = metric_raw.replace("_", " ")

    if metric in ignore_cols:
        continue
    if metric not in df_data.columns:
        continue

    print(f"Processing metric: {metric_raw} → '{metric}'")

    concentrations = [1,2,3,4]
    means, low_ci, high_ci, pvals = [], [], [], []

    for c in concentrations:
        vals = df_data[df_data["Dis_group"] == c][metric].dropna()
        if len(vals) == 0:
            means.append(np.nan)
            low_ci.append(np.nan)
            high_ci.append(np.nan)
            pvals.append(None)
            continue

        m = vals.mean()
        means.append(m)
        if len(vals) > 1:
            sem = stats.sem(vals)
            ci = stats.t.interval(0.95, len(vals)-1, loc=m, scale=sem)
            low_ci.append(ci[0])
            high_ci.append(ci[1])
        else:
            low_ci.append(m)
            high_ci.append(m)

        # --- Récupérer la p-value correctement ---
        conc_str = f"Conc_{c}"  # "Conc_1", "Conc_2", ...
        row = df_stats[(df_stats["Concentration"] == conc_str) & (df_stats["Test"] == "Wilcoxon")]
        if len(row) > 0:
            pvals.append(float(row["p-value"].values[0]))
        else:
            pvals.append(None)

    # ------------------ ERROR BARS ------------------
    means = np.array(means, dtype=float)
    low_ci = np.array(low_ci, dtype=float)
    high_ci = np.array(high_ci, dtype=float)
    yerr_lower = np.nan_to_num(means - low_ci, nan=0.0)
    yerr_upper = np.nan_to_num(high_ci - means, nan=0.0)
    yerr = [yerr_lower, yerr_upper]

    # ------------------ PLOT ------------------
    plt.figure(figsize=(6,5))
    x = np.arange(len(concentrations))
    plt.errorbar(x, means, yerr=yerr, fmt='o-', capsize=5, markersize=7, linewidth=2)

    ymax = np.nanmax(means)
    for i, p in enumerate(pvals):
        if p is not None and p < 0.05:
            plt.text(x[i], means[i] + 0.05*ymax, stars(p), ha='center', fontsize=14)

    plt.xticks(x, concentrations)
    plt.xlabel("Dis group (1–4)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Dis group (n={len(df_data)})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ------------------ SAVE FIGURE ------------------
    save_path = os.path.join(save_folder, f"{metric.replace(' ','_')}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()  # ferme la figure pour ne pas afficher

print(f"\nAll plots saved in: {save_folder}")
