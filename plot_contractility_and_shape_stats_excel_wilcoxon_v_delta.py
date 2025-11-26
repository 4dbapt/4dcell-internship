import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm
import os
from tkinter import Tk, filedialog

# ------------------ LOAD DATA ------------------
root = Tk()
root.withdraw()
data_file = filedialog.askopenfilename(title="Select your data Excel file with multiple sheets")
root.destroy()

# Lire TOUTES les sheets du fichier Excel
all_sheets = pd.read_excel(data_file, sheet_name=None)

print(f"\n{'='*60}")
print(f"Fichier chargé: {os.path.basename(data_file)}")
print(f"Nombre de sheets (mesures physiques): {len(all_sheets)}")
print(f"Sheets trouvées: {list(all_sheets.keys())}")
print(f"{'='*60}")

# ------------------ ASK FOR SAVE FOLDER ------------------
root = Tk()
root.withdraw()
save_folder = filedialog.askdirectory(title="Select folder to save plots and stats")
root.destroy()

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ------------------ FUNCTION: NON-PARAMETRIC CI FOR WILCOXON ------------------
def wilcoxon_ci_improved(x, y, alpha=0.05, method='hodges-lehmann'):
    """
    Calculate non-parametric confidence interval for paired differences
    
    Parameters:
    -----------
    x, y : array-like
        Paired samples
    alpha : float
        Significance level (default 0.05 for 95% CI)
    method : str
        'hodges-lehmann' : Uses all pairwise averages (more accurate)
        'simple' : Uses sorted differences (faster but less accurate)
    
    Returns:
    --------
    ci_low, ci_high : float
        Lower and upper bounds of confidence interval
    """
    d = np.array(y) - np.array(x)
    n = len(d)
    
    if n == 0:
        return np.nan, np.nan
    
    if method == 'hodges-lehmann':
        # Hodges-Lehmann estimator: median of all pairwise averages
        # This is the TRUE non-parametric CI for Wilcoxon signed-rank test
        walsh_averages = []
        for i in range(n):
            for j in range(i, n):
                walsh_averages.append((d[i] + d[j]) / 2)
        
        walsh_sorted = np.sort(walsh_averages)
        m = len(walsh_sorted)
        
        # Calculate confidence interval indices
        z = norm.ppf(1 - alpha/2)
        # More accurate formula for Wilcoxon CI
        k = int(np.round(m/2 - z * np.sqrt(n * (n + 1) * (2*n + 1) / 24)))
        k = max(0, min(k, m-1))
        
        return walsh_sorted[k], walsh_sorted[m - k - 1]
    
    else:  # method == 'simple'
        # Simplified version (faster but less accurate)
        d_sorted = np.sort(d)
        z = norm.ppf(1 - alpha/2)
        k = int(np.floor((n - z * np.sqrt(n)) / 2))
        k = max(0, min(k, n-1))
        
        return d_sorted[k], d_sorted[n - k - 1]


# ------------------ EXCEL WRITER ------------------
output_file = os.path.join(save_folder, "Wilcoxon_Stats.xlsx")
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

# ------------------ MAIN LOOP: ITERATE OVER SHEETS ------------------
for sheet_name, df in all_sheets.items():
    
    print(f"\n{'='*60}")
    print(f"Processing: {sheet_name}")
    print(f"{'='*60}")
    
    # Vérifier les colonnes nécessaires
    required_cols = ["Conc_-1", "Conc_1", "Conc_2", "Conc_3", "Conc_4"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"  ⚠ SKIPPED: Missing columns {missing_cols}")
        continue
    
    print(f"  Nombre de lignes: {len(df)}")
    
    # Afficher le nombre de valeurs manquantes par colonne
    for col in required_cols:
        n_missing = df[col].isna().sum()
        n_valid = df[col].notna().sum()
        print(f"    {col}: {n_valid} valeurs valides, {n_missing} manquantes")
    
    # ------------------ STATISTICS CALCULATION ------------------
    baseline = df["Conc_-1"]
    
    stats_rows = []
    means = []
    medians = []  # Ajouter médiane (plus robuste que moyenne)
    ci_low_plot = []
    ci_high_plot = []
    pvals_plot = []
    n_samples = []
    
    for c in [1, 2, 3, 4]:
        col = f"Conc_{c}"
        
        # Créer un DataFrame avec les paires baseline-test
        # et supprimer les lignes où AU MOINS UNE des deux valeurs est manquante
        paired_data = pd.DataFrame({
            "baseline": df["Conc_-1"],
            "test": df[col]
        }).dropna()
        
        x = paired_data["baseline"].values
        y = paired_data["test"].values
        n = len(paired_data)
        
        print(f"\n  {col}:")
        print(f"    - N paires valides: {n}")
        
        if n < 2:
            print(f"    - ⚠ SKIPPED (données insuffisantes)")
            stats_rows.append([col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, np.nan, n, ""])
            means.append(np.nan)
            medians.append(np.nan)
            ci_low_plot.append(np.nan)
            ci_high_plot.append(np.nan)
            pvals_plot.append(None)
            n_samples.append(n)
            continue
        
        # ---- WILCOXON PAIRED TEST ----
        try:
            # Vérifier s'il y a des différences non-nulles
            diff = y - x
            if np.all(diff == 0):
                print(f"    - ⚠ WARNING: Toutes les différences sont nulles")
                stat, pval = np.nan, 1.0
            else:
                stat, pval = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
            
            print(f"    - Statistique: {stat:.4f}")
            print(f"    - p-value: {pval:.6f}")
            
        except Exception as e:
            print(f"    - ⚠ ERROR: {e}")
            stats_rows.append([col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, np.nan, n, "ERROR"])
            means.append(np.nan)
            medians.append(np.nan)
            ci_low_plot.append(np.nan)
            ci_high_plot.append(np.nan)
            pvals_plot.append(None)
            n_samples.append(n)
            continue
        
        # ---- NON-PARAMETRIC CI (improved) ----
        # Choisir la méthode: 'hodges-lehmann' (précis) ou 'simple' (rapide)
        # Pour n > 100, utiliser 'simple' pour éviter trop de calculs
        ci_method = 'simple' if n > 100 else 'hodges-lehmann'
        ci_low, ci_high = wilcoxon_ci_improved(x, y, method=ci_method)
        
        print(f"    - IC 95% (méthode: {ci_method}): [{ci_low:.4f}, {ci_high:.4f}]")
        
        # Statistiques descriptives
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)  # Plus robuste aux outliers
        
        print(f"    - Différence moyenne: {mean_diff:.4f}")
        print(f"    - Différence médiane: {median_diff:.4f}")
        
        # Sauvegarder les statistiques
        significance = "****" if pval < 0.0001 else "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        stats_rows.append([col, "Wilcoxon", stat, pval, mean_diff, median_diff, ci_low, ci_high, n, significance])
        
        # Pour le graphique (utiliser médiane car plus adaptée au test non-paramétrique)
        means.append(mean_diff)
        medians.append(median_diff)
        ci_low_plot.append(ci_low)
        ci_high_plot.append(ci_high)
        pvals_plot.append(pval)
        n_samples.append(n)
    
    # ------------------ SAVE STATS TO EXCEL ------------------
    if stats_rows:
        df_stats = pd.DataFrame(stats_rows, 
                               columns=["Concentration", "Test", "Statistic", "p-value", 
                                       "Mean_Diff", "Median_Diff", "CI_low", "CI_high", "N", "Significance"])
        
        # Tronquer le nom de la sheet à 31 caractères (limite Excel)
        excel_sheet_name = str(sheet_name)[:31]
        df_stats.to_excel(writer, sheet_name=excel_sheet_name, index=False)
        print(f"\n  ✓ Stats sauvegardées dans la sheet: {excel_sheet_name}")
    
    # ------------------ PLOT ------------------
    # Vérifier qu'il y a au moins une valeur valide à plotter
    if not all(np.isnan(medians)):
        # Utiliser médiane plutôt que moyenne (plus cohérent avec test non-paramétrique)
        medians_arr = np.array(medians)
        ci_low_arr = np.array(ci_low_plot)
        ci_high_arr = np.array(ci_high_plot)
        
        # Calculer les barres d'erreur
        yerr_lower = np.abs(medians_arr - ci_low_arr)
        yerr_upper = np.abs(ci_high_arr - medians_arr)
        yerr = [yerr_lower, yerr_upper]
        
        x_pos = np.arange(4)
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(x_pos, medians_arr, yerr=yerr, fmt='o-', capsize=6, linewidth=2.5, 
                    markersize=9, color='steelblue', ecolor='gray', alpha=0.9)
        
        # Ajouter les étoiles de significativité
        valid_medians = medians_arr[~np.isnan(medians_arr)]
        if len(valid_medians) > 0:
            ymax = np.max(valid_medians)
            ymin = np.min(valid_medians)
            y_range = ymax - ymin if ymax != ymin else abs(ymax) if ymax != 0 else 1
        else:
            ymax, ymin, y_range = 1, 0, 1
        
        for i, p in enumerate(pvals_plot):
            if p is not None and p < 0.05 and not np.isnan(medians_arr[i]):
                stars = "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*"
                y_offset = 0.08 * y_range
                plt.text(x_pos[i], medians_arr[i] + y_offset, stars, 
                        ha="center", fontsize=16, fontweight='bold', color='red')
        
        # Ajouter les tailles d'échantillon
        for i, n in enumerate(n_samples):
            if n > 0:
                plt.text(x_pos[i], ymin - 0.12 * y_range, f"n={n}", 
                        ha="center", fontsize=9, color='dimgray')
        
        # Ligne de référence à zéro
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.4)
        
        plt.xticks(x_pos, ["Conc_1", "Conc_2", "Conc_3", "Conc_4"], fontsize=11)
        plt.xlabel("Concentration", fontsize=12, fontweight='bold')
        plt.ylabel(f"Δ {sheet_name} (médiane, vs Conc_-1)", fontsize=12, fontweight='bold')
        plt.title(f"{sheet_name}\nTest de Wilcoxon apparié: Conc_-1 vs Conc_X", 
                 fontsize=13, fontweight='bold', pad=15)
        plt.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plot_filename = f"{str(sheet_name).replace(' ', '_').replace('/', '_')}.png"
        plot_path = os.path.join(save_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Graphique sauvegardé: {plot_filename}")
    else:
        print(f"  ⚠ Pas de graphique généré (aucune donnée valide)")

writer.close()

print("\n" + "="*60)
print("✔ ANALYSE TERMINÉE!")
print("="*60)
print(f"Dossier de sortie: {save_folder}")
print(f"  - Wilcoxon_Stats.xlsx (tableaux statistiques)")
print(f"  - {len([s for s in all_sheets.keys()])} graphique(s) (.png)")
print("="*60)