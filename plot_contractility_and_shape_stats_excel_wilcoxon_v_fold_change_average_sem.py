import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, sem
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
    sems_plot = []
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
            stats_rows.append([col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, n, ""])
            means.append(np.nan)
            sems_plot.append(np.nan)
            pvals_plot.append(None)
            n_samples.append(n)
            continue
        
        # ---- WILCOXON PAIRED TEST (sur les différences) ----
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
            stats_rows.append([col, "Wilcoxon", np.nan, np.nan, np.nan, np.nan, n, "ERROR"])
            means.append(np.nan)
            sems_plot.append(np.nan)
            pvals_plot.append(None)
            n_samples.append(n)
            continue
        
        # ---- FOLD CHANGE MOYEN ET SEM ----
        # Éviter les divisions par zéro
        valid_idx = x > 0
        if np.sum(valid_idx) == 0:
            print(f"    - ⚠ WARNING: Aucune valeur baseline valide pour fold change")
            mean_fold_change = np.nan
            sem_fc = np.nan
        else:
            fold_changes = y[valid_idx] / x[valid_idx]
            mean_fold_change = np.mean(fold_changes)
            sem_fc = sem(fold_changes)
        
        print(f"    - Fold Change moyen: {mean_fold_change:.4f}")
        print(f"    - SEM FC: {sem_fc:.4f}")
        
        # Sauvegarder les statistiques
        significance = "****" if pval < 0.0001 else "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        stats_rows.append([col, "Wilcoxon", stat, pval, mean_fold_change, sem_fc, n, significance])
        
        # Pour le graphique
        means.append(mean_fold_change)
        sems_plot.append(sem_fc)
        pvals_plot.append(pval)
        n_samples.append(n)
    
    # ------------------ SAVE STATS TO EXCEL ------------------
    if stats_rows:
        df_stats = pd.DataFrame(stats_rows, 
                               columns=["Concentration", "Test", "Statistic", "p-value", 
                                       "Mean_FC", "SEM_FC", "N", "Significance"])
        
        # Tronquer le nom de la sheet à 31 caractères (limite Excel)
        excel_sheet_name = str(sheet_name)[:31]
        df_stats.to_excel(writer, sheet_name=excel_sheet_name, index=False)
        print(f"\n  ✓ Stats sauvegardées dans la sheet: {excel_sheet_name}")
    
    # ------------------ PLOT ------------------
    # Vérifier qu'il y a au moins une valeur valide à plotter
    if not all(np.isnan(means)):
        means_arr = np.array(means)
        sems_arr = np.array(sems_plot)
        
        # Barres d'erreur symétriques (SEM)
        yerr = sems_arr
        
        x_pos = np.arange(4)
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(x_pos, means_arr, yerr=yerr, fmt='o-', capsize=6, linewidth=2.5, 
                    markersize=9, color='steelblue', ecolor='gray', alpha=0.9)
        
        # Ajouter les étoiles de significativité
        valid_means = means_arr[~np.isnan(means_arr)]
        if len(valid_means) > 0:
            ymax = np.max(valid_means + sems_arr[~np.isnan(means_arr)])
            ymin = np.min(valid_means - sems_arr[~np.isnan(means_arr)])
            y_range = ymax - ymin if ymax != ymin else abs(ymax) if ymax != 0 else 1
        else:
            ymax, ymin, y_range = 1, 0, 1
        
        for i, p in enumerate(pvals_plot):
            if p is not None and p < 0.05 and not np.isnan(means_arr[i]):
                stars = "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*"
                y_offset = 0.08 * y_range
                plt.text(x_pos[i], means_arr[i] + sems_arr[i] + y_offset, stars, 
                        ha="center", fontsize=16, fontweight='bold', color='red')
        
        # Ajouter les tailles d'échantillon
        for i, n in enumerate(n_samples):
            if n > 0:
                plt.text(x_pos[i], ymin - 0.12 * y_range, f"n={n}", 
                        ha="center", fontsize=9, color='dimgray')
        
        # Ligne de référence à 1 (pas de changement)
        plt.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.4)
        
        plt.xticks(x_pos, ["Conc_1", "Conc_2", "Conc_3", "Conc_4"], fontsize=11)
        plt.xlabel("Concentration", fontsize=12, fontweight='bold')
        plt.ylabel(f"Fold Change moyen {sheet_name}\n(vs Conc_-1)", fontsize=12, fontweight='bold')
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