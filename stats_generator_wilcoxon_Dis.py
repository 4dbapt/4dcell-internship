import pandas as pd
from scipy.stats import wilcoxon
import re

# ⚙️ Paramètres
input_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Input\Dis_AK06-AK07_Pooled_Trié.xlsx"
output_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Output\Wilcoxon_results.xlsx"

# Colonnes à analyser
columns_to_test = [
    'Total video time', 'SNR', 'Initial Youngs Modulus', 'Days in culture',
    'Period (s)', 'Beating Frequency (Hz)', 'Interpeak irregularity (s)',
    'N_twitch', 'Contraction strain (A.U.)', 'Maximum contraction speed (A.U./s)',
    'Maximum relaxation speed (A.U./s)', 'Contraction stress (mN/mm2)',
    'Contraction time (s)', 'Relaxation time (s)', 'Contraction-relaxation time (s)',
    'Contraction time 10 (s)', 'Relaxation time 10 (s)',
    'Contraction time 50 (s)', 'Relaxation time 50 (s)',
    'Contraction time 90 (s)', 'Relaxation time 90 (s)'
]

# Lire le fichier et nettoyer les colonnes
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

# Nettoyage des noms de puits pour merge fiable
df['Well'] = df['Well'].astype(str).str.strip().str.upper()

# Fonction pour nettoyer noms de sheets Excel
def clean_sheet_name(name):
    name = re.sub(r'[\\/*?:[\]]', '_', name)
    return name[:31]

results_dict = {}

for col in columns_to_test:
    result_rows = []
    df[col] = pd.to_numeric(df[col], errors='coerce')

    for i in range(1, 5):
        conc_name = f"Conc_{i}"

        # Sélection DMSO et DisX
        df_control = df[df['Concentration']=='DMSO'][['Well', col]].dropna()
        df_dis = df[df['Concentration']==f'Dis{i}'][['Well', col]].dropna()

        # Merge sur Well pour apparier correctement
        paired_df = pd.merge(df_control, df_dis, on='Well', suffixes=('_control','_dis'))

        print(f"{col} - {conc_name}: {len(paired_df)} paires trouvées")  # DEBUG

        if len(paired_df) > 0:
            try:
                stat, p_value = wilcoxon(paired_df[f'{col}_dis'], paired_df[f'{col}_control'])
                significance = p_value < 0.05
            except ValueError:
                stat, p_value, significance = None, None, None
        else:
            stat, p_value, significance = None, None, None

        result_rows.append({
            'Concentration': conc_name,
            'Test': 'Wilcoxon',
            'Stat': stat,
            'p-value': p_value,
            'Significance': significance
        })

    results_dict[col] = pd.DataFrame(result_rows)

# Écriture Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for col_name, df_result in results_dict.items():
        df_result.to_excel(writer, sheet_name=clean_sheet_name(col_name), index=False)

print(f"✅ Résultats Wilcoxon appariés sur Well enregistrés dans {output_file}")
