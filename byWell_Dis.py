import pandas as pd
import re

# ⚙️ Paramètres
input_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Input\Dis_AK06-AK07_Pooled_Trié.xlsx"
output_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Output\By_Well.xlsx"

# Lire le fichier
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

# Nettoyage des noms de sheets Excel
def clean_sheet_name(name):
    # Remplace les caractères invalides pour Excel et tronque à 31 caractères
    name = re.sub(r'[\\/*?:[\]]', '_', str(name))
    return name[:31]

# Identifier tous les puits uniques
wells = df['Well'].astype(str).str.strip().unique()

# Écriture Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for well in wells:
        well_clean = clean_sheet_name(well)
        df_well = df[df['Well'].astype(str).str.strip() == well]
        df_well.to_excel(writer, sheet_name=well_clean, index=False)

print(f"✅ Fichier par puits créé : {output_file}")
