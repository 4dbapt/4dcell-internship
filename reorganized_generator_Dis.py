import pandas as pd
import re

# === INPUT / OUTPUT ===
input_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Input\Dis_AK06-AK07_Pooled_Trié.xlsx"
output_file = r"Z:\RnD\Baptiste\Example analyses\Contractility\Output\Dis_reorganized_vBapt.xlsx"

# === Charger le fichier ===
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

# Colonnes à ignorer (meta)
meta_cols = ["Name", "Concentration", "Well"]

# Colonnes de mesures physiques
measure_cols = [c for c in df.columns if c not in meta_cols]

# Nettoyage noms de sheets Excel
def clean_sheet_name(name):
    name = re.sub(r'[\\/*?:[\]]', '_', name)
    return name[:31]

# === Construction fichier de sortie ===
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

    for measure in measure_cols:
        # Nouveau tableau
        wells = sorted(df["Well"].unique())
        out_df = pd.DataFrame({"Well": wells})

        # Conc_-1 = 1
        out_df["Conc_-1"] = 1

        # Pour Conc_1 → Conc_4
        for c in range(1, 5):
            sub = df[df["Concentration"] == f"Dis{c}"][["Well", measure]]

            # conversion numérique + NA si impossible
            sub[measure] = pd.to_numeric(sub[measure], errors="coerce")

            # merge sur Well
            out_df = out_df.merge(
                sub.rename(columns={measure: f"Conc_{c}"}),
                on="Well",
                how="left"
            )

        # Écrire sheet
        out_df.to_excel(writer, sheet_name=clean_sheet_name(measure), index=False)

print(f"✅ Fichier généré : {output_file}")
