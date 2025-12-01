import pandas as pd
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

##### NEW SYNTAX #####
# experiment_date_dayofculture_zoom_type_pacing_drugdose_well_comment.mp4

def extract_drug_name(video_name):
    """Extract drug name from video name (field at position -3 when split by '_')"""
    parts = video_name.split('_')
    if len(parts) >= 4:
        drug_field = parts[-3]  # Get the field at position -3
        # Extract the drug name without the number (e.g., "Basal2" -> "Basal")
        match = re.match(r'^([A-Za-z]+)', drug_field)
        if match:
            return match.group(1)
    return None

def select_input_file():
    """Open file dialog to select CSV file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def process_csv_to_xlsx(csv_path):
    """Process CSV and create XLSX files for each drug"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract drug names for all rows
    df['Drug'] = df['Video name'].apply(extract_drug_name)
    
    # Get unique drugs (excluding None and DMSO)
    drugs = [d for d in df['Drug'].dropna().unique() if d.upper() != 'DMSO']
    
    # Base input folder
    input_dir = Path(csv_path).parent
    
    # Create "by_drug" subfolder
    output_dir = input_dir / "by_drug"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output folder: {output_dir}")
    print(f"Found {len(drugs)} unique drugs: {', '.join(drugs)}")
    
    # Get all DMSO rows (to be included in every drug file)
    dmso_mask = df['Video name'].str.contains(r'DMSO\d*', case=False, na=False, regex=True)
    dmso_rows = df[dmso_mask].copy()
    print(f"Found {len(dmso_rows)} DMSO control rows (will be included in each drug file)")
    
    # Process each drug
    for drug in drugs:
        drug_pattern = rf'{drug}\d*'
        drug_mask = df['Video name'].str.contains(drug_pattern, case=False, na=False, regex=True)
        
        # Get drug-specific rows
        drug_rows = df[drug_mask].copy()
        
        # Combine DMSO rows with drug rows
        filtered_df = pd.concat([dmso_rows, drug_rows], ignore_index=True)
        
        # Remove temp column
        filtered_df = filtered_df.drop('Drug', axis=1)
        
        if not filtered_df.empty and len(drug_rows) > 0:
            output_file = output_dir / f"{drug}.xlsx"
            filtered_df.to_excel(output_file, index=False)
            
            print(f"Created {output_file.name} → {len(filtered_df)} rows "
                  f"({len(drug_rows)} drug + {len(dmso_rows)} DMSO)")
        else:
            print(f"No drug-specific data found for {drug}, skipping")
    
    print(f"\nAll files saved to: {output_dir}")

# Main execution
if __name__ == "__main__":
    print("Please select your CSV file...")
    csv_file = select_input_file()
    
    if csv_file:
        print(f"Processing: {csv_file}")
        process_csv_to_xlsx(csv_file)
        print("\nProcessing complete!")
    else:
        print("No file selected. Exiting.")
