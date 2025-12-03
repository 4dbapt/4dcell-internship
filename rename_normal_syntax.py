import os
import shutil
import tkinter as tk
from tkinter import filedialog

def select_folder(title):
    """Open a dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def rename_video(old_name):
    """
    Expected input format:
    date_nomexp_jour_zoom_type_pacing_drugdose_comment_well.mp4
    
    But handles cases with missing fields or underscores in comments.
    
    Output format:
    nomexp_date_jour_zoom_type_pacing_drugdose_well_comment.mp4
    """
    
    if not old_name.endswith(".mp4"):
        return None, None
    
    core = old_name[:-4]
    parts = core.split("_")
    
    # Minimum required parts: at least date, nomexp, and well
    if len(parts) < 3:
        print(f"Skipping {old_name} (too few parts: {len(parts)})")
        return None, None
    
    # Extract well (last part before .mp4, but after last underscore)
    well = parts[-1]
    
    # Extract date (first part)
    date = parts[0]
    
    # Extract nomexp (second part)
    if len(parts) >= 2:
        nomexp = parts[1]
    else:
        print(f"Skipping {old_name} (missing nomexp)")
        return None, None
    
    # Extract jour (third part, if exists)
    jour = parts[2] if len(parts) >= 3 else ""
    
    # Extract zoom, type_, pacing, drugdose (parts 3-6 if they exist)
    zoom = parts[3] if len(parts) >= 4 else ""
    type_ = parts[4] if len(parts) >= 5 else ""
    pacing = parts[5] if len(parts) >= 6 else ""
    drugdose = parts[6] if len(parts) >= 7 else ""
    
    # Comment is everything between drugdose and well (could be multiple parts)
    # If we have at least 8 parts, parts[7:-1] is the comment
    # If we have exactly 8 parts, parts[7] is the comment (and parts[-1] is well)
    # If we have more than 8 parts, join the middle parts as comment
    if len(parts) >= 8:
        comment_parts = parts[7:-1]  # Everything between drugdose index and well
        comment = "_".join(comment_parts) if comment_parts else ""
    else:
        comment = ""
    
    # Build new name, only including non-empty fields (except well which should always be included)
    new_parts = [
        nomexp,
        date,
        jour if jour else "NA",
        zoom if zoom else "NA",
        type_ if type_ else "NA",
        pacing if pacing else "NA",
        drugdose if drugdose else "NA",
        well,
        comment if comment else "NA"
    ]
    
    new_name = "_".join(new_parts) + ".mp4"
    
    return old_name, new_name

def main():
    """Main function to rename video files."""
    print("Select the input folder containing the wrongly named videos...")
    input_folder = select_folder("Select input folder with .mp4 videos")
    if not input_folder:
        print("No input folder selected. Exiting.")
        return
    
    print("Select the output folder where renamed videos will be stored...")
    output_folder = select_folder("Select output folder for renamed videos")
    if not output_folder:
        print("No output folder selected. Exiting.")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".mp4"):
            continue
        
        old_name, new_name = rename_video(filename)
        
        if new_name is None:
            skipped += 1
            continue
        
        old_path = os.path.join(input_folder, old_name)
        new_path = os.path.join(output_folder, new_name)
        
        # Check if the new filename would be too long for the filesystem
        if len(new_path) > 255:
            print(f"Warning: New filename too long for {old_name}, using shorter version")
            # Create a shorter version by truncating comment
            new_name_parts = new_name[:-4].split("_")
            if len(new_name_parts) > 8:  # If there's a comment
                new_name_parts[8] = new_name_parts[8][:20] + "..."
                new_name = "_".join(new_name_parts) + ".mp4"
                new_path = os.path.join(output_folder, new_name)
        
        print(f"{old_name}  →  {new_name}")
        
        try:
            shutil.copy2(old_path, new_path)
            processed += 1
        except Exception as e:
            print(f"Error copying {old_name}: {e}")
            skipped += 1
    
    print(f"\n✔ Done! Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__":
    main()