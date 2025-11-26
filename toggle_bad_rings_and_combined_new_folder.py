import os
import shutil
import cv2
import pandas as pd
from tkinter import filedialog, Tk

# ============================================================
# CONFIG
# ============================================================
BASE_OUTPUT_ROOT = r"Y:\RnD\Baptiste\Example analyses\Contractility\Output"

# ============================================================
# 1) Select input folder
# ============================================================

def select_folder(title="Select the root folder containing videos/ and CSV/"):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

# ============================================================
# 2) Compute dynamic output root (Output\<InputFolderName>)
# ============================================================

def compute_output_root(input_root):
    """
    Given an input folder like
      ...\Input\Batch42
    return
      BASE_OUTPUT_ROOT\Batch42
    and ensure it exists.
    """
    input_name = os.path.basename(os.path.normpath(input_root))
    output_root = os.path.join(BASE_OUTPUT_ROOT, input_name)
    os.makedirs(output_root, exist_ok=True)
    return output_root

# ============================================================
# 3) Copy input -> output (preserve everything)
# ============================================================

def copy_input_to_output(input_root, output_root):
    """
    Copy the entire input folder contents into the output_root.
    Uses shutil.copy2 for files and shutil.copytree for directories
    with dirs_exist_ok=True (Python 3.8+).
    """
    # iterate items in input_root and copy under output_root
    for item in os.listdir(input_root):
        src = os.path.join(input_root, item)
        dst = os.path.join(output_root, item)

        if os.path.isdir(src):
            # copy tree (merge if dst exists)
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            # file: ensure parent exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    print(f"\n📁 Input folder contents copied to output: {output_root}\n")

# ============================================================
# 4) Matching the aggregated CSV to a video
# ============================================================

def get_corresponding_csv(video_path, csv_root):
    """
    Find the video-specific aggregated CSV corresponding to the video,
    e.g. video 'Batch42_250924_120415_B1.mp4' -> 'Batch42_250924_120415_B1_analysis_aggregated.csv'
    """
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    if not os.path.isdir(csv_root):
        return None

    for file in os.listdir(csv_root):
        if file.startswith(prefix) and file.endswith("_analysis_aggregated.csv"):
            return os.path.join(csv_root, file)
    return None

# ============================================================
# 5) Video review – user marks bad rings (A-I)
# ============================================================

def review_video(video_path):
    print(f"\n🎬 Reviewing {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return set()

    bad_rings = set()

    print("➡️ Controls:")
    print("   - Press A–I to toggle bad rings.")
    print("   - ENTER to confirm selection.")
    print("   - ESC to skip.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        overlay = frame.copy()
        text = f"Bad rings: {', '.join(sorted(bad_rings)) if bad_rings else 'None'}"
        cv2.putText(overlay, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        cv2.imshow("Video Review", overlay)
        key = cv2.waitKey(60) & 0xFF

        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            bad_rings.clear()
            break
        elif 97 <= key <= 105:  # keys a–i
            ring = chr(key).upper()
            if ring in bad_rings:
                bad_rings.remove(ring)
            else:
                bad_rings.add(ring)

    cap.release()
    cv2.destroyAllWindows()
    return bad_rings

# ============================================================
# 6) Apply filtering & save into output CSV folder
# ============================================================

def filter_csv(input_csv_path, output_root, bad_rings):
    """
    Read input_csv_path (already copied to output), filter rows where Ring ID is in bad_rings,
    and write a *_filtered.csv next to the original within the output CSV folder.
    """
    if not os.path.exists(input_csv_path):
        print(f"⚠️ CSV not found: {input_csv_path}")
        return None

    df = pd.read_csv(input_csv_path)
    if "Ring ID" not in df.columns:
        print(f"⚠️ Missing column 'Ring ID' in {input_csv_path}")
        return None

    filtered_df = df[~df["Ring ID"].isin(bad_rings)]

    # Determine output CSV path: keep same folder structure under output_root/CSV
    # input_csv_path is already under output_root in our workflow, so just change suffix
    base, ext = os.path.splitext(input_csv_path)
    output_csv_path = base + "_filtered" + ext

    filtered_df.to_csv(output_csv_path, index=False)
    removed = len(df) - len(filtered_df)
    print(f"✅ {os.path.basename(input_csv_path)} → removed {removed} rows -> {os.path.basename(output_csv_path)}")
    return output_csv_path

# ============================================================
# 7) Generate combined global summary from filtered files
# ============================================================

def create_global_filtered_csv(output_root):
    """
    Concatenate all *_analysis_aggregated_filtered.csv found in
    output_root/CSV/ into output_root/CSV/combined/global_results_summary_filtered.csv
    """
    csv_folder = os.path.join(output_root, "CSV")
    if not os.path.isdir(csv_folder):
        print("⚠️ No CSV folder in output, skipping combined creation.")
        return None

    # collect filtered files in CSV root (not in subfolders)
    filtered_files = []
    for root_dir, _, files in os.walk(csv_folder):
        for f in files:
            if f.endswith("_analysis_aggregated_filtered.csv"):
                filtered_files.append(os.path.join(root_dir, f))

    if not filtered_files:
        print("⚠️ No *_analysis_aggregated_filtered.csv found. Skipping global summary.")
        return None

    dfs = []
    for f in filtered_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}")

    if not dfs:
        print("⚠️ No readable filtered CSVs found.")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_folder = os.path.join(csv_folder, "combined")
    os.makedirs(combined_folder, exist_ok=True)

    output_path = os.path.join(combined_folder, "global_results_summary_filtered.csv")
    combined_df.to_csv(output_path, index=False)

    print(f"\n🎉 Combined filtered summary created:\n{output_path}\n")
    return output_path

# ============================================================
# 8) Main workflow
# ============================================================

def main():
    input_root = select_folder()
    if not input_root:
        print("❌ No folder selected.")
        return

    # compute dynamic output root: ...\Output\<InputFolderName>
    output_root = compute_output_root(input_root)
    print(f"\n📁 Output root will be: {output_root}\n")

    # Copy everything from input_root into output_root (merge)
    print("📂 Copying input contents into output folder...")
    copy_input_to_output(input_root, output_root)

    # We'll operate on the videos and CSV inside the output_root
    videos_folder = os.path.join(output_root, "videos")
    csv_folder = os.path.join(output_root, "CSV")

    if not os.path.isdir(videos_folder):
        print(f"❌ No videos folder found at {videos_folder}")
        return

    # list mp4 videos (non-recursive)
    videos = [os.path.join(videos_folder, v) for v in os.listdir(videos_folder) if v.lower().endswith(".mp4")]
    if not videos:
        print("❌ No .mp4 videos found in videos/.")
        return

    # For each video: find matched aggregated CSV in csv_folder (non-recursive),
    # let user mark bad rings, and write *_filtered.csv into the same CSV folder
    for video_path in videos:
        csv_path = get_corresponding_csv(video_path, csv_folder)
        if not csv_path:
            print(f"⚠️ No matching aggregated CSV found for {os.path.basename(video_path)} (looking in {csv_folder})")
            continue

        bad_rings = review_video(video_path)
        if bad_rings:
            filter_csv(csv_path, output_root, bad_rings)
        else:
            print(f"👌 {os.path.basename(video_path)} → no bad rings marked")

    # After all videos processed, create the combined global summary from filtered per-video CSVs
    create_global_filtered_csv(output_root)
    print("🎉 All done — filtered files are in the output folder.")

if __name__ == "__main__":
    main()
