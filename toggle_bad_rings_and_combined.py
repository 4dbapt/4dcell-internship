import os
import cv2
import pandas as pd
from tkinter import filedialog, Tk

def select_folder(title="Select output folder"):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def get_corresponding_csv(video_path, csv_root):
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    for file in os.listdir(csv_root):
        if file.startswith(prefix) and file.endswith("_analysis_aggregated.csv"):
            return os.path.join(csv_root, file)
    return None

def review_video(video_path):
    print(f"\n🎬 Reviewing {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    bad_rings = set()

    print("➡️ Controls:")
    print("   - Press A–I to toggle bad rings.")
    print("   - ENTER to confirm selection.")
    print("   - ESC to skip this video.\n")

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

def filter_csv(csv_path, bad_rings):
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if "Ring ID" not in df.columns:
        print(f"⚠️ Column 'Ring ID' not found in {csv_path}")
        return None

    filtered_df = df[~df["Ring ID"].isin(bad_rings)]
    filtered_path = csv_path.replace(".csv", "_filtered.csv")
    filtered_df.to_csv(filtered_path, index=False)
    removed = len(df) - len(filtered_df)
    print(f"✅ {os.path.basename(csv_path)} → removed {removed} rows ({len(filtered_df)} kept)")
    return filtered_path

def create_global_filtered_csv(csv_folder, output_name="global_results_summary_filtered.csv"):
    """
    Concatenate all *_analysis_aggregated_filtered.csv to generate a new global summary CSV.
    """
    filtered_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder)
                      if f.endswith("_analysis_aggregated_filtered.csv")]

    if not filtered_files:
        print("⚠️ No filtered CSVs found to create global summary.")
        return None

    dfs = [pd.read_csv(f) for f in filtered_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save in CSV/combined/
    combined_folder = os.path.join(csv_folder, "combined")
    os.makedirs(combined_folder, exist_ok=True)
    output_path = os.path.join(combined_folder, output_name)
    combined_df.to_csv(output_path, index=False)
    print(f"\n🎉 Combined global summary saved: {output_path}")
    return output_path

def main():
    folder = select_folder("Select the output folder containing videos/ and CSV/")
    videos_folder = os.path.join(folder, "videos")
    csv_folder = os.path.join(folder, "CSV")

    if not os.path.exists(videos_folder) or not os.path.exists(csv_folder):
        print("❌ Invalid folder structure. Expecting 'videos/' and 'CSV/'.")
        return

    videos = [os.path.join(videos_folder, v) for v in os.listdir(videos_folder) if v.endswith(".mp4")]
    if not videos:
        print("❌ No .mp4 videos found.")
        return

    for video_path in videos:
        csv_path = get_corresponding_csv(video_path, csv_folder)
        if not csv_path:
            print(f"⚠️ No matching CSV found for {os.path.basename(video_path)}")
            continue

        bad_rings = review_video(video_path)
        if bad_rings:
            filter_csv(csv_path, bad_rings)
        else:
            print(f"👌 {os.path.basename(video_path)} → no bad rings marked.")

    # After all videos are processed, generate combined global summary from filtered CSVs
    create_global_filtered_csv(csv_folder)

if __name__ == "__main__":
    main()
