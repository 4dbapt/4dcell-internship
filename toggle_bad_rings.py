import os
import cv2
import pandas as pd
from tkinter import filedialog, Tk

def select_folder(title="Select output folder"):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    return folder

def get_corresponding_csv(video_path, csv_root):
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    for file in os.listdir(csv_root):
        if file.startswith(prefix) and file.endswith("_analysis_aggregated.csv"):
            return os.path.join(csv_root, file)
    return None

def review_video(video_path):
    print(f"\nReviewing {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    bad_rings = set()

    instructions = "Press A–I to toggle bad rings. ENTER to confirm, ESC to skip."
    print(instructions)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        # Display text overlay
        overlay = frame.copy()
        cv2.putText(overlay, f"Bad rings: {', '.join(sorted(bad_rings)) if bad_rings else 'None'}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Video Review", overlay)

        key = cv2.waitKey(50) & 0xFF
        if key == 13:  # Enter key
            break
        elif key == 27:  # ESC
            bad_rings.clear()
            break
        elif 97 <= key <= 105:  # a–i
            ring = chr(key).upper()
            if ring in bad_rings:
                bad_rings.remove(ring)
            else:
                bad_rings.add(ring)

    cap.release()
    cv2.destroyAllWindows()
    return bad_rings

def filter_csv(csv_path, bad_rings):
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    filtered_df = df[~df["Ring ID"].isin(bad_rings)]
    filtered_path = csv_path.replace("_analysis_aggregated.csv", "_filtered.csv")
    filtered_df.to_csv(filtered_path, index=False)
    print(f"Filtered {csv_path}: removed {initial_rows - len(filtered_df)} rows -> {filtered_path}")
    return filtered_path

def main():
    folder = select_folder("Select the output folder containing videos/ and CSV/")
    videos_folder = os.path.join(folder, "videos")
    csv_folder = os.path.join(folder, "CSV")

    videos = [os.path.join(videos_folder, v) for v in os.listdir(videos_folder) if v.endswith(".mp4")]
    if not videos:
        print("No videos found.")
        return

    for video_path in videos:
        csv_path = get_corresponding_csv(video_path, csv_folder)
        if not csv_path:
            print(f"No corresponding CSV found for {video_path}")
            continue

        bad_rings = review_video(video_path)
        if bad_rings:
            filter_csv(csv_path, bad_rings)
        else:
            print("No bad rings marked, skipping filtering.")

    print("✅ Review complete. Filtered CSVs saved.")

if __name__ == "__main__":
    main()
