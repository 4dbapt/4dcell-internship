import os
import cv2
import pandas as pd
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import Tk, Toplevel, Frame, Label, Button, Checkbutton, IntVar

# ============================================================
# 1) Choose folders
# ============================================================

def select_folder(title):
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

# ============================================================
# 2) Match video -> CSV
# ============================================================

def get_corresponding_csv(video_path, csv_root):
    prefix = os.path.splitext(os.path.basename(video_path))[0]

    if not os.path.isdir(csv_root):
        return None

    for file in os.listdir(csv_root):
        if file.startswith(prefix) and file.endswith("_analysis_aggregated.csv"):
            return os.path.join(csv_root, file)

    return None

# ============================================================
# 3) Threaded video player (LEFT SIDE)
# ============================================================

def play_video_thread(video_path, stop_flag):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    # Force window to appear immediately
    cv2.namedWindow("Video Review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Review", 960, 540)
    cv2.moveWindow("Video Review", 0, 0)   # LEFT SIDE

    while True:
        if stop_flag["stop"]:
            break

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow("Video Review", frame)

        if cv2.waitKey(20) & 0xFF == 27:
            stop_flag["stop"] = True
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# 4) Tkinter ring selection grid (RIGHT SIDE)
# ============================================================

def ask_rings_gui(video_name):
    win = Toplevel()
    win.title(f"Exclude rings for {video_name}")

    # Get screen size
    win.update_idletasks()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()

    # Window size
    win_w = 350
    win_h = 400

    # RIGHT-SIDE placement
    pos_x = screen_w - win_w - 50
    pos_y = 50

    win.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

    Label(win, text="Select bad rings:", font=("Arial", 14)).grid(row=0, column=0, columnspan=3, pady=10)

    letters = ["A","B","C","D","E","F","G","H","I"]
    vars_dict = {}

    for i, letter in enumerate(letters):
        var = IntVar()
        vars_dict[letter] = var
        r = (i // 3) + 1
        c = i % 3
        chk = Checkbutton(win, text=letter, variable=var, font=("Arial", 16))
        chk.grid(row=r, column=c, padx=15, pady=10)

    selected = set()

    def validate():
        for k, v in vars_dict.items():
            if v.get() == 1:
                selected.add(k)
        win.destroy()

    Button(win, text="OK", command=validate, font=("Arial", 16)).grid(row=4, column=0, columnspan=3, pady=20)

    win.wait_window()
    return selected

# ============================================================
# 5) Review video AND show grid simultaneously
# ============================================================

def review_video(video_path):
    print(f"\n🎬 Reviewing {os.path.basename(video_path)}")
    print("➡ Video is playing (left side)")
    print("➡ Tkinter grid appears (right side)")
    print("➡ Click OK when done\n")

    stop_flag = {"stop": False}

    # Start video thread
    thread = threading.Thread(target=play_video_thread, args=(video_path, stop_flag))
    thread.start()

    # Show Tkinter grid
    bad_rings = ask_rings_gui(os.path.basename(video_path))

    stop_flag["stop"] = True
    thread.join()

    return bad_rings

# ============================================================
# 6) Filter CSV
# ============================================================

def filter_csv(input_csv_path, output_csv_dir, bad_rings):
    if not os.path.exists(input_csv_path):
        print(f"⚠ CSV not found: {input_csv_path}")
        return None

    df = pd.read_csv(input_csv_path)
    if "Ring ID" not in df.columns:
        print(f"⚠ Missing column 'Ring ID' in {input_csv_path}")
        return None

    filtered_df = df[~df["Ring ID"].isin(bad_rings)]

    base = os.path.basename(input_csv_path)
    base_no_ext, ext = os.path.splitext(base)

    output_path = os.path.join(output_csv_dir, base_no_ext + "_filtered" + ext)
    filtered_df.to_csv(output_path, index=False)

    removed = len(df) - len(filtered_df)
    print(f"✅ {base}: removed {removed} rows → {os.path.basename(output_path)}")

    return output_path

# ============================================================
# 7) Combine all filtered CSVs
# ============================================================

def create_global_filtered_csv(output_csv_dir):
    filtered_files = [
        os.path.join(output_csv_dir, f)
        for f in os.listdir(output_csv_dir)
        if f.endswith("_analysis_aggregated_filtered.csv")
    ]

    if not filtered_files:
        print("⚠ No filtered CSVs found. No combined summary created.")
        return None

    dfs = []
    for f in filtered_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠ Failed reading {f}: {e}")

    if not dfs:
        print("⚠ No readable filtered CSVs.")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_path = os.path.join(output_csv_dir, "global_results_summary_filtered.csv")
    combined_df.to_csv(combined_path, index=False)

    print(f"\n🎉 Global summary created: {combined_path}\n")
    return combined_path

# ============================================================
# 8) Main
# ============================================================

def main():
    root = Tk()
    root.withdraw()

    input_root = filedialog.askdirectory(title="Select INPUT folder (contains videos/ and CSV/)")
    if not input_root:
        print("❌ No input folder selected.")
        return

    output_root = filedialog.askdirectory(title="Select OUTPUT folder for filtered CSVs")
    if not output_root:
        print("❌ No output folder selected.")
        return

    root.destroy()

    videos_folder = os.path.join(input_root, "videos")
    csv_folder = os.path.join(input_root, "CSV")

    if not os.path.isdir(videos_folder):
        print(f"❌ No videos folder found at {videos_folder}")
        return
    if not os.path.isdir(csv_folder):
        print(f"❌ No CSV folder found at {csv_folder}")
        return

    videos = [
        os.path.join(videos_folder, v)
        for v in os.listdir(videos_folder)
        if v.lower().endswith(".mp4")
    ]

    if not videos:
        print("❌ No .mp4 videos found.")
        return

    print(f"\n📁 Writing filtered CSVs to: {output_root}\n")
    os.makedirs(output_root, exist_ok=True)

    for video_path in videos:
        csv_path = get_corresponding_csv(video_path, csv_folder)
        if not csv_path:
            print(f"⚠ No matching CSV for {os.path.basename(video_path)}")
            continue

        bad_rings = review_video(video_path)

        if bad_rings:
            filter_csv(csv_path, output_root, bad_rings)
        else:
            print(f"👌 No rings excluded for {os.path.basename(video_path)}")

    create_global_filtered_csv(output_root)
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
