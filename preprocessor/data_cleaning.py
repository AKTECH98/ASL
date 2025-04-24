import os
import json
import cv2 as cv
from tqdm import tqdm
from utils import get_video_url, download_video
import argparse

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("last_index", -1)
    return -1

def save_checkpoint(index, path):
    with open(path, "w") as f:
        json.dump({"last_index": index}, f)

def load_cleaned_data(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def video_review(video_path, label_name, start_frame, end_frame, show_video=False):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if show_video:
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Could not read frame {frame_num} in {video_path}")
                break

            cv.putText(frame, f"Label: {label_name}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow("Video Review", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
    return True

def process_video(video, cleaned_data, show_video=False):
    url = video["url"]
    label_name = video["clean_text"]
    start_frame, end_frame = video["start"], video["end"]

    try:
        video_file = get_video_url(url)
        if video_file and video_review(video_file, label_name, start_frame, end_frame, show_video):
            cleaned_data.append(video)
            return True

        print("🔄 Attempting download...")
        video_path = download_video(url)
        if video_path and video_review(video_path, label_name, start_frame, end_frame, show_video):
            cleaned_data.append(video)
            return True

    except Exception as e:
        print(f"[ERROR] Failed processing {url}: {e}")
    return False

def clean_dataset(raw_data_path, clean_data_path, checkpoint_path, show_video=False):
    cleaned_data = load_cleaned_data(clean_data_path)
    last_index = load_checkpoint(checkpoint_path)

    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)

    start_index = last_index + 1
    last_idx = last_index

    try:
        for idx in tqdm(range(start_index, len(raw_data)), desc="Processing videos"):
            if process_video(raw_data[idx], cleaned_data, show_video):
                print("✅ Added video to cleaned data. Total:", len(cleaned_data))
                last_idx = idx
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        with open(clean_data_path, "w") as f:
            json.dump(cleaned_data, f, indent=4)

        save_checkpoint(last_idx, checkpoint_path)
        print(f"\nCleaning complete. Videos added: {len(cleaned_data)}")
        print(f"Last processed index saved: {last_idx}")

def main():
    parser = argparse.ArgumentParser(description="Clean MS-ASL dataset videos.")
    parser.add_argument('--raw_data', type=str, default="../Data/MS-ASL/MSASL_test.json", help="Path to raw MS-ASL data")
    parser.add_argument('--cleaned_data', type=str, default="../Data/MS-ASL-Clean-Data/test.json", help="Path to cleaned output data")
    parser.add_argument('--checkpoint', type=str, default="../Data/MS-ASL-Clean-Data/checkpoint.json", help="Checkpoint file to resume data cleaning")
    parser.add_argument('--show_video', action='store_true', help="Enable visual review of each video")

    args = parser.parse_args()

    clean_dataset(
        raw_data_path=args.raw_data,
        clean_data_path=args.cleaned_data,
        checkpoint_path=args.checkpoint,
        show_video=args.show_video
    )

if __name__ == "__main__":
    main()
