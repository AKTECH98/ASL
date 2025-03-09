import json
import os
import yt_dlp
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = os.getenv("DOWNLOAD_DIR")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_video(url, output_file):
    ydl_opts = {
        "outtmpl": output_file,
        "format": "best",  # You can change to 'bestvideo+bestaudio' for better quality
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    downloaded = 0
    failed = 0
    copy = 0
    total_videos = len(data)

    with tqdm(total=total_videos, desc="Downloading Videos", unit="video") as pbar:
        for video in data:
            if video["url"] is None:
                pbar.update(1)
                continue

            label = video["label"]

            file_dir = os.path.join(OUTPUT_DIR,"train/"+str(label))
            os.makedirs(file_dir, exist_ok=True)

            url = video["url"]
            vid_name = str(downloaded)

            output_file = os.path.join(file_dir, vid_name + ".mp4")

            if os.path.exists(output_file):
                print(f"Video {vid_name} already exists. Skipping download.")
                copy += 1
                pbar.update(1)
                continue

            print(f"Downloading {vid_name}")
            try:
                download_video(url, output_file)
            except Exception as e:
                print(f"Failed to download {vid_name}: {e}")
                failed += 1
                pbar.update(1)
                continue

            print(f"Downloaded {vid_name} to {output_file}")
            downloaded += 1
            pbar.update(1)

    print(f"Downloaded {downloaded} videos, failed to download {failed} videos, Duplicate {copy} videos.")


if __name__ == "__main__":
    file_path = "Data/MS-ASL/MSASL_train.json"
    main(file_path)
