import os
import ssl
import yt_dlp
import certifi
import subprocess
import urllib.request
import random
import torch
import numpy as np

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_video_url(url):

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    urllib.request.urlopen("https://www.youtube.com", context=ssl_context)

    command = [
        "yt-dlp",
        "-f", "best[ext=mp4]",  # Get the best available MP4 format
        "-g", url  # Extract the direct video URL
    ]

    try:
        video_url = subprocess.run(command, capture_output=True, text=True, check=True)
        return video_url.stdout.strip()
    except Exception as e:
        print(f"Failed to get direct URL for {url}: {e}")
        return None

def download_video(url, output_dir = "../temp_videos"):

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "0.mp4"),
        "format": "best[ext=mp4]",
        "quiet": True,
        "no_warnings": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("DOWNLOAD SUCCESSFUL")
        return os.path.join(output_dir, "0.mp4")
    except Exception as e:
        print(f"Failed to download video: {e}")
        return None

def get_clipped_segment(youtube_url, start_time, end_time, output_path="../temp_videos/segment_clip.mp4"):
    video_url = get_video_url(youtube_url)
    if not video_url:
        print("Failed to get stream URL.")
        return None

    duration = end_time - start_time

    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_url,
        "-t", str(duration),
        "-c:v", "libx264",  # Ensures compatibility
        "-c:a", "aac",
        "-y",  # Overwrite
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        print(f"Clipped segment saved at {output_path}")
        return output_path
    except Exception as e:
        print(f"Error clipping video: {e}")
        return None

def main():

    output_path = get_clipped_segment(
        "https://www.youtube.com/watch?v=HPz_C5XM4o4",
        start_time=4.991,
        end_time=6.61
    )

    print(output_path)

if __name__ == "__main__":
    main()


