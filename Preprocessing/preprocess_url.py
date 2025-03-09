import subprocess
import ssl
import certifi
import urllib.request

ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.urlopen("https://www.youtube.com", context=ssl_context)

def get_direct_url(youtube_url):
    try:
        command = [
            "yt-dlp",
            "-f", "best[ext=mp4]",  # Get the best available MP4 format
            "-g", youtube_url       # Extract the direct video URL
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()  # Extracted direct URL
    except Exception as e:
        print(f"Failed to get direct URL for {youtube_url}: {e}")
        return None