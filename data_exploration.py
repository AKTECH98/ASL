import cv2
from Preprocessing.preprocess_url import get_direct_url
import numpy as np
import json
import tqdm


class VideoStatsAnalyzer:
    def __init__(self, file_path):
        self.video_json_file = file_path
        self.video_stats = []

    def analyze_videos(self):
        """Extract statistics from all videos in the directory."""
        with open(self.video_json_file, "r") as f:
            video_data = json.load(f)

        video_entries = [data for data in video_data if data["signer_id"] == 0]

        for data in tqdm.tqdm(video_entries, desc="Processing Videos", unit="video"):
            url = get_direct_url(data["url"])

            cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                print(f"Error opening video file: {url}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = f"{width}x{height}"

            self.video_stats.append({
                "file_name": url,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": resolution,
                "width": width,
                "height": height
            })

            cap.release()

    def compute_overall_statistics(self):
        """Compute aggregate statistics for the video dataset."""
        if not self.video_stats:
            return {}

        fps_values = [vid["fps"] for vid in self.video_stats]
        frame_counts = [vid["frame_count"] for vid in self.video_stats]
        durations = [vid["duration"] for vid in self.video_stats]
        resolutions = [vid["resolution"] for vid in self.video_stats]

        return {
            "total_videos": len(self.video_stats),
            "average_fps": np.mean(fps_values) if fps_values else 0,
            "min_fps": np.min(fps_values) if fps_values else 0,
            "max_fps": np.max(fps_values) if fps_values else 0,
            "average_frames": np.mean(frame_counts) if frame_counts else 0,
            "min_frames": np.min(frame_counts) if frame_counts else 0,
            "max_frames": np.max(frame_counts) if frame_counts else 0,
            "average_duration": np.mean(durations) if durations else 0,
            "min_duration": np.min(durations) if durations else 0,
            "max_duration": np.max(durations) if durations else 0,
            "unique_resolutions": list(set(resolutions))
        }

    def display_statistics(self):
        """Print video statistics and overall analysis."""
        for vid_stat in self.video_stats:
            print(
                f"Video: {vid_stat['file_name']} | FPS: {vid_stat['fps']:.2f} | Frames: {vid_stat['frame_count']} | Duration: {vid_stat['duration']:.2f}s | Resolution: {vid_stat['resolution']}")

        overall_stats = self.compute_overall_statistics()
        print("\nOverall Statistics:")
        for key, value in overall_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")


# Example Usage:

def main():
    file = "Data/MS-ASL/MSASL_val.json"
    analyzer = VideoStatsAnalyzer(file)
    analyzer.analyze_videos()
    analyzer.display_statistics()


if __name__ == "__main__":
    main()