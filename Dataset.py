import os
import json
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from Preprocessing.preprocess_url import get_direct_url

# Initialize MediaPipe Holistic Model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class YouTubeVideoDataset:
    def __init__(self, json_file, frame_skip=5, temp_dir="temp_videos"):
        """
        TensorFlow Dataset to fetch MediaPipe landmarks from YouTube videos.

        Args:
        - json_file (str): Path to JSON file with video URLs and labels.
        - frame_skip (int): Number of frames to skip.
        - temp_dir (str): Directory to store downloaded videos temporarily.
        """
        self.frame_skip = frame_skip
        self.temp_dir = temp_dir

        # Load video URLs and labels from JSON
        with open(json_file, "r") as f:
            self.video_data = json.load(f)

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

    # def get_video_path(self, youtube_url):
    #     """Download video and return file path."""
    #     try:
    #         command = [
    #             "yt-dlp",
    #             "-f", "best[ext=mp4]",
    #             "-o", f"{self.temp_dir}/%(title)s.%(ext)s",
    #             youtube_url
    #         ]
    #         subprocess.run(command, capture_output=True, text=True, check=True)
    #
    #         # Find the downloaded file
    #         files = os.listdir(self.temp_dir)
    #         for file in files:
    #             if file.endswith(".mp4"):
    #                 return os.path.join(self.temp_dir, file)
    #
    #     except Exception as e:
    #         print(f"Failed to download {youtube_url}: {e}")
    #     return None

    def extract_landmarks(self, video_path):
        """Extract holistic landmarks from video frames."""
        holistic = mp_holistic.Holistic(static_image_mode=False)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        sequence = []  # Store frame-wise landmarks

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)

                # Extract pose, hands, face landmarks
                landmarks = self.process_landmarks(results)
                sequence.append(landmarks)

            frame_count += 1

        cap.release()
        holistic.close()
        return np.array(sequence, dtype=np.float32)  # Shape: (num_frames, total_landmarks)

    def process_landmarks(self, results):
        """Extract MediaPipe landmarks and flatten into a feature vector."""

        def extract_coords(landmark_list, num_landmarks):
            """Helper function to get (x, y, z) from landmarks."""
            if landmark_list:
                return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]).flatten()
            else:
                return np.zeros(num_landmarks * 3)  # If missing, pad with zeros

        # Pose: 33 landmarks (99 values)
        pose = extract_coords(results.pose_landmarks, 33)

        # Face: 468 landmarks (1404 values)
        face = extract_coords(results.face_landmarks, 468)

        # Hands: 21 landmarks each (126 values total)
        left_hand = extract_coords(results.left_hand_landmarks, 21)
        right_hand = extract_coords(results.right_hand_landmarks, 21)

        return np.concatenate([pose, face, left_hand, right_hand])  # Final feature vector

    def generator(self):
        """Generator function for TensorFlow Dataset, yielding (landmarks, label)."""
        for data in self.video_data:

            if data["signer_id"]!=0:
                continue

            youtube_url = data["url"]
            label = data["label"]

            video_path = get_direct_url(youtube_url)
            if video_path:
                landmarks = self.extract_landmarks(video_path)

                # Remove the video after processing
                # os.remove(video_path)

                yield landmarks, label  # Return landmarks and label

                # print("Generated data for:", youtube_url)

    # def get_tf_dataset(self, batch_size=4, shuffle=True):
    #     """Create a TensorFlow Dataset with (landmarks, label)."""
    #     dataset = tf.data.Dataset.from_generator(
    #         self.generator,
    #         output_signature=(
    #             tf.TensorSpec(shape=(None, 1629), dtype=tf.float32),  # Landmarks (pose + face + hands)
    #             tf.TensorSpec(shape=(), dtype=tf.int32)  # Label
    #         )
    #     )
    #
    #     if shuffle:
    #         dataset = dataset.shuffle(buffer_size=10)
    #
    #     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #     return dataset

    def get_tf_dataset(self, batch_size=4, shuffle=True):
        """Create a TensorFlow Dataset with variable number of frames per video."""
        dataset = tf.data.Dataset.from_generator(
            lambda: self.generator(),  # Ensure function is called correctly
            output_signature=(
                tf.TensorSpec(shape=(None, 1629), dtype=tf.float32),  # Variable frames, fixed landmarks
                tf.TensorSpec(shape=(), dtype=tf.int32)  # Label
            )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10)

        # Apply dynamic padding for batching videos with different frame counts
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 1629], []))

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def main():
    # Create dataset instance
    file_path = "Data/MS-ASL/MSASL_train.json"
    dataset = YouTubeVideoDataset(json_file=file_path, frame_skip=10)

    # Convert to TensorFlow Dataset
    tf_dataset = dataset.get_tf_dataset(batch_size=4)

    # Example: Fetch a batch
    for batch in tf_dataset:
        landmarks, labels = batch
        print("Landmarks shape:", landmarks.shape)  # (batch_size, num_frames, 1632)
        print("Labels shape:", labels.shape)  # (batch_size,)
        # break

if __name__ == "__main__":
    main()
