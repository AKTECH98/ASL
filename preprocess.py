import os
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import argparse
import contextlib
import yaml

from utils import get_clipped_segment
from Modules.skeleton import SkeletonLandmarks

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield

def filter_data(data, min_samples=1, banned_ids=None):
    """Filter samples by banned signer IDs and minimum label count."""
    if banned_ids is None:
        banned_ids = [46, 32, 36, 18, 88, 172, 9]

    filtered = [s for s in data if s['signer_id'] not in banned_ids]
    label_counts = Counter(s['label'] for s in filtered)
    valid_labels = {label for label, count in label_counts.items() if count >= min_samples}
    return [s for s in filtered if s['label'] in valid_labels]

def save_json(data, path):
    """Save data to JSON file with proper directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_dataset(data, config):
    """Preprocess the dataset according to configuration."""
    skeleton = SkeletonLandmarks()
    label_counts = defaultdict(int)
    invalid_data = []
    output_dir = os.path.join(config['output']['dir'], f"Data{config['input']['min_samples']}")
    os.makedirs(output_dir, exist_ok=True)

    for video in tqdm(data, desc="Extracting features"):
        try:
            with suppress_all_output():
                video_path = get_clipped_segment(video["url"], video["start_time"], video["end_time"])

            if not video_path or not os.path.exists(video_path):
                invalid_data.append(video)
                continue

            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                invalid_data.append(video)
                continue

            sequence = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                skeleton.extract_landmarks(frame)
                features = skeleton.get_features()
                if features is not None:
                    sequence.append(features)
                    if config['processing']['show_video']:
                        canvas = np.zeros_like(frame)
                        vis = skeleton.draw_landmarks(canvas)
                        cv.putText(frame, f"{video['label_class']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv.putText(vis, f"Signer Id: {video['signer_id']}", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        combined = np.hstack((frame, vis))
                        cv.imshow("Frame", combined)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break

            cap.release()
            if config['processing']['show_video']:
                cv.destroyAllWindows()

            if sequence:
                label = video["label"]
                label_name = video["label_class"]
                label_dir = os.path.join(output_dir, label_name)
                os.makedirs(label_dir, exist_ok=True)

                if label_counts[label] == 0:
                    label_counts[label] = len(os.listdir(label_dir))

                file_idx = label_counts[label]
                output_path = os.path.join(label_dir, f"{file_idx}.npy")
                np.save(output_path, np.array(sequence))
                label_counts[label] += 1

                print(f"✅ Saved: {output_path} | Shape: {np.array(sequence).shape}")
            else:
                invalid_data.append(video)

        except Exception as e:
            print(f"[ERROR] Failed video: {video['url']}\nReason: {e}")
            invalid_data.append(video)

    print(f"\nFinished: {sum(label_counts.values())} samples across {len(label_counts)} labels.")
    print(f"Invalid: {len(invalid_data)} samples.")
    save_json(invalid_data, os.path.join(output_dir, "invalid_data.json"))

def main():
    parser = argparse.ArgumentParser(description="MS-ASL Dataset Preprocessing")
    parser.add_argument("--config", type=str, default="config/preprocess.yaml", help="Path to config file")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override show_video from command line if specified
    if args.show_video:
        config['processing']['show_video'] = True

    # Create necessary directories
    os.makedirs(config['output']['dir'], exist_ok=True)

    # Load and process data
    input_json = os.path.join(config['input']['data_path'])
    output_json = os.path.join(config['output']['dir'], f"preprocessed_data{config['input']['min_samples']}.json")

    if os.path.exists(output_json):
        print(f"📂 Loading existing cleaned data: {output_json}")
        data = load_json(output_json)
    else:
        clean_data = load_json(input_json)
        data = filter_data(clean_data, 
                         min_samples=config['input']['min_samples'],
                         banned_ids=config['input']['banned_ids'])
        save_json(data, output_json)

    preprocess_dataset(data, config)

if __name__ == "__main__":
    main()
