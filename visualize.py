import cv2
import numpy as np
from Dataset import YouTubeVideoDataset

def visualize_landmarks(landmark_sequence, label):
    """
    Visualizes a sequence of landmarks on a blank canvas.

    Args:
    - landmark_sequence (numpy.ndarray): Shape (num_frames, 1632)
    - label (int): The class label of the action
    """
    canvas_size = 600  # Size of the blank canvas
    frame_delay = 50  # Delay between frames in ms

    # Iterate through frames
    for frame in landmark_sequence:
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)  # Black canvas

        # Extract pose, face, hands from flattened array
        pose = frame[:99].reshape(-1, 3)  # (33, 3)
        face = frame[99:1503].reshape(-1, 3)  # (468, 3)
        left_hand = frame[1503:1566].reshape(-1, 3)  # (21, 3)
        right_hand = frame[1566:].reshape(-1, 3)  # (21, 3)

        # Scale landmarks to fit on the canvas
        def scale_points(points, scale=250, offset=300):
            return np.array([[int(x * scale) + offset, int(y * scale) + offset] for x, y, _ in points])

        # Convert landmarks to 2D pixel coordinates
        pose_points = scale_points(pose)
        face_points = scale_points(face, scale=100, offset=300)
        left_hand_points = scale_points(left_hand, scale=100, offset=150)
        right_hand_points = scale_points(right_hand, scale=100, offset=450)

        # Draw landmarks on the canvas
        for pt in pose_points:
            cv2.circle(canvas, tuple(pt), 3, (255, 255, 255), -1)

        for pt in face_points:
            cv2.circle(canvas, tuple(pt), 1, (0, 255, 0), -1)

        for pt in left_hand_points:
            cv2.circle(canvas, tuple(pt), 3, (255, 0, 0), -1)

        for pt in right_hand_points:
            cv2.circle(canvas, tuple(pt), 3, (0, 0, 255), -1)

        # Display label
        cv2.putText(canvas, f"Label: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Landmark Visualization", canvas)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cv2.destroyAllWindows()

def main():
    print("Starting dataset processing...")
    dataset = YouTubeVideoDataset(json_file="Data/MS-ASL/MSASL_train.json", frame_skip=10)

    # Convert to TensorFlow Dataset with variable-length sequences
    tf_dataset = dataset.get_tf_dataset(batch_size=4)

    # Get first batch
    for batch in tf_dataset.take(1):
        landmarks, labels = batch  # batch is a tuple (landmarks, labels)

        # Convert each to NumPy arrays
        landmarks = landmarks.numpy()
        labels = labels.numpy()
        break

    # Select first sample from batch
    first_landmark_sequence = landmarks[1]  # Shape: (num_frames, 1632)
    first_label = labels[1]

    print(f"Visualizing first sample with label {first_label}...")
    visualize_landmarks(first_landmark_sequence, first_label)

if __name__ == "__main__":
    main()
