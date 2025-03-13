import cv2 as cv
import numpy as np
import mediapipe as mp


def process_landmarks(results):
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

    return np.concatenate([pose, face, left_hand, right_hand])

def visualize_landmarks(landmark_sequence):

    sequence = np.load(landmark_sequence)

    canvas_size = 500  # Size of the blank canvas

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)  # Black canvas

    frame = 1
    for landmark in sequence:
        # Extract pose, face, hands from flattened array
        pose = landmark[:99].reshape(-1, 3)
        face = landmark[99:1503].reshape(-1, 3)
        left_hand = landmark[1503:1566].reshape(-1, 3)
        right_hand = landmark[1566:].reshape(-1, 3)

        # Scale landmarks to fit on the canvas
        def scale_points(points, scale=canvas_size, offset=0):
            return np.array([[int(x * scale) + offset, int(y * scale) + offset] for x, y, _ in points])

        # Convert landmarks to 2D pixel coordinates
        pose_points = scale_points(pose)
        face_points = scale_points(face)
        left_hand_points = scale_points(left_hand)
        right_hand_points = scale_points(right_hand)

        landmark_canvas = np.copy(canvas)

        # Draw landmarks on the canvas
        for pt in pose_points:
            cv.circle(landmark_canvas, tuple(pt), 3, (255, 255, 255), -1)

        for pt in face_points:
            cv.circle(landmark_canvas, tuple(pt), 1, (0, 255, 0), -1)

        for pt in left_hand_points:
            cv.circle(landmark_canvas, tuple(pt), 3, (255, 0, 0), -1)

        for pt in right_hand_points:
            cv.circle(landmark_canvas, tuple(pt), 3, (0, 0, 255), -1)

        # Show frame
        cv.imshow("Landmark Visualization", landmark_canvas)

        frame += 1

        if frame==30:
            cv.imwrite(f"Data/downloaded_videos/landmark_visualization{canvas_size}.jpg",landmark_canvas)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


def main():

    file = "Data/downloaded_videos/match [light-a-MATCH].mp4"

    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv.VideoCapture(file)

    sequence = []  # Store frame-wise landmarks

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistic.process(frame)

        # Extract pose, hands, face landmarks
        landmarks = process_landmarks(results)
        sequence.append(landmarks)

    cap.release()
    holistic.close()
    np.save("Data/downloaded_videos/landmarks_match",np.array(sequence,dtype=np.float32))  # Shape: (num_frames, total_landmarks)

if __name__ == "__main__":
    # main()
    visualize_landmarks("Data/downloaded_videos/landmarks_match.npy")