import mediapipe as mp
import cv2 as cv
import numpy as np

class SkeletonLandmarks:
    def __init__(self):
        self._mp_holistic = mp.solutions.holistic
        self._holistic_model = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.6,min_tracking_confidence=0.6)
        self._mp_drawing = mp.solutions.drawing_utils

        self._holistic_landmarks = None

    def extract_landmarks(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self._holistic_landmarks = self._holistic_model.process(rgb_frame)

    def draw_landmarks(self, frame):
        if self._holistic_landmarks:
            self._mp_drawing.draw_landmarks(frame, self._holistic_landmarks.face_landmarks, self._mp_holistic.FACEMESH_TESSELATION,
                                      self._mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      self._mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )
            # Draw pose connections
            self._mp_drawing.draw_landmarks(frame, self._holistic_landmarks.pose_landmarks, self._mp_holistic.POSE_CONNECTIONS,
                                      self._mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      self._mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )
            # Draw left hand connections
            self._mp_drawing.draw_landmarks(frame, self._holistic_landmarks.left_hand_landmarks, self._mp_holistic.HAND_CONNECTIONS,
                                      self._mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      self._mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )
            # Draw right hand connections
            self._mp_drawing.draw_landmarks(frame, self._holistic_landmarks.right_hand_landmarks, self._mp_holistic.HAND_CONNECTIONS,
                                      self._mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      self._mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

        return frame

    def get_features(self, feature_type = ["pose", "face", "left_hand", "right_hand"]):

        def _flatten_landmarks(landmarks, num_landmarks, num_features,visibility=False):
            return np.array([[lm.x, lm.y, lm.z, lm.visibility] if visibility else [lm.x, lm.y, lm.z]
                             for lm in landmarks.landmark]).flatten() if landmarks else np.zeros(num_landmarks*num_features)

        features = []

        if self._holistic_landmarks and self._holistic_landmarks.left_hand_landmarks or self._holistic_landmarks.right_hand_landmarks:

            if "pose" in feature_type:
                pose = _flatten_landmarks(self._holistic_landmarks.pose_landmarks,33,4,True)
                features.append(pose)

            if "face" in feature_type:
                face = _flatten_landmarks(self._holistic_landmarks.face_landmarks,468,3)
                features.append(face)

            if "left_hand" in feature_type:
                lh = _flatten_landmarks(self._holistic_landmarks.left_hand_landmarks,21,3)
                features.append(lh)

            if "right_hand" in feature_type:
                rh = _flatten_landmarks(self._holistic_landmarks.right_hand_landmarks,21,3)
                features.append(rh)

            return np.concatenate(features)

        return None

def main():

    cap = cv.VideoCapture(0)

    skeleton = SkeletonLandmarks()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        skeleton.extract_landmarks(frame)
        frame = skeleton.draw_landmarks(frame)

        features = skeleton.get_features(["left_hand", "right_hand"])
        if features is not None:
            print("Features shape:", features.shape)
        else:
            print("No landmarks detected.")

        cv.imshow("Skeleton", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()