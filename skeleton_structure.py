import cv2 as cv
import mediapipe as mp
import numpy as np


class SkeletonStructure:
    def __init__(self):
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

        self._pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.8)

        self.skeleton_connections = frozenset([(11,12),(12,14),(11,13)])
        self.skeleton_hand_connections = {'Right': 13, 'Left': 14}

        self._hand_results = None
        self._pose_results = None

    def _get_hands(self,frame):
        self._hand_results = self._hands.process(frame)

    def _get_pose(self,frame):
        self._pose_results = self._pose.process(frame)

    def process(self,frame):
        self._get_hands(frame)
        self._get_pose(frame)

        return {}

    def _normalized_to_pixel_coordinates(self,normalized_x, normalized_y, image_width, image_height):
        x_px = min(int(normalized_x * image_width), image_width - 1)
        y_px = min(int(normalized_y * image_height), image_height - 1)
        return (x_px,y_px)

    def _draw_connections(self,img,start,end):
        cv.line(img, start, end, (255, 255, 255), 2)
        cv.circle(img, start, 5, (0, 0, 255), cv.FILLED)
        cv.circle(img, end, 5, (0, 0, 255), cv.FILLED)
        cv.circle(img, start, 6, (255, 255, 255), 1)
        cv.circle(img, end, 6, (255, 255, 255), 1)

    def _draw_skeleton(self,img):

        if self._pose_results.pose_landmarks:
            for connection in self.skeleton_connections:
                start = connection[0]
                end = connection[1]

                start_pt = self._normalized_to_pixel_coordinates(self._pose_results.pose_landmarks.landmark[start].x, self._pose_results.pose_landmarks.landmark[start].y, img.shape[1], img.shape[0])
                end_pt = self._normalized_to_pixel_coordinates(self._pose_results.pose_landmarks.landmark[end].x, self._pose_results.pose_landmarks.landmark[end].y, img.shape[1], img.shape[0])

                self._draw_connections(img,start_pt,end_pt)

            if self._hand_results.multi_hand_landmarks:
                for i,hand_landmarks in enumerate(self._hand_results.multi_hand_landmarks):
                    hand_side = self._hand_results.multi_handedness[i].classification[0].label

                    start = self.skeleton_hand_connections[hand_side]
                    end = 0

                    start_pt = self._normalized_to_pixel_coordinates(self._pose_results.pose_landmarks.landmark[start].x, self._pose_results.pose_landmarks.landmark[start].y, img.shape[1], img.shape[0])
                    end_pt = self._normalized_to_pixel_coordinates(hand_landmarks.landmark[end].x, hand_landmarks.landmark[end].y, img.shape[1], img.shape[0])

                    self._draw_connections(img,start_pt,end_pt)

    def draw(self,img):

        x_ray_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        if self._hand_results.multi_hand_landmarks:
            for hand_landmarks in self._hand_results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(x_ray_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        self._draw_skeleton(x_ray_image)

        return x_ray_image

def main():

    skeleton = SkeletonStructure()

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame for mirror effect
        # frame = cv.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = skeleton.process(rgb_frame)

        skeleton_img = skeleton.draw(frame)

        cv.imshow("Original", frame)
        cv.imshow("Skeleton", skeleton_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()