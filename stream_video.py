import cv2 as cv
# from skeleton_structure import SkeletonStructure
from Preprocessing.preprocess_url import get_direct_url
import mediapipe as mp

# ✅ Function to process YouTube video without downloading
def process_video(url="https://www.youtube.com/watch?v=C37R_Ix8-qs"):
    try:
        print(f"🔄 Processing video: {url}")

        # Get the direct video URL
        video_stream_url = get_direct_url(url)
        if not video_stream_url:
            print(f"❌ Skipping {url} - Could not retrieve stream URL.")
            return

        # Open the video stream with OpenCV
        cap = cv.VideoCapture(video_stream_url)

        # skeleton_structure = SkeletonStructure()

        if not cap.isOpened():
            print(f"❌ Failed to open video stream: {url}")
            return

        frame_count = 0
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.6,
                                                  min_tracking_confidence=0.5)

        mp_drawing = mp.solutions.drawing_utils

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop when the video ends

            # skeleton_structure.process(frame)
            # skeleton_image = skeleton_structure.draw(frame)

            # ✅ Display the skeleton image
            # cv.imshow("Skeleton", skeleton_image)

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # ✅ Draw the landmarks
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)


            # ✅ Display the processed frame
            cv.putText(frame, f"Frame: {frame_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow("Processed Frame", frame)

            frame_count += 1


            # Press 'q' to exit early
            if cv.waitKey(0) & 0xFF == ord("q"):
                break

        print(f"🎉 Processed {frame_count} frames from {url}")
        cap.release()
        cv.destroyAllWindows()
        print(f"✅ Processing completed for {url}")

    except Exception as e:
        print(f"❌ Error processing {url}: {e}")


# ✅ Process a single video
process_video()

print("🎉 Done processing video!")
