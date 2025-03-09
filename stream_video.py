import cv2
from skeleton_structure import SkeletonStructure
from Preprocessing.preprocess_url import get_direct_url


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
        cap = cv2.VideoCapture(video_stream_url)

        skeleton_structure = SkeletonStructure()

        if not cap.isOpened():
            print(f"❌ Failed to open video stream: {url}")
            return

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop when the video ends

            skeleton_structure.process(frame)
            skeleton_image = skeleton_structure.draw(frame)

            # ✅ Display the skeleton image
            cv2.imshow("Skeleton", skeleton_image)

            # ✅ Display the processed frame
            # cv2.imshow("Processed Frame", frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"➡ Processed {frame_count} frames from {url}")

            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Processing completed for {url}")

    except Exception as e:
        print(f"❌ Error processing {url}: {e}")


# ✅ Process a single video
process_video()

print("🎉 Done processing video!")
