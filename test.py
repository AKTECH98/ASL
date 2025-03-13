import os
from dotenv import load_dotenv
load_dotenv()

import cv2 as cv

def main():

    video_file = "Data/downloaded_videos/2.mp4"

    cap = cv.VideoCapture(video_file)

    frames = 1
    fps = cap.get(cv.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv.putText(frame, f"Frame : {frames}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Frame", frame)

        frames += 1

        if cv.waitKey(0) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

    print("Total Frames: ", frames)
    print("FPS: ", fps)

if __name__ == "__main__":
    main()