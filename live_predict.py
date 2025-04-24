import cv2 as cv
import torch
import numpy as np
import argparse
from collections import deque
from Modules.skeleton import SkeletonLandmarks
from Models.model_utils import load_model
import threading

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames in sequence")
    parser.add_argument("--hands", action="store_true", help="Use only hand features")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, LABEL_MAP = load_model(args.model_path, device)
    INPUT_SIZE = model.input_size

    frame_buffer = deque(maxlen=args.sequence_length)
    skeleton = SkeletonLandmarks()

    vs = VideoStream(src=args.camera)
    print("🚀 Starting gesture recognition...")

    frame_count = 0
    start_time = cv.getTickCount()
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        skeleton.extract_landmarks(frame)
        features = skeleton.get_features()

        if features is not None:
            hand_features = features[-126:] if args.hands else features
            frame_buffer.append(hand_features)

            canvas = np.zeros_like(frame)
            canvas = skeleton.draw_landmarks(canvas)
            cv.imshow("Landmarks", canvas)


            if len(frame_buffer) == args.sequence_length and frame_count % 5 == 0:
                sequence = np.array(frame_buffer, dtype=np.float32)
                with torch.no_grad():
                    x = torch.tensor(sequence).unsqueeze(0).to(device)
                    lengths = torch.tensor([args.sequence_length])
                    output = model(x, lengths)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    label = LABEL_MAP[pred]
                    confidence = probs[0][pred].item()
                    cv.putText(frame, f'{label} ({confidence:.2f})', (10, 40),
                                cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    print(f"Prediction: {label} ({confidence:.2f})")


                frame_buffer.clear()

        end_time = cv.getTickCount()

        fps = cv.getTickFrequency()/(end_time - start_time)

        cv.putText(frame, f"FPS: {fps:.2f}", (10, 80),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("ASL Gesture Recognition", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        start_time = end_time

    vs.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
