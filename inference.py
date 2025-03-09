import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load trained ASL classification model
model = tf.keras.models.load_model("asl_hand_landmark_model.h5")

# Load class names (ensure this matches the training classes)
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
               "V", "W", "X", "Y", "Z", "del", "space"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for mirror effect
    # frame = cv2.flip(frame, 0)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    # Extract hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])  # Only (x, y) since z isn't needed

            # Convert landmarks to NumPy array
            landmarks = np.array(landmarks).reshape(1, 21, 2)  # Match model input shape

            # Predict using the trained model
            prediction = model.predict(landmarks)
            predicted_class = class_names[np.argmax(prediction)]  # Get predicted label

            # Display prediction on screen
            cv2.putText(frame, f"Prediction: {predicted_class}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video frame with predictions
    cv2.imshow("ASL Recognition", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
