import cv2
import numpy as np
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('D:/aitech internship/finaldataset/watch2.pt')

# Open the video or webcam
cap = cv2.VideoCapture(0)  # Change '0' to the path of your video file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Kalman filter setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Loop until the watch is detected
detected = False
bbox = None

while not detected and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video source.")
        break

    # Detect objects in the current frame using YOLO
    results = model(frame)

    # Extract bounding box of the watch
    for result in results:
        for detection in result.boxes:
            print(f"Detection: {detection.xyxy[0]}, Confidence: {detection.conf[0]}")
            if detection.conf[0] > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
                bbox = (x1, y1, x2, y2)
                detected = True

                # Initialize Kalman filter state
                kalman.statePre = np.array([x1, y1, 0, 0], dtype=np.float32).reshape(-1, 1)
                kalman.statePost = np.array([x1, y1, 0, 0], dtype=np.float32).reshape(-1, 1)

                # Draw the bounding box for visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                break
        if detected:
            break

    # Show the frame with detection (if any)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Check if the watch was detected
if not detected:
    print("Watch not detected.")
    cap.release()
    cv2.destroyAllWindows()
    exit()
else:
    print(f"Watch detected with bounding box: {bbox}")

# Start tracking the detected object
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video source.")
        break

    # Predict the next position with Kalman filter
    predicted = kalman.predict()
    x_pred, y_pred = int(predicted[0]), int(predicted[1])

    # Update the Kalman filter with the measurement from the detected bounding box
    results = model(frame)
    for result in results:
        for detection in result.boxes:
            if detection.conf[0] > 0.5:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                kalman.correct(np.array([x1, y1], dtype=np.float32).reshape(-1, 1))

                # Update the bbox
                bbox = (x1, y1, x2, y2)
                break

    # Draw the updated bounding box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Optical Flow Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
