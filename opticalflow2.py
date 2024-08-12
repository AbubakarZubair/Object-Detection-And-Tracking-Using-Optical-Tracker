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

# Initialize the tracking points for Optical Flow
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Define the region of interest (ROI) for the detected object
roi = old_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]

# Make sure ROI is valid
if roi.size == 0:
    print("Error: Invalid ROI.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Detect good features to track within the ROI
p0 = cv2.goodFeaturesToTrack(roi, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Ensure p0 is not None and reshape it correctly
if p0 is not None:
    p0 = np.float32(p0).reshape(-1, 1, 2)
    # Adjust the points to the original frame coordinates
    p0[:, :, 0] += bbox[0]
    p0[:, :, 1] += bbox[1]
else:
    print("Error: No good features to track found.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

mask = np.zeros_like(frame)

# Start tracking the detected object
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video source.")
        break

    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, p0, None, **lk_params)

    # Check if Optical Flow calculation succeeded
    if p1 is not None:
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        if len(good_new) > 0:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = map(int, new.ravel())  # Ensure coordinates are integers
                c, d = map(int, old.ravel())  # Ensure coordinates are integers
                if (0 <= a < frame.shape[1]) and (0 <= b < frame.shape[0]):
                    mask = cv2.line(mask, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                    frame = cv2.circle(frame, (a, b), 5, color=(0, 255, 0), thickness=-1)

            img = cv2.add(frame, mask)

            # Update the previous frame and previous points
            old_gray = gray_frame.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            img = frame

    else:

        img = frame

    # Show the image
    cv2.imshow('Optical Flow Tracking', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
