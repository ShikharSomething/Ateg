import cv2
from ultralytics import YOLO
import os

# Load the trained model
model = YOLO("best.pt")  # Adjust the path to your trained model


def detect_kills(video_path: str, timestamps_file: str):
    # Initialize list to store timestamps for kills
    timestamps = []

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # Process only every 30th frame and skip the rest
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every 30th frame
        if frame_count % 10 == 0:
            # Run inference on the current frame (detection only on every 30th frame)
            results = model(frame)  # Detection only on every 30th frame

            # Check if detections exist and log timestamps for each "kill"
            boxes = results[0].boxes  # Assuming the result is a list of detections

            for box in boxes:
                cls = int(box.cls[0])  # Class of the detected object (you can refine this if needed)
                timestamp = frame_count / fps  # Derive timestamp based on frame count and FPS
                timestamps.append(timestamp)

        frame_count += 1

    cap.release()

    # Save timestamps to a file for later use
    with open(timestamps_file, "w") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")

    print(f"Detected kill timestamps saved to '{timestamps_file}'")
    return timestamps