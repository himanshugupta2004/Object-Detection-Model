import cv2
import torch
import pyttsx3
import numpy as np

# Load YOLOv8 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Open webcam (Change 0 to IP Camera URL if using a wireless webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # **Step 1: Perform Object Detection**
    results = model(frame)

    # Extract detected objects and their positions
    detected_objects = results.pandas().xyxy[0]
    
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # **Step 2: Divide the Screen into 3 Regions**
    left_region = frame_width // 3
    right_region = 2 * (frame_width // 3)

    # **Step 3: Analyze Detected Objects**
    obstacle_detected = False
    left_clear = True
    right_clear = True
    closest_distance = float('inf')

    for index, row in detected_objects.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        object_name = row['name']
        
        # Estimate distance (Assumption: Larger objects are closer)
        distance = max(1, int((frame_width / (x_max - x_min)) * 0.5))  # Rough distance estimation
        
        # **Check which region the obstacle is in**
        center_x = (x_min + x_max) // 2
        if center_x < left_region:
            left_clear = False  # Left side has an obstacle
        elif center_x > right_region:
            right_clear = False  # Right side has an obstacle
        else:
            obstacle_detected = True  # Obstacle is in the center

        # Update closest obstacle distance
        closest_distance = min(closest_distance, distance)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{object_name} ({distance}m)", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # **Step 4: Navigation Decision**
    navigation_command = "Move Forward"
    
    if obstacle_detected:
        if left_clear:
            navigation_command = f"Obstacle ahead in {closest_distance} meters, turn left"
        elif right_clear:
            navigation_command = f"Obstacle ahead in {closest_distance} meters, turn right"
        else:
            navigation_command = f"Obstacle ahead in {closest_distance} meters, stop!"

    # **Step 5: Voice Feedback**
    engine.say(navigation_command)
    engine.runAndWait()

    # Show the processed video
    cv2.imshow("Smart Navigation System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
