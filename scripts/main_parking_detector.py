import cv2
import torch
from ultralytics import YOLO
from parking_slots import parking_slots  # Make sure this exists in same directory
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load video
video_path = 'parking_lot_video.mp4'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Drop delayed frames

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter('output_parking.avi',
                      cv2.VideoWriter_fourcc(*'XVID'), fps,
                      (frame_width, frame_height))

# Frame skipping config
frame_skip = 2
frame_count = 0
prev_detections = []

# FPS tracking
start_time = time.time()
frame_counter = 0

# Check car center inside slot
def is_car_in_slot(car_box, slot):
    cx = (car_box[0] + car_box[2]) / 2
    cy = (car_box[1] + car_box[3]) / 2
    return slot[0] <= cx <= slot[2] and slot[1] <= cy <= slot[3]

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Resize for faster YOLO detection
    resized = cv2.resize(frame, (640, 480))

    cars = []

    if frame_count % frame_skip == 0:
        results = model(resized)[0]

        prev_detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls.item())
                label = model.names[cls_id]

                if label == 'car':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Rescale back to original frame size
                    scale_x = frame_width / 640
                    scale_y = frame_height / 480
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    prev_detections.append([x1, y1, x2, y2])

    frame_count += 1
    cars = prev_detections

    # Draw car boxes
    for car_box in cars:
        x1, y1, x2, y2 = car_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

    # Slot occupancy
    occupied = [False] * len(parking_slots)
    for i, slot in enumerate(parking_slots):
        for car in cars:
            if is_car_in_slot(car, slot):
                occupied[i] = True
                break

    parked = 0
    for i, slot in enumerate(parking_slots):
        color = (0, 0, 255) if occupied[i] else (0, 255, 0)
        parked += int(occupied[i])
        cv2.rectangle(frame, (slot[0], slot[1]), (slot[2], slot[3]), color, 2)
        label = f"Slot {i+1}"
        cv2.putText(frame, label, (slot[0], slot[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    # Overlay parked/empty count
    total_slots = len(parking_slots)
    empty_slots = total_slots - parked
    count_text = f"Parked: {parked} | Empty: {empty_slots}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # FPS overlay
    elapsed_time = time.time() - start_time
    fps_display = frame_counter / elapsed_time
    cv2.putText(frame, f'FPS: {fps_display:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    # Output frame
    out.write(frame)
    cv2.imshow('Parking Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processing complete! Output saved as 'output_parking.avi'")
