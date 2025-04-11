from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')  # downloads the model if not present

# Run on a sample parking lot video
results = model('parking_lot_video.mp4', save=True, conf=0.5)

# Output video with detections is saved in 'runs/detect/predict'
print("✅ Detection complete! Check the 'runs/detect/predict' folder.")
