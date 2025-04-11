from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can change this to yolov8s.pt, yolov8m.pt etc.

# Export the model to ONNX format
model.export(format='onnx')

print("✅ Model exported as yolov8n.onnx")
