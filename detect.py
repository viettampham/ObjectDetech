from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("images/image1.jpg", save=True)
print("Done")