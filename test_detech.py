from ultralytics import YOLO

model = YOLO("runs/detect/train-6/weights/best.pt")

results = model(
    "images/3.jpg",
    save=True,
    conf=0.3
)

count = len(results[0].boxes)

print(f"So thung hang: {count}")