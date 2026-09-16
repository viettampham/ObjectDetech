from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import shutil
import uuid
import os
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train/weights/best.pt")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.post("/count")
async def count_cartons(
    request: Request,
    file: UploadFile = File(...)
):
    file_name = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)

    count = len(results[0].boxes)

    fontsize = 1.5

    # Vẽ bounding box lên ảnh
    # Đọc ảnh gốc
    img = cv2.imread(file_path)

    # Vẽ box + label nhỏ
    for idx, box in enumerate(results[0].boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        # Bounding box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Label
        label = f"{idx}"
        #label = f"{idx} ({conf:.2f})"

        # Tính kích thước label
        (w, h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,  # font size
            1
        )

        # Background label
        cv2.rectangle(
            img,
            (x1, y1 - h - 6),
            (x1 + w + 6, y1),
            (255, 0, 0),
            -1
        )

        # Text
        cv2.putText(
            img,
            label,
            (x1 + 3, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontsize,        # chỉnh kích thước chữ ở đây
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    annotated_img = img


    result_name = f"result_{file_name}"
    result_path = os.path.join(RESULT_DIR, result_name)

    cv2.imwrite(result_path, annotated_img)

    return {
        "count": count,
        "image_url": f"{request.base_url}results/{result_name}"
    }