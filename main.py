from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import io
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO

app = FastAPI(title="Tooth Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")  # set via env or put best.pt next to main.py
model = None

def get_model():
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(500, f"Model not found at '{MODEL_PATH}'. Set MODEL_PATH env var.")
        model = YOLO(MODEL_PATH)
    return model

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "model_path": MODEL_PATH, "model_loaded": model is not None}

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    # Validate
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")

    # Run inference
    m = get_model()
    results = m(img)
    result = results[0]

    # Annotated image
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Encode both original and annotated as base64
    _, orig_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, ann_buf  = cv2.imencode(".jpg", annotated_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])

    orig_b64 = base64.b64encode(orig_buf).decode()
    ann_b64  = base64.b64encode(ann_buf).decode()

    # Extract detection metadata
    detections = []
    boxes = result.boxes
    masks = result.masks

    names = m.names  # class id -> name

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        xyxy   = box.xyxy[0].tolist()
        detections.append({
            "id": i + 1,
            "class": names.get(cls_id, str(cls_id)),
            "confidence": round(conf * 100, 1),
            "bbox": [round(v) for v in xyxy],
        })

    return JSONResponse({
        "original":   f"data:image/jpeg;base64,{orig_b64}",
        "annotated":  f"data:image/jpeg;base64,{ann_b64}",
        "count":      len(detections),
        "detections": detections,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
    })

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
