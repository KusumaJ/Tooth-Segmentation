# DentSeg — Tooth Segmentation Web App

A clean, production-ready web interface for your YOLO tooth segmentation model.

## Structure

```
tooth-seg/
├── main.py            # FastAPI backend
├── requirements.txt
├── best.pt            # ← Put your YOLO weights here (or set MODEL_PATH)
└── static/
    └── index.html     # Full frontend (self-contained)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your model weights next to main.py
cp /path/to/your/weights.pt ./best.pt

# 3. Run
python main.py
# → Open http://localhost:8000
```

## Custom Model Path

Set the `MODEL_PATH` environment variable to point to any `.pt` file:

```bash
MODEL_PATH=/models/tooth_100ep.pt python main.py
```

## API Endpoints

| Method | Path       | Description                              |
|--------|------------|------------------------------------------|
| GET    | `/`        | Serves the web UI                        |
| GET    | `/health`  | Model status check                       |
| POST   | `/segment` | Upload image → returns original + annotated base64 + detection data |

### POST `/segment` — Response

```json
{
  "original":   "data:image/jpeg;base64,...",
  "annotated":  "data:image/jpeg;base64,...",
  "count":      32,
  "detections": [
    {
      "id": 1,
      "class": "tooth",
      "confidence": 94.2,
      "bbox": [120, 80, 210, 190]
    }
  ],
  "image_size": { "width": 640, "height": 480 }
}
```

## Deploy to a Server

```bash
# With gunicorn (production)
pip install gunicorn
gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With Docker (add a Dockerfile)
# FROM python:3.11-slim
# COPY . /app && WORKDIR /app
# RUN pip install -r requirements.txt
# CMD ["python", "main.py"]
```

## Dev Mode (frontend only)

If you want to test the UI without the backend, open `static/index.html` directly —
it will show you the layout. Point `API` variable in the JS to your running backend URL:

```js
const API = 'http://localhost:8000';  // change from '' to this for cross-origin dev
```
