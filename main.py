from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import uvicorn

app = FastAPI(title="Automotive Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model — we'll replace this path after training
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = YOLO("best.pt")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model not found yet: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    if model is None:
        return {
            "error": "Model not loaded",
            "recommendation": "manual_review",
            "defects": [],
            "confidence_score": 0,
            "severity": "unknown"
        }

    # Read image
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run inference
    results = model(pil_image)
    detections = results[0]

    defects = []
    max_confidence = 0

    for box in detections.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        defects.append({
            "type": class_name,
            "confidence": round(confidence, 3),
            "bbox": box.xyxy[0].tolist()
        })

        if confidence > max_confidence:
            max_confidence = confidence

    # Determine severity
    if max_confidence >= 0.75:
        severity = "high"
        recommendation = "reject"
    elif max_confidence >= 0.45:
        severity = "medium"
        recommendation = "manual_review"
    elif max_confidence > 0:
        severity = "low"
        recommendation = "accept"
    else:
        severity = "none"
        recommendation = "accept"

    return {
        "defects": defects,
        "confidence_score": round(max_confidence, 3),
        "severity": severity,
        "recommendation": recommendation,
        "total_defects_found": len(defects)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)