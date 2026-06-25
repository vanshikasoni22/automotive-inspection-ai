# Automotive Defect Detection AI

YOLOv8-based defect detection microservice for automotive parts inspection.

## Tech Stack
- Python + FastAPI
- YOLOv8 (Ultralytics)
- Trained on industrial defect dataset (1924 images)
- mAP50: 0.991

## Setup
1. Clone the repo
2. Create virtual env: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Add `best.pt` model file
6. Run: `uvicorn main:app --reload --port 8000`

## Endpoints
- GET /health
- POST /analyze