from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import url_to_vector, extract_features

app = FastAPI(title="Phishing Detector API")

# CORS - cho phep frontend goi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/phishing_rf.pkl")

class URLRequest(BaseModel):
    url: str

class DetectionResult(BaseModel):
    url: str
    is_phishing: bool
    confidence: float
    features: dict

@app.get("/")
def root():
    return {"message": "Phishing Detector API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=DetectionResult)
def predict(request: URLRequest):
    features = extract_features(request.url)
    vector = [url_to_vector(request.url)]
    prob = model.predict_proba(vector)[0]
    pred = model.predict(vector)[0]
    confidence = float(max(prob))
    return DetectionResult(
        url=request.url,
        is_phishing=bool(pred == 1),
        confidence=confidence,
        features=features
    )
