import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from model.predict import MentalHealthPredictor
from .rag_pipeline import RAGPipeline

app = FastAPI(
    title="MindGuard API",
    description="AI-Powered Mental Health Screening & Empathetic Support",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[MentalHealthPredictor] = None
rag_pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup():
    global predictor, rag_pipeline
    try:
        predictor = MentalHealthPredictor()
    except Exception as e:
        print(f"Warning: Model not loaded ({e}). Using fallback predictions.")
        predictor = None
    rag_pipeline = RAGPipeline()


class ScreeningRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/")
def root():
    return {
        "service": "MindGuard API",
        "status": "active",
        "endpoints": ["/screen", "/chat", "/health"],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "rag_ready": rag_pipeline is not None,
    }


@app.post("/screen")
def screen_text(request: ScreeningRequest):
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 10 characters for accurate screening.",
        )

    if predictor:
        classification = predictor.predict(request.text)
    else:
        classification = _fallback_classification(request.text)

    response = rag_pipeline.generate_response(request.text, classification)

    return {
        "classification": classification,
        "response": response,
        "disclaimer": (
            "This is an AI-powered screening tool and NOT a medical diagnosis. "
            "Please consult a qualified mental health professional for proper evaluation."
        ),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if predictor:
        classification = predictor.predict(request.message)
    else:
        classification = _fallback_classification(request.message)

    response = rag_pipeline.generate_response(request.message, classification)

    return {
        "response": response,
        "detected_state": classification["label"],
        "confidence": classification["confidence"],
    }


def _fallback_classification(text: str) -> dict:
    """Keyword-based fallback when the BERT model isn't loaded."""
    text_lower = text.lower()

    severe_keywords = ["suicide", "kill myself", "end it all", "no reason to live", "self-harm"]
    depression_keywords = ["hopeless", "worthless", "empty", "numb", "can't go on", "no energy"]
    anxiety_keywords = ["panic", "worried", "racing thoughts", "can't breathe", "nervous"]
    stress_keywords = ["overwhelmed", "pressure", "exhausted", "too much", "burned out"]

    if any(kw in text_lower for kw in severe_keywords):
        label, severity = "severe", 1.0
    elif any(kw in text_lower for kw in depression_keywords):
        label, severity = "depression", 0.75
    elif any(kw in text_lower for kw in anxiety_keywords):
        label, severity = "anxiety", 0.5
    elif any(kw in text_lower for kw in stress_keywords):
        label, severity = "stress", 0.25
    else:
        label, severity = "normal", 0.0

    return {
        "label": label,
        "confidence": 0.75,
        "severity_score": severity,
        "probabilities": {
            "normal": 0.2, "depression": 0.2, "anxiety": 0.2,
            "stress": 0.2, "severe": 0.2,
        },
    }
