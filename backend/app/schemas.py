from pydantic import BaseModel, Field
from typing import Optional


class ScreeningRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to analyze for mental health indicators")


class ClassificationResult(BaseModel):
    label: str
    confidence: float
    severity_score: float
    probabilities: dict[str, float]


class ScreeningResponse(BaseModel):
    classification: ClassificationResult
    response: str
    disclaimer: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    detected_state: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_ready: bool
    vector_store_size: Optional[int] = None
