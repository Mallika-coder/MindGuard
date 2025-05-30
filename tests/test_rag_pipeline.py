"""Tests for the RAG pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.rag_pipeline import RAGPipeline


def test_rag_initialization():
    rag = RAGPipeline()
    assert rag.vector_store is not None
    assert rag.embeddings is not None


def test_retrieve_context():
    rag = RAGPipeline()
    results = rag.retrieve_context("I feel anxious and can't sleep", k=3)
    assert len(results) == 3
    assert all("content" in r for r in results)
    assert all("topic" in r for r in results)


def test_generate_response_depression():
    rag = RAGPipeline()
    classification = {"label": "depression", "severity_score": 0.75, "confidence": 0.9}
    response = rag.generate_response("I feel hopeless", classification)
    assert len(response) > 50
    assert "professional" in response.lower() or "support" in response.lower()


def test_generate_response_severe():
    rag = RAGPipeline()
    classification = {"label": "severe", "severity_score": 1.0, "confidence": 0.95}
    response = rag.generate_response("I want to end it", classification)
    assert "988" in response or "crisis" in response.lower()


def test_generate_response_normal():
    rag = RAGPipeline()
    classification = {"label": "normal", "severity_score": 0.0, "confidence": 0.85}
    response = rag.generate_response("I had a good day today", classification)
    assert len(response) > 20


if __name__ == "__main__":
    test_rag_initialization()
    test_retrieve_context()
    test_generate_response_depression()
    test_generate_response_severe()
    test_generate_response_normal()
    print("All RAG pipeline tests passed!")
