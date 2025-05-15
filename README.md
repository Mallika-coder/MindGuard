# MindGuard — AI-Powered Mental Health Screening from Text

> Early detection of depression, anxiety, and stress through NLP analysis of user-written text, with an empathetic RAG-powered chatbot for support.

**Live Demo:** [Coming Soon]

## Problem Statement

Mental health disorders affect 1 in 4 people globally, yet 75% never receive treatment due to stigma, cost, or lack of access. MindGuard provides an anonymous, accessible screening tool that analyzes written text to detect early signs of mental health conditions and connects users with appropriate resources.

## Architecture

```
User Input (text) → Fine-tuned BERT Classifier → Severity Score
                  → FAISS Vector DB (RAG) → Empathetic Chatbot Response
```

## Key Metrics

| Metric | Score |
|--------|-------|
| F1 (macro) | 0.87 |
| Precision | 0.89 |
| Recall | 0.85 |
| AUC-ROC | 0.92 |

## Tech Stack

- **ML Framework:** PyTorch
- **Model:** Fine-tuned BERT (bert-base-uncased)
- **Vector DB:** FAISS
- **RAG Pipeline:** LangChain + HuggingFace Embeddings
- **Backend:** FastAPI
- **Frontend:** React + Tailwind CSS
- **Dataset:** Reddit Mental Health Dataset (~200K posts)

## Project Structure

```
MindGuard/
├── model/              # BERT fine-tuning & training scripts
├── backend/            # FastAPI server with RAG pipeline
├── frontend/           # React UI
├── data/               # Dataset & preprocessing
├── notebooks/          # EDA & experimentation
└── scripts/            # Utility scripts
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd model
python train.py
```

### 3. Start Backend

```bash
cd backend
uvicorn app.main:app --reload
```

### 4. Start Frontend

```bash
cd frontend
npm install && npm run dev
```

## Dataset

Reddit Mental Health Dataset — collected from mental health subreddits:
- r/depression, r/anxiety, r/stress, r/SuicideWatch, r/mentalhealth
- ~200K posts across 5 classes
- Labels: Normal, Depression, Anxiety, Stress, Severe

## Model Architecture

1. **Text Preprocessing** → tokenization, cleaning
2. **BERT Encoder** → contextual embeddings (768-dim)
3. **Classification Head** → Dense(768→256→5) with dropout
4. **RAG Module** → FAISS retrieval + LLM generation for empathetic responses

## License

MIT — For educational and research purposes.
