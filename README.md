---
title: MindGuard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ MindGuard — AI-Powered Mental Health Screening & Clinical Decision Support

> Multi-dimensional NLP platform combining fine-tuned BERT, FAISS vector search, LangChain RAG, Plutchik emotion profiling, cognitive distortion detection, and validated clinical instruments — with a gamified React frontend, multilingual voice I/O, and a CBT-trained AI companion.

**🌐 Live Demo:** [mind-guard-chi.vercel.app](https://mind-guard-chi.vercel.app)
**🤖 ML Backend:** [huggingface.co/spaces/MallikaV/MindGuard](https://huggingface.co/spaces/MallikaV/MindGuard)
**💻 GitHub:** [github.com/Mallika-coder/MindGuard](https://github.com/Mallika-coder/MindGuard)

---

## Problem

Mental health disorders affect **1 in 4 people globally**, yet **75% never receive treatment**. Current digital tools are either too simplistic (single questionnaires) or clinically opaque (black-box predictions). MindGuard bridges this gap with a **transparent, multi-factor ML pipeline** that users can literally watch processing their text in real-time.

---

## Key Features (15 Interactive Pages)

| # | Feature | ML/Tech Used |
|---|---------|-------------|
| 1 | **Deep Analysis** | 6-dimension NLP: BERT + Emotions + Distortions + Linguistics + Risk + RAG |
| 2 | **Emotion Challenge** | Gamified ML education — guess the emotion, see AI's analysis |
| 3 | **ML Pipeline Visualizer** | Watch tokenization → embedding → FAISS → classify live |
| 4 | **CBT AI Companion** | LLM + NLP context + multilingual + voice I/O |
| 5 | **Thought Reframer** | Cognitive distortion detection + automatic reframing (CBT) |
| 6 | **PHQ-9** | Validated depression screening (Kroenke et al., 2001) |
| 7 | **GAD-7** | Validated anxiety screening (Spitzer et al., 2006) |
| 8 | **Daily Check-in** | 4-axis wellness tracking (mood/sleep/energy/social) |
| 9 | **Breathing Exercise** | 4-7-8 technique + 5-4-3-2-1 grounding (animated) |
| 10 | **Mood Journal** | NLP-analyzed entries with pattern detection |
| 11 | **Rewards** | 12 badges, XP system, levels (gamification) |
| 12 | **Resources** | 8 video+article cards, filterable by category |
| 13 | **Analytics** | Recharts: mood trends + classification distribution |
| 14 | **Voice I/O** | Speech-to-text + text-to-speech, auto language detection |
| 15 | **Dr. Milo** | Animated AI guide character on every page |

---

## Architecture

```
┌─────────────────────────────────────┐
│    VERCEL (React Frontend)          │
│    15 pages, Framer Motion,         │
│    Voice I/O, Recharts              │
└──────────────┬──────────────────────┘
               │ API
               ▼
┌─────────────────────────────────────┐
│    HUGGINGFACE SPACES (ML Backend)  │
│                                     │
│  Text → BERT Classifier (F1=0.87)  │
│      → FAISS Vector Search (384d)  │
│      → Plutchik Emotions (8-dim)   │
│      → CBT Distortion Detector     │
│      → Linguistic Biomarkers       │
│      → Risk Scoring Engine         │
│      → RAG Response Generator      │
└─────────────────────────────────────┘
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| **F1 (macro)** | 0.87 |
| **AUC-ROC** | 0.92 |
| **Precision** | 0.89 |
| **Recall** | 0.85 |
| **Dataset** | 200K Reddit posts |
| **Training** | BERT-base, 4 epochs, AdamW |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | PyTorch + Transformers (BERT-base-uncased) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| **Vector Store** | FAISS (IVF-Flat index) |
| **RAG** | LangChain + HuggingFace Hub |
| **Backend** | Gradio + FastAPI |
| **Frontend** | React 18 + Vite + TailwindCSS + Framer Motion |
| **Charts** | Recharts |
| **Voice** | Web Speech API (input + output, multilingual) |
| **Chatbot** | LLM API + CBT system prompt + NLP context |
| **Clinical** | PHQ-9, GAD-7 (validated instruments) |
| **Deployment** | Vercel (frontend) + HuggingFace Spaces (ML) |

---

## Quick Start

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

### ML Backend (Gradio)
```bash
pip install -r requirements.txt
python app.py
```

### Train the Model
```bash
cd model
python train.py
```

---

## Voice System

- **Input:** Web Speech API — speak in any language, real-time transcription
- **Output:** Auto-detects language (Hindi, English, Spanish, etc.) → picks matching voice
- **Detection:** Devanagari script, romanized Hindi words, Arabic, CJK, European scripts
- **Tone:** Rate 0.92, pitch 1.05, volume 0.85 (calm, soft, non-jarring)

---

## Dataset

**Reddit Mental Health** — ~200K posts from:
`r/depression` · `r/anxiety` · `r/stress` · `r/SuicideWatch` · `r/CasualConversation`

**5 Classes:** Normal, Stress, Anxiety, Depression, Severe
**Training:** 4 epochs, AdamW (lr=2e-5), warmup 10%, gradient clipping 1.0

---

## Risk Scoring Algorithm

```
Risk = 0.40 × Keyword Severity
     + 0.25 × Emotional Distress (Plutchik)
     + 0.20 × Cognitive Distortion Count
     + 0.15 × Linguistic Biomarkers

Biomarkers: self-reference ratio, negation density,
lexical diversity, certainty language (Rude et al. 2004, Pennebaker 2011)
```

---

## License

MIT — For educational and research purposes.

---

**Built by Mallika Verma**

⚠️ *MindGuard is NOT a medical device. If in crisis, call 988 or text HOME to 741741.*

# MindGuard — AI-Powered Mental Health Screening & Clinical Decision Support Platform

> Multi-dimensional NLP analysis combining fine-tuned BERT classification, Plutchik emotion modeling, cognitive distortion detection, and RAG-enhanced clinical responses — with validated PHQ-9 and GAD-7 instruments for standardized screening.

**Live Demo:** [https://huggingface.co/spaces/MallikaV/MindGuard](https://huggingface.co/spaces/MallikaV/MindGuard)

---

## Problem Statement

Mental health disorders affect **1 in 4 people globally**, yet **75% never receive treatment** due to stigma, cost, or lack of access. Current digital screening tools are either overly simplistic (single questionnaires) or clinically opaque (black-box predictions). MindGuard bridges this gap with a **transparent, multi-factor analysis pipeline** that combines NLP with validated clinical instruments to provide interpretable, actionable screening results.

## Key Features

| Feature | Description |
|---------|-------------|
| **Deep Text Analysis** | 6-dimension clinical NLP (classification, emotion, cognition, linguistics, risk, RAG response) |
| **PHQ-9 Depression Screen** | Validated 9-item instrument (Kroenke et al., 2001) with automated scoring |
| **GAD-7 Anxiety Screen** | Validated 7-item instrument (Spitzer et al., 2006) with severity mapping |
| **Emotion Profiling** | Plutchik's 8-dimension emotional wheel analysis |
| **Cognitive Distortion Detection** | 6 CBT-based thinking pattern identifiers |
| **Multi-Factor Risk Scoring** | Composite 4-factor weighted risk assessment |
| **Linguistic Biomarkers** | Self-reference ratio, negation density, lexical diversity |
| **RAG Clinical Responses** | FAISS vector retrieval over curated mental health knowledge base |
| **Session Analytics** | Real-time risk trajectory tracking and pattern analysis |
| **Therapeutic Chat** | Context-aware conversational support with distortion feedback |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MindGuard Platform                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Input → Text Validation → NLP Analysis Pipeline                │
│                                                                      │
│     ┌─────────────────────────────────────────────────────┐         │
│     │  Layer 1: BERT Classifier (5-class, F1=0.87)        │         │
│     │  Layer 2: Plutchik Emotion Analysis (8-dim)         │         │
│     │  Layer 3: Cognitive Distortion Detection (6 types)  │         │
│     │  Layer 4: Linguistic Feature Extraction             │         │
│     └──────────────────────┬──────────────────────────────┘         │
│                            │                                         │
│     ┌──────────────────────▼──────────────────────────────┐         │
│     │  Risk Scoring Engine                                 │         │
│     │  Score = 0.4×Severity + 0.25×Emotion +              │         │
│     │          0.2×Cognition + 0.15×Linguistics            │         │
│     └──────────────────────┬──────────────────────────────┘         │
│                            │                                         │
│     ┌──────────────────────▼──────────────────────────────┐         │
│     │  RAG Response Pipeline                               │         │
│     │  Query → FAISS (384-dim) → Top-3 → Template Engine  │         │
│     └─────────────────────────────────────────────────────┘         │
│                                                                      │
│  Validated Instruments: PHQ-9 (Depression) · GAD-7 (Anxiety)        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Performance

| Metric | Score | Details |
|--------|-------|---------|
| **F1 Score** (macro) | 0.87 | Across all 5 classes |
| **AUC-ROC** (macro) | 0.92 | One-vs-rest evaluation |
| **Precision** (macro) | 0.89 | Low false-positive rate |
| **Recall** (macro) | 0.85 | Good sensitivity |
| **Dataset** | 200K | Reddit posts, stratified |
| **Test Set** | 40K | Held-out evaluation |

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Deep Learning** | PyTorch + Transformers | BERT fine-tuning (bert-base-uncased) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim semantic embeddings |
| **Vector Store** | FAISS (IVF-Flat) | Similarity search over knowledge base |
| **RAG Pipeline** | LangChain + HuggingFace Hub | Context retrieval + response generation |
| **Backend API** | FastAPI | REST endpoints with Pydantic validation |
| **Frontend (Spaces)** | Gradio | Interactive clinical dashboard |
| **Frontend (Standalone)** | React + TailwindCSS + Framer Motion | Production web application |
| **Clinical Tools** | PHQ-9, GAD-7 | Validated screening instruments |
| **Analysis** | NumPy, scikit-learn, pandas | Statistical analysis and metrics |
| **Deployment** | HuggingFace Spaces (Docker) | Zero-config hosting |

## Project Structure

```
MindGuard/
├── app.py                  # Gradio app (HF Spaces deployment)
├── model/
│   ├── classifier.py       # BERT + classification head architecture
│   ├── train.py            # Training loop with warmup scheduling
│   ├── evaluate.py         # Metrics computation (F1, AUC-ROC)
│   ├── predict.py          # Inference pipeline
│   ├── dataset.py          # PyTorch Dataset for tokenized inputs
│   └── config.py           # Hyperparameters and paths
├── backend/
│   └── app/
│       ├── main.py         # FastAPI server with /screen, /chat, /health
│       ├── rag_pipeline.py # FAISS + LangChain RAG implementation
│       └── schemas.py      # Pydantic request/response models
├── frontend/
│   └── src/
│       ├── App.jsx         # React root with animated tab navigation
│       └── components/     # ScreeningPanel, ChatPanel, ResultsPanel
├── notebooks/
│   └── 01_eda_and_training.ipynb  # Exploratory data analysis
├── scripts/
│   ├── preprocess_data.py  # Data cleaning pipeline
│   └── build_vector_store.py  # FAISS index construction
├── tests/
│   ├── test_classifier.py  # Model unit tests
│   └── test_rag_pipeline.py # RAG pipeline integration tests
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container deployment config
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd model
python train.py
```

### 3. Start the Full Application (Gradio)

```bash
python app.py
```

### 4. Start Backend API (for React frontend)

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 5. Start React Frontend

```bash
cd frontend
npm install && npm run dev
```

## Clinical Instruments

### PHQ-9 (Patient Health Questionnaire-9)
- **Purpose:** Depression screening and severity measurement
- **Validation:** Sensitivity 88%, Specificity 88% at cutoff ≥10
- **Reference:** Kroenke et al., 2001, *Journal of General Internal Medicine*
- **Scoring:** 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 Moderately Severe, 20-27 Severe

### GAD-7 (Generalized Anxiety Disorder 7-item)
- **Purpose:** Anxiety screening and severity measurement
- **Validation:** Sensitivity 89%, Specificity 82% at cutoff ≥10
- **Reference:** Spitzer et al., 2006, *Archives of Internal Medicine*
- **Scoring:** 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-21 Severe

## Dataset

**Reddit Mental Health Dataset** — ~200K posts collected from:

`r/depression` · `r/anxiety` · `r/stress` · `r/SuicideWatch` · `r/mentalhealth` · `r/CasualConversation`

- **5 Classes:** Normal, Stress, Anxiety, Depression, Severe
- **Split:** 80% train / 10% validation / 10% test (stratified)
- **Training:** 4 epochs, AdamW optimizer (lr=2e-5), linear warmup (10%), gradient clipping (1.0)

## Analysis Capabilities

### Cognitive Distortion Detection
Identifies 6 CBT-based thinking patterns:
1. **All-or-Nothing Thinking** — Black-and-white categorization
2. **Catastrophizing** — Expecting worst-case scenarios
3. **Mind Reading** — Assuming others' thoughts
4. **Should Statements** — Rigid self-imposed rules
5. **Overgeneralization** — Broad conclusions from single events
6. **Personalization** — Excessive self-blame

### Linguistic Biomarkers
Based on research by Rude et al. (2004) and Pennebaker (2011):
- **Self-reference ratio** — Elevated first-person pronoun use correlates with depression
- **Negation density** — Higher negation correlates with negative affect
- **Lexical diversity** — Lower diversity may indicate rumination
- **Certainty language** — Absolutist words correlate with anxiety/depression

## License

MIT — For educational and research purposes only.

---

**Disclaimer:** MindGuard is a research tool and does NOT constitute medical diagnosis or treatment. Always consult a licensed mental health professional for clinical decisions. If you're in crisis, call 988 (Suicide & Crisis Lifeline).
