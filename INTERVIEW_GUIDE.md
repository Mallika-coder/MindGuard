# MindGuard — Complete Interview Preparation Guide

## AI-Powered Mental Health Screening & Clinical Decision Support Platform

**Live Demo (Frontend):** https://mind-guard-chi.vercel.app
**ML Backend:** https://huggingface.co/spaces/MallikaV/MindGuard
**GitHub:** https://github.com/Mallika-coder/MindGuard
**Tech Stack:** React 18, Vite, TailwindCSS, Framer Motion, Recharts, Web Speech API, PyTorch, BERT, FAISS, LangChain, Gradio
**Total Lines of Code:** ~4,500+ (frontend) + ~1,800+ (ML backend)

---

## TABLE OF CONTENTS

1. Project Overview & Elevator Pitch
2. How MindGuard is Different from MindLink & CureCue
3. Architecture & System Design
4. ML Pipeline Deep Dive
5. Frontend Architecture (React App)
6. Backend Architecture (Gradio + FastAPI)
7. Feature Deep Dives (15 Features)
8. Voice System (Speech-to-Text + Text-to-Speech)
9. CBT Integration & Chatbot Intelligence
10. Gamification System
11. Clinical Instruments (PHQ-9, GAD-7)
12. Performance & Deployment
13. Interview Questions & Answers (50+)
14. How to Present This in an Interview

---

## 1. PROJECT OVERVIEW & ELEVATOR PITCH

### 30-Second Pitch:

"MindGuard is an AI-powered mental health screening platform that combines a fine-tuned BERT classifier with a FAISS vector database and LangChain RAG pipeline to provide multi-dimensional mental health analysis. It screens text across 6 NLP dimensions — classification, Plutchik's emotion profiling, cognitive distortion detection, linguistic biomarkers, multi-factor risk scoring, and evidence-based therapeutic responses. It includes validated clinical instruments (PHQ-9, GAD-7), a CBT-trained chatbot with real-time emotion detection, voice input/output in any language, an ML pipeline visualizer, and a gamified emotion challenge. The frontend is React with Framer Motion, the ML backend runs on HuggingFace Spaces. I trained the BERT model on 200K Reddit mental health posts achieving 0.87 F1 score."

### 15-Second Version:

"MindGuard is a mental health AI platform with BERT classification, FAISS vector search, RAG pipeline, CBT chatbot, voice I/O in any language, and clinical screening instruments — all with a gamified React frontend and ML backend on HuggingFace."

### Why This Project Wins:

| Aspect | What It Proves |
|--------|---------------|
| BERT Fine-tuning | Can train and deploy custom deep learning models |
| FAISS + RAG | Understands vector databases and retrieval-augmented generation |
| Multi-factor Risk Scoring | Can design weighted ensemble algorithms |
| Linguistic Biomarkers | Knows research literature (Pennebaker, Rude et al.) |
| Clinical Instruments | Domain research + proper validation methodology |
| CBT Chatbot with Memory | Can build context-aware conversational AI |
| Voice I/O (multilingual) | Browser API mastery + language detection |
| ML Pipeline Visualizer | Can explain and visualize ML concepts |
| React + Framer Motion | Production-grade frontend skills |
| HuggingFace Deployment | MLOps — deploying models to production |

---

## 2. HOW MINDGUARD IS DIFFERENT FROM MINDLINK & CURECUE

| Dimension | CureCue | MindLink | MindGuard |
|-----------|---------|----------|-----------|
| **Proves** | Full-stack (backend + auth + DB) | Complex frontend (Web Audio, Recharts) | **ML/AI Engineering** |
| **Architecture** | Next.js + MongoDB + JWT | React + localStorage | React + BERT + FAISS + RAG |
| **AI Usage** | Generic chat widget | CBT chat (add-on) | **AI IS the core product** |
| **Database** | MongoDB Atlas | localStorage | FAISS vector store |
| **Model** | None | None | Fine-tuned BERT (200K samples) |
| **Unique Tech** | httpOnly cookies, middleware | Web Audio oscillators | NLP pipelines, embeddings, RAG |
| **Auth** | Custom JWT + bcrypt | None | None (ML-focused) |
| **Design** | Dark/mystical alchemy | Student wellness | Clinical health-tech + gamified |
| **Voice** | None | None | Full voice I/O, multilingual |

### How to Explain in Interview:

"I have three projects that prove different engineering skills. CureCue demonstrates full-stack backend — JWT auth, MongoDB, API security, rate limiting. MindLink proves complex frontend — Web Audio API, data visualization, localStorage persistence. MindGuard proves **ML engineering** — I trained a BERT model on 200K samples, built a FAISS vector database for semantic search, implemented a RAG pipeline with LangChain, designed a multi-factor risk scoring algorithm, and deployed the ML model to production on HuggingFace Spaces. Together they show I can handle any part of the stack, but MindGuard specifically shows deep AI/ML capability."

---

## 3. ARCHITECTURE & SYSTEM DESIGN

### High-Level Architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VERCEL (React Frontend)                            │
│                    mind-guard-chi.vercel.app                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  React 18 SPA                                                        │
│  ├── 15 Page Components (Analysis, Chat, Pipeline, etc.)            │
│  ├── Framer Motion (animations, page transitions)                   │
│  ├── Recharts (data visualization)                                  │
│  ├── Web Speech API (voice input/output, multilingual)              │
│  ├── Client-side NLP (keyword classification, emotions)             │
│  └── Axios → API calls to ML backend                                │
│                                                                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ API calls
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              HUGGINGFACE SPACES (ML Backend)                          │
│              huggingface.co/spaces/MallikaV/MindGuard                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Gradio App (app.py)                                                 │
│  ├── BERT Classifier (5-class, F1=0.87)                             │
│  ├── Sentence-Transformers (all-MiniLM-L6-v2, 384-dim)             │
│  ├── FAISS Vector Store (8 documents, IVF-Flat)                     │
│  ├── LangChain RAG Pipeline (retrieve → generate)                   │
│  ├── Plutchik Emotion Analyzer (8 dimensions)                       │
│  ├── Cognitive Distortion Detector (6 CBT patterns)                 │
│  ├── Linguistic Feature Extractor (9 biomarkers)                    │
│  └── Multi-factor Risk Scoring Engine (4-factor weighted)           │
│                                                                      │
│  Model Training (model/train.py)                                     │
│  ├── Dataset: Reddit Mental Health (~200K posts)                    │
│  ├── Architecture: BERT-base → Dense(768→256→5)                     │
│  ├── Training: AdamW, 4 epochs, linear warmup                      │
│  └── Evaluation: macro-F1=0.87, AUC-ROC=0.92                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Analysis):

```
User types/speaks text
       │
       ▼
[Client-side] Quick classification (keyword-based, instant feedback)
       │
       ▼ (async)
[API call to HuggingFace Spaces]
       │
       ├── BERT Tokenizer → WordPiece tokens → IDs
       ├── Sentence-Transformer → 384-dim embedding
       ├── FAISS → cosine similarity → top-3 contexts
       ├── BERT Classifier → 5-class softmax probabilities
       ├── Plutchik Analyzer → 8-dimension emotion vector
       ├── Distortion Detector → matched CBT patterns
       ├── Linguistic Extractor → biomarker features
       └── Risk Engine → composite weighted score
       │
       ▼
Results rendered with animations + Dr. Milo speaks summary
```

---

## 4. ML PIPELINE DEEP DIVE

### Model Architecture:

```
Input Text
    │
    ▼
BERT Tokenizer (WordPiece, vocab=30522, max_len=256)
    │
    ▼
BERT-base-uncased (12 layers, 768 hidden, 12 attention heads, 110M params)
    │
    ▼
[CLS] token output (768-dim)
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(768 → 256) + ReLU
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(256 → 5) + Softmax
    │
    ▼
Output: [Normal, Stress, Anxiety, Depression, Severe]
```

### Training Details:

| Parameter | Value |
|-----------|-------|
| Dataset | Reddit Mental Health (~200K posts) |
| Sources | r/depression, r/anxiety, r/stress, r/SuicideWatch, r/CasualConversation |
| Classes | 5 (Normal, Stress, Anxiety, Depression, Severe) |
| Split | 80% train / 10% val / 10% test (stratified) |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Scheduler | Linear warmup (10% of total steps) |
| Epochs | 4 |
| Batch Size | 16 |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | On macro-F1 (validation set) |

### Performance:

| Metric | Score |
|--------|-------|
| F1 (macro) | 0.87 |
| AUC-ROC (macro, OVR) | 0.92 |
| Precision (macro) | 0.89 |
| Recall (macro) | 0.85 |
| Test Samples | 40,000 |

### RAG Pipeline:

```
User Query
    │
    ▼
all-MiniLM-L6-v2 Encoder → 384-dim embedding
    │
    ▼
FAISS Index (IVF-Flat, 8 documents chunked at 500 tokens)
    │
    ▼
Cosine Similarity → Top-3 most relevant chunks
    │
    ▼
Template Engine (conditioned on severity level + retrieved contexts)
    │
    ▼
Personalized therapeutic response
```

### Knowledge Base Documents:
1. Depression Support (WHO statistics, CBT approaches, when to seek help)
2. Anxiety Management (301M affected, breathing techniques, exposure therapy)
3. Stress Reduction (fight-or-flight, Pomodoro, boundary setting)
4. Crisis Resources (988 Lifeline, Crisis Text Line, IASP)
5. Sleep Hygiene (CBT-I, temperature, screen time)
6. Mindfulness Practices (body scan, loving-kindness, 8-week brain changes)
7. Professional Help (psychiatrists vs psychologists, telehealth, EAPs)
8. Self-Care Strategies (physical, emotional, social, spiritual pillars)

### Multi-Factor Risk Scoring:

```
Composite Risk = 0.40 × Keyword Severity
               + 0.25 × Emotional Distress Index
               + 0.20 × Cognitive Distortion Count
               + 0.15 × Linguistic Marker Score

Where:
- Keyword Severity = severity_map[predicted_label] (0 to 1)
- Emotional Distress = sadness×0.3 + fear×0.25 + anger×0.15 - joy×0.2
- Cognitive Distortion = min(distortion_count × 0.12, 0.35)
- Linguistic Markers = negation_ratio×0.4 + first_person×0.2 + (1-diversity)×0.15 + certainty×0.25
```

### Linguistic Biomarkers (Research-Backed):

| Feature | Research Basis |
|---------|---------------|
| Self-reference ratio | High "I/me/my" correlates with depression (Rude et al., 2004) |
| Negation density | More "not/never/can't" = negative affect |
| Lexical diversity | Low diversity may indicate rumination (Pennebaker, 2011) |
| Certainty language | Absolutist words correlate with anxiety/depression |
| Temporal focus | Past-orientation linked to depression, future-anxiety |

---

## 5. FRONTEND ARCHITECTURE (React)

### File Structure:

```
frontend/
├── public/
│   └── milo-character.avif        # Dr. Milo cartoon character
├── src/
│   ├── App.jsx                     # Root with sidebar + page routing
│   ├── main.jsx                    # React entry point
│   ├── index.css                   # Tailwind + global styles
│   ├── components/
│   │   ├── Sidebar.jsx             # Navigation (15 items + crisis line)
│   │   ├── MiloGuide.jsx           # AI guide with character image
│   │   └── VoiceButton.jsx         # Mic button + speak button
│   ├── hooks/
│   │   ├── useLocalState.js        # localStorage persistence
│   │   └── useVoice.js             # Speech-to-Text + TTS + lang detect
│   ├── pages/
│   │   ├── Landing.jsx             # Hero with Milo, CTA, features
│   │   ├── Analysis.jsx            # 6-dimension NLP dashboard
│   │   ├── EmotionChallenge.jsx    # Game: guess emotion vs AI
│   │   ├── Pipeline.jsx            # ML pipeline visualizer
│   │   ├── Chat.jsx                # CBT chatbot (LLM + voice)
│   │   ├── ThoughtReframer.jsx     # CBT distortion + reframe
│   │   ├── PHQ9.jsx                # Depression questionnaire
│   │   ├── GAD7.jsx                # Anxiety questionnaire
│   │   ├── DailyCheckin.jsx        # 4-axis wellness tracker
│   │   ├── Breathing.jsx           # 4-7-8 + grounding
│   │   ├── Journal.jsx             # NLP-analyzed mood journal
│   │   ├── Rewards.jsx             # XP, badges, levels
│   │   ├── Resources.jsx           # Videos + articles
│   │   ├── Analytics.jsx           # Recharts mood/classification
│   │   └── HowItWorks.jsx         # Architecture diagram
│   └── utils/
│       ├── wellness.js             # NLP functions (classify, emotions, etc.)
│       └── api.js                  # HuggingFace API client
├── package.json
├── tailwind.config.js
├── vite.config.js
└── postcss.config.js
```

### Design System:

| Token | Value |
|-------|-------|
| Primary (Healing Green) | #1a3d2e → #4a9d6e → #e8f5ee |
| Accent (Calm Lavender) | #4a3d6e → #7c6aad → #f3f0fa |
| Surface | #f4f7fa |
| Card | white, 1px #e8ecf0 border |
| Card Radius | 16px |
| Button Radius | 24px (pill) |
| Font (body) | Inter |
| Font (data) | JetBrains Mono |
| Animations | Framer Motion (fade+slide, spring physics) |

---

## 6. VOICE SYSTEM

### Speech-to-Text (Input):
- Uses Web Speech API (webkitSpeechRecognition)
- Auto-detects language (no hardcoded lang)
- Real-time interim results fill input field
- Visual: pulsing red ring + waveform bars when listening

### Text-to-Speech (Output):
- Language auto-detection from response text
- Detects: Hindi (Devanagari + romanized), Arabic, Chinese, Japanese, Korean, Spanish, French
- Picks matching female voice for detected language
- Tone: rate 0.92, pitch 1.05, volume 0.85 (calm, soft)
- Dr. Milo narrates analysis results and chat responses

### Language Detection Algorithm:

```javascript
function detectLanguage(text) {
  if (/[ऀ-ॿ]/.test(text)) return 'hi-IN'           // Hindi script
  if (/\b(hai|mein|mujhe|kya|nahi|bahut)\b/i.test(text)) return 'hi-IN'  // Romanized Hindi
  if (/[؀-ۿ]/.test(text)) return 'ar-SA'           // Arabic
  if (/[一-鿿]/.test(text)) return 'zh-CN'          // Chinese
  if (/[가-힯]/.test(text)) return 'ko-KR'          // Korean
  if (/[áéíóúñ¿¡]/i.test(text)) return 'es-ES'     // Spanish
  return 'en-US'
}
```

---

## 7. CBT CHATBOT INTELLIGENCE

### How It Works:

```
User Message
    │
    ├── Client-side NLP: classify, emotions, distortions
    │
    ▼
Build context object:
{
  label: "anxiety",
  confidence: 0.67,
  topEmotions: "fear, sadness",
  distortions: "Catastrophizing, All-or-Nothing",
  riskLevel: "Moderate"
}
    │
    ▼
LLM API call with system prompt + context
    │
    ▼
Response (in user's language) + CBT technique
    │
    ▼
Dr. Milo speaks response aloud
```

### System Prompt Enforces:
1. Validate feelings FIRST
2. Identify cognitive distortions
3. Guide Socratic reframing
4. Suggest actionable strategies
5. Respond in user's language
6. Crisis detection → 988 referral

### Fallback System (when API unavailable):
- 3+ unique responses per category (anxiety, depression, stress, normal, severe)
- Validation → Strategy → Follow-up question pattern
- Never repeats consecutively

---

## 8. CLINICAL INSTRUMENTS

### PHQ-9 (Patient Health Questionnaire-9):
- **Purpose:** Depression screening
- **Validation:** Sensitivity 88%, Specificity 88% at cutoff ≥10
- **Reference:** Kroenke, Spitzer & Williams, 2001, *J Gen Intern Med*
- **Scoring:** 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 Moderately Severe, 20-27 Severe
- **Safety:** Item 9 ≥2 triggers crisis alert

### GAD-7 (Generalized Anxiety Disorder 7-item):
- **Purpose:** Anxiety screening
- **Validation:** Sensitivity 89%, Specificity 82% at cutoff ≥10
- **Reference:** Spitzer, Kroenke, Williams & Löwe, 2006, *Arch Intern Med*
- **Scoring:** 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-21 Severe

---

## 9. GAMIFICATION

### Badge System (12 badges, 3 tiers):

| Badge | Requirement | Tier |
|-------|-------------|------|
| First Analysis | 1 deep analysis | Bronze |
| Pattern Seeker | 5 analyses | Silver |
| Mind Explorer | 15 analyses | Gold |
| Self-Aware | 1 daily check-in | Bronze |
| Consistent | 3 check-ins | Silver |
| Dedicated | 7 check-ins | Gold |
| Reflective | 1 journal entry | Bronze |
| Deep Thinker | 5 journal entries | Silver |
| Thought Reframer | 1 CBT reframe | Silver |
| CBT Master | 5 reframes | Gold |
| Healthy Mind | Risk below 15% | Silver |
| Thriving | 80%+ wellness | Gold |

### XP Economy:
- 1 badge = 100 XP
- 3 badges = 1 level (300 XP per level)
- Max Level 5 (1200 XP)

---

## 10. INTERVIEW QUESTIONS & ANSWERS

### ML / AI Questions:

**Q: How does the ML pipeline work?**
"Text goes through 4 stages: First, BERT tokenizer converts it to WordPiece tokens. Then sentence-transformers encodes it into a 384-dimensional dense vector. That vector does two things simultaneously — FAISS searches the knowledge base for the 3 most similar documents, and the BERT classifier predicts one of 5 mental health states through a softmax layer. The retrieved contexts feed into a RAG pipeline that generates a response conditioned on the detected severity."

**Q: What's FAISS and why use it?**
"FAISS is Facebook's library for efficient similarity search over dense vectors. I use it to find which knowledge base documents are most relevant to the user's input. Each document is pre-embedded into 384 dimensions with all-MiniLM-L6-v2. When a user types text, it's embedded the same way, then FAISS finds the nearest neighbors by cosine distance in milliseconds. This is the 'Retrieval' in RAG."

**Q: What's RAG?**
"Retrieval-Augmented Generation. Instead of generating responses from scratch (which can hallucinate), the system RETRIEVES relevant factual content first, then generates a response grounded in that context. My knowledge base has 8 curated clinical documents. This ensures responses are evidence-based."

**Q: Explain the risk scoring algorithm.**
"It's a weighted ensemble of 4 independent signals: keyword severity (40%), emotional distress from Plutchik analysis (25%), cognitive distortion count (20%), and linguistic biomarkers (15%). Each produces a 0-1 score, they're weighted and summed, then clipped to [0,1]. This multi-factor approach is more robust than any single signal."

**Q: What are the linguistic biomarkers?**
"Research by Pennebaker (2011) and Rude et al. (2004) shows specific language patterns correlate with mental health. I extract: self-reference ratio (high 'I/me/my' → depression), negation density (more 'not/never' → negative affect), lexical diversity (low → rumination), certainty language (absolutist words → anxiety), and temporal focus."

**Q: How is the chatbot different from just calling ChatGPT?**
"Three ways: (1) I run my own fine-tuned BERT for classification — not a generic model. (2) RAG ensures domain-specific, evidence-based responses. (3) Multi-factor risk scoring combines 4 NLP signals into a clinically-interpretable score. Plus the chatbot passes real-time NLP analysis to the LLM as context — it doesn't just forward the message."

**Q: How did you train the model?**
"200K Reddit posts from mental health subreddits, labeled into 5 classes. Stratified 80/10/10 split. BERT-base-uncased with a custom classification head (768→256→5, dropout 0.3). AdamW optimizer at 2e-5 with linear warmup for 10% of steps. Gradient clipping at 1.0. Early stopping on validation F1. Final test: 0.87 macro-F1 on 40K held-out samples."

### Frontend Questions:

**Q: Why Vite over Next.js for this project?**
"MindGuard's frontend is a single-page application — no SEO needed, no server-side rendering required. Vite gives 10-50x faster dev server starts and HMR in <50ms. The ML backend is separate on HuggingFace, so I don't need API routes. If I needed SEO for a landing page, I'd use Next.js (like I did with CureCue)."

**Q: How does the voice system handle multiple languages?**
"I built a language detection function that checks the text for script patterns — Devanagari for Hindi, Arabic script, CJK characters, etc. It also detects romanized Hindi by matching common words (hai, mein, kya, nahi). Based on detected language, it selects a matching voice from the browser's available voices. Speech recognition doesn't have a hardcoded language — it accepts whatever the user speaks."

**Q: Explain the ML Pipeline Visualizer page.**
"It's a 4-stage animated visualization showing what happens inside the neural network. Stage 1: tokenization (color-coded tokens with vocab IDs). Stage 2: embedding (heatmap of 48/384 dimensions). Stage 3: FAISS search (similarity bars against knowledge base). Stage 4: classification (probability bars for 5 classes). Each stage appears sequentially with a 700ms delay to create a 'watching the AI think' experience."

**Q: How does the Emotion Challenge game work?**
"8 curated text passages with a hidden dominant emotion. User reads the text, picks from 8 Plutchik emotions, then sees if they matched the ML model. After each guess, the full NLP emotion breakdown is shown with an explanation of WHY the AI detected that emotion. It scores correct/total and teaches NLP concepts through play."

### Behavioral Questions:

**Q: What was the hardest part?**
"Two things: (1) The multi-factor risk scoring — balancing the weights so that no single signal dominates required iterative testing. (2) Making the voice system work across languages — the Web Speech API's voice list is async, and matching voices to detected languages has edge cases (romanized Hindi vs English, code-switching)."

**Q: What would you add with more time?**
"(1) SHAP/LIME explainability — show which specific words triggered the classification. (2) Longitudinal pattern detection — track users over weeks and predict deterioration using time-series analysis. (3) Multimodal input — analyze voice tone (pitch, speed, pauses) alongside text for better accuracy. (4) Federated learning for personalization without data leaving the device."

**Q: Why mental health?**
"75% of mental health conditions develop before age 24. 75% never receive treatment. Current digital tools are either too simple (single questionnaire) or opaque (black-box). MindGuard provides transparent, multi-factor analysis that users can actually understand — the ML Pipeline Visualizer literally shows them how the AI reaches its conclusions."

---

## 11. HOW TO PRESENT IN AN INTERVIEW

### Opening:

"I'll talk about MindGuard — it's an ML-powered mental health screening platform. What makes it technically interesting: I trained a BERT model on 200K samples for 5-class mental health classification, built a FAISS vector database for semantic retrieval, implemented a full RAG pipeline with LangChain, and designed a multi-factor risk scoring algorithm that combines 4 independent NLP signals. The frontend has 15 interactive pages including an ML pipeline visualizer where you can watch the neural network process text in real-time, a CBT chatbot with conversation memory and multilingual voice I/O, and a gamified emotion detection challenge."

### Demo Flow (90 seconds):

1. Open landing page → show Dr. Milo character and features
2. Deep Analysis → type/speak text → show 6-dimension results with risk gauge
3. ML Pipeline → same text → watch tokenization → embedding → FAISS → classify
4. Emotion Challenge → play one round → show AI explanation
5. Chat → speak in Hindi → AI responds in Hindi with CBT technique

### When They Compare to Other Projects:

"CureCue proves backend engineering — JWT auth, MongoDB, API security. MindLink proves complex frontend — Web Audio, data visualization, client-side state. MindGuard proves **ML engineering** — model training, vector databases, RAG pipelines, NLP feature engineering, and deploying ML to production. The combination shows full-stack + ML capability."

---

## 12. QUICK REFERENCE CARD

```
Project:     MindGuard — AI Mental Health Platform
Stack:       React + BERT + FAISS + LangChain + Gradio
Frontend:    https://mind-guard-chi.vercel.app
Backend:     https://huggingface.co/spaces/MallikaV/MindGuard
GitHub:      github.com/Mallika-coder/MindGuard

ML:          BERT (F1=0.87), FAISS, RAG, 200K dataset
Frontend:    React 18, Vite, Tailwind, Framer Motion, Recharts
Voice:       Web Speech API (input + output, multilingual)
Clinical:    PHQ-9, GAD-7, CBT chatbot, risk scoring
Deploy:      Vercel (frontend) + HuggingFace Spaces (ML)

Key Files:
  app.py                  — ML backend (Gradio + all NLP)
  model/train.py          — BERT training pipeline
  model/classifier.py     — Model architecture
  backend/app/rag_pipeline.py — FAISS + LangChain RAG
  frontend/src/pages/     — 15 React page components
  frontend/src/hooks/useVoice.js — Voice I/O + lang detection
  frontend/src/utils/wellness.js — Client-side NLP functions

Differentiators:
  ✅ Real BERT model training (not just API calls)
  ✅ FAISS vector search (not keyword matching)
  ✅ RAG pipeline (not hallucinating responses)
  ✅ Multi-factor risk scoring (published research basis)
  ✅ ML Pipeline Visualizer (interactive education)
  ✅ Voice I/O with auto language detection
  ✅ CBT chatbot with conversation memory
  ✅ Gamified emotion challenge (learn NLP by playing)
```
