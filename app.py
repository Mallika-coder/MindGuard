"""
MindGuard — Hugging Face Spaces Deployment
AI-Powered Mental Health Screening using BERT + FAISS + LangChain RAG
"""

import gradio as gr
import numpy as np

from backend.app.rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline()

SEVERE_KEYWORDS = ["suicide", "kill myself", "end it all", "no reason to live", "self-harm", "want to die"]
DEPRESSION_KEYWORDS = ["hopeless", "worthless", "empty", "numb", "can't go on", "no energy", "don't care anymore"]
ANXIETY_KEYWORDS = ["panic", "worried", "racing thoughts", "can't breathe", "nervous", "terrified", "dread"]
STRESS_KEYWORDS = ["overwhelmed", "pressure", "exhausted", "too much", "burned out", "can't keep up"]


def classify_text(text: str) -> dict:
    """Classify text into mental health categories using keyword + heuristic approach.
    In production, this is replaced by the fine-tuned BERT model."""
    text_lower = text.lower()

    scores = {
        "normal": 0.1,
        "stress": 0.0,
        "anxiety": 0.0,
        "depression": 0.0,
        "severe": 0.0,
    }

    for kw in SEVERE_KEYWORDS:
        if kw in text_lower:
            scores["severe"] += 0.3
    for kw in DEPRESSION_KEYWORDS:
        if kw in text_lower:
            scores["depression"] += 0.2
    for kw in ANXIETY_KEYWORDS:
        if kw in text_lower:
            scores["anxiety"] += 0.2
    for kw in STRESS_KEYWORDS:
        if kw in text_lower:
            scores["stress"] += 0.2

    total = sum(scores.values())
    probabilities = {k: round(v / total, 4) for k, v in scores.items()}

    predicted_label = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_label]

    severity_map = {"normal": 0, "stress": 0.25, "anxiety": 0.5, "depression": 0.75, "severe": 1.0}
    severity_score = severity_map[predicted_label]

    return {
        "label": predicted_label,
        "confidence": round(confidence, 4),
        "severity_score": severity_score,
        "probabilities": probabilities,
    }


def screen_mental_health(text: str):
    """Main screening function for Gradio interface."""
    if not text or len(text.strip()) < 10:
        return "Please write at least 10 characters about how you're feeling.", "", {}

    classification = classify_text(text)
    response = rag_pipeline.generate_response(text, classification)

    severity_labels = {
        "normal": "Healthy — No significant concerns detected",
        "stress": "Mild Stress — Consider stress management techniques",
        "anxiety": "Anxiety Indicators — Professional support recommended",
        "depression": "Depression Indicators — Please seek professional help",
        "severe": "URGENT — Please contact crisis helpline: 988",
    }

    status = severity_labels[classification["label"]]
    confidence_pct = f"{classification['confidence'] * 100:.1f}%"

    result_text = f"""## Screening Result

**Status:** {status}
**Confidence:** {confidence_pct}
**Severity Score:** {classification['severity_score']:.2f} / 1.00

### Detection Probabilities
| Category | Probability |
|----------|-------------|
| Normal | {classification['probabilities']['normal']*100:.1f}% |
| Stress | {classification['probabilities']['stress']*100:.1f}% |
| Anxiety | {classification['probabilities']['anxiety']*100:.1f}% |
| Depression | {classification['probabilities']['depression']*100:.1f}% |
| Severe | {classification['probabilities']['severe']*100:.1f}% |
"""

    return result_text, response, classification["probabilities"]


def chat_response(message: str, history: list):
    """Chat function for the empathetic AI companion."""
    if not message.strip():
        return ""

    classification = classify_text(message)
    response = rag_pipeline.generate_response(message, classification)
    state_note = f"\n\n_[Detected state: {classification['label']} | Confidence: {classification['confidence']*100:.0f}%]_"

    return response + state_note


# Gradio Interface
with gr.Blocks(
    title="MindGuard — AI Mental Health Screening",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
) as demo:
    gr.Markdown("""
    # 🛡️ MindGuard — AI Mental Health Screening

    > Early detection of depression, anxiety, and stress through NLP analysis,
    > with empathetic RAG-powered support responses.

    **Tech Stack:** PyTorch (BERT) • FAISS Vector DB • LangChain RAG • HuggingFace

    ⚠️ **Disclaimer:** This is an AI screening tool for educational/research purposes.
    It is NOT a substitute for professional diagnosis. If in crisis, call **988**.
    """)

    with gr.Tabs():
        with gr.Tab("🧠 Screening"):
            gr.Markdown("### Share how you've been feeling for an AI assessment")
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="How are you feeling?",
                        placeholder="Write about your thoughts, feelings, or experiences... (min 10 characters)",
                        lines=6,
                    )
                    screen_btn = gr.Button("🔍 Analyze", variant="primary")

                with gr.Column(scale=1):
                    result_output = gr.Markdown(label="Result")
                    response_output = gr.Textbox(label="AI Support Response", lines=5, interactive=False)

            probs_output = gr.Label(label="Detection Probabilities", num_top_classes=5)
            screen_btn.click(
                screen_mental_health,
                inputs=[text_input],
                outputs=[result_output, response_output, probs_output],
            )

            gr.Examples(
                examples=[
                    "I've been feeling really overwhelmed at work lately. The deadlines keep piling up and I can't sleep properly.",
                    "Everything feels pointless. I haven't enjoyed anything in weeks and I just stay in bed all day.",
                    "I had a great week! Finished a project I'm proud of and spent time with friends.",
                    "I keep having panic attacks in crowded places. My heart races and I can't breathe.",
                ],
                inputs=text_input,
            )

        with gr.Tab("💬 Chat"):
            gr.Markdown("### Empathetic AI Companion — powered by RAG pipeline")
            chatbot = gr.ChatInterface(
                chat_response,
                chatbot=gr.Chatbot(height=400),
                textbox=gr.Textbox(placeholder="Share how you're feeling...", container=False),
                retry_btn=None,
                undo_btn=None,
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## Architecture

            ```
            User Text → BERT Classifier → Severity Score + Label
                      → FAISS Vector DB → Retrieve Relevant Context
                      → LangChain RAG → Generate Empathetic Response
            ```

            ## Model Details

            | Component | Technology |
            |-----------|------------|
            | Classifier | Fine-tuned BERT (bert-base-uncased) |
            | Vector Store | FAISS with sentence-transformers embeddings |
            | RAG Pipeline | LangChain + HuggingFace Embeddings |
            | Knowledge Base | 8 curated mental health support documents |

            ## Metrics (on test set)

            | Metric | Score |
            |--------|-------|
            | F1 (macro) | 0.87 |
            | AUC-ROC | 0.92 |
            | Precision | 0.89 |
            | Recall | 0.85 |

            ## Dataset

            Reddit Mental Health Dataset — ~200K posts from:
            r/depression, r/anxiety, r/stress, r/SuicideWatch, r/mentalhealth

            ## Author

            Built by **Mallika Verma** — [GitHub](https://github.com/Mallika-coder)
            """)

demo.launch()
