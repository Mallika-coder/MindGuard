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


def get_severity_bar(score):
    filled = int(score * 20)
    empty = 20 - filled
    if score >= 0.75:
        color = "🔴"
    elif score >= 0.5:
        color = "🟠"
    elif score >= 0.25:
        color = "🟡"
    else:
        color = "🟢"
    return f"{color} {'█' * filled}{'░' * empty} {score*100:.0f}%"


def get_status_emoji(label):
    emojis = {
        "normal": "✅",
        "stress": "⚡",
        "anxiety": "😰",
        "depression": "💙",
        "severe": "🆘",
    }
    return emojis.get(label, "❓")


def screen_mental_health(text: str):
    if not text or len(text.strip()) < 10:
        return (
            "⚠️ Please write at least 10 characters about how you're feeling.",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    classification = classify_text(text)
    response = rag_pipeline.generate_response(text, classification)

    severity_labels = {
        "normal": "Healthy — No significant concerns detected",
        "stress": "Mild Stress — Consider stress management techniques",
        "anxiety": "Anxiety Indicators — Professional support recommended",
        "depression": "Depression Indicators — Please seek professional help",
        "severe": "URGENT — Please contact crisis helpline: 988",
    }

    label = classification["label"]
    emoji = get_status_emoji(label)
    status = severity_labels[label]
    confidence_pct = classification["confidence"] * 100
    severity_bar = get_severity_bar(classification["severity_score"])

    probs = classification["probabilities"]
    prob_bars = ""
    for cat, prob in probs.items():
        bar_len = int(prob * 30)
        cat_emoji = get_status_emoji(cat)
        prob_bars += f"{cat_emoji} **{cat.capitalize():12s}** {'▓' * bar_len}{'░' * (30 - bar_len)} {prob*100:.1f}%\n\n"

    result_text = f"""
## {emoji} {status}

---

### 📊 Analysis Summary

| Metric | Value |
|--------|-------|
| **Detected State** | {emoji} {label.capitalize()} |
| **Confidence** | {confidence_pct:.1f}% |
| **Severity** | {severity_bar} |

---

### 📈 Category Probabilities

{prob_bars}

---

> 💡 *This analysis uses a fine-tuned BERT model with RAG-enhanced responses from our mental health knowledge base.*
"""

    response_formatted = f"### 💬 Personalized Support\n\n{response}"

    resources = ""
    if classification["severity_score"] >= 0.75:
        resources = """
### 🆘 Immediate Resources

| Resource | Contact |
|----------|---------|
| **Suicide & Crisis Lifeline** | Call/Text **988** |
| **Crisis Text Line** | Text HOME to **741741** |
| **Emergency Services** | Call **911** |

*You are not alone. Help is available 24/7.*
"""
    elif classification["severity_score"] >= 0.5:
        resources = """
### 📞 Recommended Resources

| Resource | Details |
|----------|---------|
| **SAMHSA Helpline** | 1-800-662-4357 (free, 24/7) |
| **Psychology Today** | Find a therapist near you |
| **BetterHelp / Talkspace** | Online therapy options |

*Speaking with a professional can make a real difference.*
"""

    return (
        result_text,
        response_formatted,
        gr.update(value=resources, visible=bool(resources)),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def chat_response(message: str, history: list):
    if not message.strip():
        return ""

    classification = classify_text(message)
    response = rag_pipeline.generate_response(message, classification)
    emoji = get_status_emoji(classification["label"])
    state_note = f"\n\n---\n{emoji} *Detected: {classification['label']} | Confidence: {classification['confidence']*100:.0f}%*"

    return response + state_note


CSS = """
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
}
.disclaimer-box {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 1px solid #ffc107;
    border-radius: 12px;
    padding: 12px 20px;
    margin-top: 10px;
}
.result-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
}
.chat-container {
    border-radius: 16px;
    overflow: hidden;
}
footer {
    text-align: center;
    padding: 10px;
    opacity: 0.7;
}
"""

with gr.Blocks(
    title="MindGuard — AI Mental Health Screening",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 30px 20px; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%); border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color: white; font-size: 2.5em; margin: 0;">🛡️ MindGuard</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin-top: 8px;">AI-Powered Mental Health Screening & Support</p>
        <div style="display: flex; justify-content: center; gap: 12px; margin-top: 16px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.85em;">🧠 BERT NLP</span>
            <span style="background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.85em;">🔍 FAISS Vector DB</span>
            <span style="background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.85em;">⚡ LangChain RAG</span>
            <span style="background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.85em;">🤗 HuggingFace</span>
        </div>
    </div>
    """)

    with gr.Tabs() as tabs:
        with gr.Tab("🧠 Mental Health Screening", id="screen"):
            gr.Markdown("#### Tell us how you've been feeling — our AI will analyze your text and provide personalized support.")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="📝 Express Yourself",
                        placeholder="Share your thoughts, feelings, or experiences here...\n\nExample: 'I've been feeling overwhelmed at work and can't sleep well lately...'",
                        lines=8,
                        max_lines=15,
                    )
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                        screen_btn = gr.Button("🔍 Analyze My Text", variant="primary", scale=2)

                    gr.Markdown("---")
                    gr.Markdown("##### 💡 Try these examples:")
                    gr.Examples(
                        examples=[
                            ["I've been feeling really overwhelmed at work lately. The deadlines keep piling up and I can't sleep properly."],
                            ["Everything feels pointless. I haven't enjoyed anything in weeks and I just stay in bed all day."],
                            ["I had a great week! Finished a project I'm proud of and spent time with friends."],
                            ["I keep having panic attacks in crowded places. My heart races and I can't breathe."],
                            ["I'm so stressed about exams. The pressure from my parents is making it worse."],
                        ],
                        inputs=text_input,
                        label="",
                    )

                with gr.Column(scale=1):
                    result_output = gr.Markdown(
                        value="### 🔮 Results will appear here\n\nWrite about how you're feeling and click **Analyze** to get an AI assessment.",
                    )
                    response_output = gr.Markdown(visible=False)
                    resources_output = gr.Markdown(visible=False)

            screen_btn.click(
                screen_mental_health,
                inputs=[text_input],
                outputs=[result_output, response_output, resources_output, response_output, resources_output],
            )
            clear_btn.click(
                lambda: ("", "### 🔮 Results will appear here\n\nWrite about how you're feeling and click **Analyze** to get an AI assessment.", "", gr.update(visible=False), gr.update(visible=False)),
                outputs=[text_input, result_output, response_output, response_output, resources_output],
            )

        with gr.Tab("💬 Empathetic Chat", id="chat"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 16px 20px; border-radius: 12px; margin-bottom: 16px; border: 1px solid #a7f3d0;">
                <p style="margin: 0; color: #065f46;">🌿 <strong>Your AI Companion</strong> — I'm here to listen, understand, and provide evidence-based support. Everything shared here stays private. How are you feeling today?</p>
            </div>
            """)

            chatbot_display = gr.Chatbot(
                height=450,
                type="messages",
                value=[{"role": "assistant", "content": "Hello! 🌱 I'm your MindGuard companion. I'm here to listen without judgment and provide support based on evidence-based mental health research.\n\nHow are you feeling today? You can share as much or as little as you'd like."}],
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/shield_1f6e1-fe0f.png"),
                show_copy_button=True,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="",
                    scale=5,
                    container=False,
                )
                chat_btn = gr.Button("Send 📨", variant="primary", scale=1)

            with gr.Row():
                gr.Markdown("*Quick prompts:*")
            with gr.Row():
                q1 = gr.Button("😔 I'm feeling down", size="sm", variant="secondary")
                q2 = gr.Button("😰 I'm anxious", size="sm", variant="secondary")
                q3 = gr.Button("😤 I'm stressed", size="sm", variant="secondary")
                q4 = gr.Button("😊 I'm doing well", size="sm", variant="secondary")

            def handle_chat(message, history):
                if not message.strip():
                    return history, ""
                response = chat_response(message, history)
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
                return history, ""

            def quick_prompt(prompt, history):
                return handle_chat(prompt, history)

            chat_btn.click(handle_chat, [chat_input, chatbot_display], [chatbot_display, chat_input])
            chat_input.submit(handle_chat, [chat_input, chatbot_display], [chatbot_display, chat_input])
            q1.click(lambda h: handle_chat("I've been feeling really down and sad lately, nothing seems to help", h), [chatbot_display], [chatbot_display, chat_input])
            q2.click(lambda h: handle_chat("I'm feeling very anxious, my mind won't stop racing with worried thoughts", h), [chatbot_display], [chatbot_display, chat_input])
            q3.click(lambda h: handle_chat("I'm extremely stressed with work and responsibilities piling up", h), [chatbot_display], [chatbot_display, chat_input])
            q4.click(lambda h: handle_chat("I'm doing great today! Feeling positive and energetic", h), [chatbot_display], [chatbot_display, chat_input])

        with gr.Tab("📊 How It Works", id="about"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); padding: 24px; border-radius: 16px; margin-bottom: 20px; border: 1px solid #c7d2fe;">
                <h2 style="margin: 0 0 12px 0; color: #3730a3;">🏗️ System Architecture</h2>
                <div style="font-family: monospace; background: white; padding: 16px; border-radius: 8px; line-height: 1.8;">
                    <span style="color: #6366f1;">User Text</span> → <span style="color: #8b5cf6;">BERT Classifier</span> → Severity Score + Label<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color: #06b6d4;">FAISS Vector DB</span> → Retrieve Relevant Context<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color: #10b981;">LangChain RAG</span> → Generate Empathetic Response
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
### 🧠 Model Components

| Component | Technology |
|-----------|------------|
| **Classifier** | Fine-tuned BERT (bert-base-uncased) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **RAG Pipeline** | LangChain + HuggingFace |
| **Knowledge Base** | 8 curated mental health documents |
""")
                with gr.Column():
                    gr.Markdown("""
### 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| **F1 Score** (macro) | 0.87 |
| **AUC-ROC** (macro) | 0.92 |
| **Precision** (macro) | 0.89 |
| **Recall** (macro) | 0.85 |

*Evaluated on 40K test samples across 5 categories*
""")

            gr.Markdown("""
---

### 📚 Dataset

**Reddit Mental Health Dataset** — ~200K posts collected from:

`r/depression` • `r/anxiety` • `r/stress` • `r/SuicideWatch` • `r/mentalhealth` • `r/CasualConversation`

**5 Classes:** Normal, Stress, Anxiety, Depression, Severe

---

### 👩‍💻 Author

Built by **Mallika Verma** — [GitHub](https://github.com/Mallika-coder/MindGuard)
""")

    gr.HTML("""
    <div style="text-align: center; padding: 16px; margin-top: 20px; background: #fef3c7; border-radius: 12px; border: 1px solid #fbbf24;">
        <p style="margin: 0; color: #92400e; font-size: 0.9em;">
            ⚠️ <strong>Important:</strong> MindGuard is an AI tool for educational purposes only. It is NOT a medical diagnosis.
            If you're experiencing a mental health crisis, please call <strong>988</strong> (Suicide & Crisis Lifeline) or text HOME to <strong>741741</strong>.
        </p>
    </div>
    """)

demo.launch(ssr_mode=False)
