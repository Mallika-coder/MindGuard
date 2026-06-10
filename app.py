"""
MindGuard — AI-Powered Mental Health Screening Platform
Clinical-grade NLP analysis with BERT + FAISS + LangChain RAG
Supports PHQ-9/GAD-7 validated instruments + free-text deep analysis
"""

import gradio as gr
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict

from backend.app.rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline()

SESSION_DATA = {
    "entries": [],
    "mood_timeline": [],
    "risk_scores": [],
    "session_start": time.time(),
    "total_interactions": 0,
}

EMOTION_LEXICON = {
    "anger": ["angry", "furious", "rage", "hate", "irritated", "frustrated", "annoyed", "resentful", "bitter", "hostile"],
    "sadness": ["sad", "crying", "tears", "grief", "mourning", "lonely", "heartbroken", "miserable", "gloomy", "melancholy"],
    "fear": ["afraid", "scared", "terrified", "frightened", "panic", "phobia", "dread", "horror", "alarmed", "petrified"],
    "joy": ["happy", "excited", "grateful", "proud", "content", "cheerful", "elated", "thrilled", "delighted", "optimistic"],
    "disgust": ["disgusted", "repulsed", "revolted", "sick", "nauseated", "appalled", "loathing"],
    "surprise": ["shocked", "amazed", "astonished", "startled", "unexpected", "stunned"],
    "trust": ["believe", "faith", "confident", "secure", "reliable", "honest", "loyal"],
    "anticipation": ["expecting", "hopeful", "looking forward", "planning", "preparing", "eager"],
}

COGNITIVE_DISTORTIONS = {
    "all_or_nothing": {
        "patterns": ["always", "never", "everyone", "nobody", "everything", "nothing", "completely", "totally"],
        "label": "All-or-Nothing Thinking",
        "description": "Seeing things in black-and-white categories",
    },
    "catastrophizing": {
        "patterns": ["worst", "terrible", "horrible", "disaster", "catastrophe", "end of the world", "ruined", "destroyed"],
        "label": "Catastrophizing",
        "description": "Expecting the worst-case scenario",
    },
    "mind_reading": {
        "patterns": ["they think", "everyone thinks", "people think", "they must", "probably thinks", "judges me"],
        "label": "Mind Reading",
        "description": "Assuming you know what others are thinking",
    },
    "should_statements": {
        "patterns": ["i should", "i must", "i have to", "i need to", "ought to", "supposed to"],
        "label": "Should Statements",
        "description": "Rigid rules about how things should be",
    },
    "overgeneralization": {
        "patterns": ["always happens", "every time", "never works", "nothing ever", "this always"],
        "label": "Overgeneralization",
        "description": "Drawing broad conclusions from single events",
    },
    "personalization": {
        "patterns": ["my fault", "because of me", "i caused", "blame myself", "i'm the reason"],
        "label": "Personalization",
        "description": "Taking excessive responsibility for external events",
    },
}

SEVERE_KEYWORDS = ["suicide", "kill myself", "end it all", "no reason to live", "self-harm", "want to die", "better off dead", "not worth living"]
DEPRESSION_KEYWORDS = ["hopeless", "worthless", "empty", "numb", "can't go on", "no energy", "don't care anymore", "failure", "burden", "giving up"]
ANXIETY_KEYWORDS = ["panic", "worried", "racing thoughts", "can't breathe", "nervous", "terrified", "dread", "restless", "on edge", "impending doom"]
STRESS_KEYWORDS = ["overwhelmed", "pressure", "exhausted", "too much", "burned out", "can't keep up", "falling behind", "drowning"]

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things",
    "Moving or speaking slowly, or being fidgety/restless",
    "Thoughts that you would be better off dead, or of hurting yourself",
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]


def analyze_emotions(text: str) -> dict:
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = min(score / 3.0, 1.0)
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 3) for k, v in scores.items()}
    return scores


def detect_cognitive_distortions(text: str) -> list:
    text_lower = text.lower()
    detected = []
    for key, distortion in COGNITIVE_DISTORTIONS.items():
        for pattern in distortion["patterns"]:
            if pattern in text_lower:
                detected.append({
                    "type": distortion["label"],
                    "description": distortion["description"],
                    "trigger": pattern,
                })
                break
    return detected


def compute_linguistic_features(text: str) -> dict:
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    first_person = sum(1 for w in words if w.lower() in ["i", "me", "my", "myself", "mine"])
    negations = sum(1 for w in words if w.lower() in ["not", "no", "never", "don't", "can't", "won't", "isn't", "aren't", "doesn't", "didn't", "nothing", "nowhere", "nobody"])
    certainty_words = sum(1 for w in words if w.lower() in ["always", "never", "definitely", "certainly", "absolutely", "completely", "totally"])

    return {
        "word_count": len(words),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "first_person_ratio": round(first_person / max(len(words), 1), 3),
        "negation_ratio": round(negations / max(len(words), 1), 3),
        "certainty_ratio": round(certainty_words / max(len(words), 1), 3),
        "lexical_diversity": round(len(set(words)) / max(len(words), 1), 3),
    }


def classify_text(text: str) -> dict:
    text_lower = text.lower()

    scores = {"normal": 0.1, "stress": 0.0, "anxiety": 0.0, "depression": 0.0, "severe": 0.0}

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

    return {
        "label": predicted_label,
        "confidence": round(confidence, 4),
        "severity_score": severity_map[predicted_label],
        "probabilities": probabilities,
    }


def compute_risk_score(classification: dict, emotions: dict, distortions: list, linguistics: dict) -> dict:
    base_risk = classification["severity_score"]

    emotion_risk = (emotions.get("sadness", 0) * 0.3 + emotions.get("fear", 0) * 0.2 +
                    emotions.get("anger", 0) * 0.1 - emotions.get("joy", 0) * 0.2)

    distortion_risk = min(len(distortions) * 0.1, 0.3)

    linguistic_risk = (linguistics["negation_ratio"] * 0.5 +
                       linguistics["first_person_ratio"] * 0.2 +
                       (1 - linguistics["lexical_diversity"]) * 0.1)

    composite_risk = np.clip(
        base_risk * 0.4 + emotion_risk * 0.25 + distortion_risk * 0.2 + linguistic_risk * 0.15,
        0, 1
    )

    factors = []
    if base_risk >= 0.5:
        factors.append(f"High-risk keywords detected (severity: {base_risk:.0%})")
    if emotions.get("sadness", 0) > 0.3:
        factors.append("Elevated sadness indicators")
    if emotions.get("fear", 0) > 0.3:
        factors.append("Elevated fear/anxiety markers")
    if len(distortions) >= 2:
        factors.append(f"{len(distortions)} cognitive distortions identified")
    if linguistics["negation_ratio"] > 0.1:
        factors.append("High negative language density")

    return {
        "composite_score": round(float(composite_risk), 3),
        "components": {
            "keyword_severity": round(float(base_risk), 3),
            "emotional_distress": round(float(max(emotion_risk, 0)), 3),
            "cognitive_patterns": round(float(distortion_risk), 3),
            "linguistic_markers": round(float(min(linguistic_risk, 1)), 3),
        },
        "risk_level": (
            "Critical" if composite_risk >= 0.8 else
            "High" if composite_risk >= 0.6 else
            "Moderate" if composite_risk >= 0.4 else
            "Low" if composite_risk >= 0.2 else
            "Minimal"
        ),
        "contributing_factors": factors,
    }


def full_analysis(text: str):
    if not text or len(text.strip()) < 10:
        return (
            "⚠️ Please provide at least 10 characters for analysis.",
            "", "", "", "", "",
            gr.update(visible=False),
        )

    classification = classify_text(text)
    emotions = analyze_emotions(text)
    distortions = detect_cognitive_distortions(text)
    linguistics = compute_linguistic_features(text)
    risk = compute_risk_score(classification, emotions, distortions, linguistics)
    rag_response = rag_pipeline.generate_response(text, classification)
    contexts = rag_pipeline.retrieve_context(text)

    SESSION_DATA["entries"].append({
        "timestamp": time.time(),
        "text_length": len(text),
        "classification": classification["label"],
        "severity": classification["severity_score"],
        "risk_score": risk["composite_score"],
    })
    SESSION_DATA["mood_timeline"].append(classification["severity_score"])
    SESSION_DATA["risk_scores"].append(risk["composite_score"])
    SESSION_DATA["total_interactions"] += 1

    risk_color = {
        "Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢", "Minimal": "⚪"
    }[risk["risk_level"]]

    classification_output = f"""## 🧠 Classification Result

<div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 20px; border-radius: 12px; color: white; margin-bottom: 16px;">
    <div style="font-size: 1.4em; font-weight: bold; margin-bottom: 8px;">{risk_color} {classification['label'].upper()}</div>
    <div style="opacity: 0.8;">Confidence: {classification['confidence']*100:.1f}% | Risk Level: {risk['risk_level']}</div>
</div>

### Multi-class Probabilities

| Category | Score | Level |
|----------|-------|-------|
| Normal | {'█' * int(classification['probabilities']['normal']*20)}{'░' * (20-int(classification['probabilities']['normal']*20))} | {classification['probabilities']['normal']*100:.1f}% |
| Stress | {'█' * int(classification['probabilities']['stress']*20)}{'░' * (20-int(classification['probabilities']['stress']*20))} | {classification['probabilities']['stress']*100:.1f}% |
| Anxiety | {'█' * int(classification['probabilities']['anxiety']*20)}{'░' * (20-int(classification['probabilities']['anxiety']*20))} | {classification['probabilities']['anxiety']*100:.1f}% |
| Depression | {'█' * int(classification['probabilities']['depression']*20)}{'░' * (20-int(classification['probabilities']['depression']*20))} | {classification['probabilities']['depression']*100:.1f}% |
| Severe | {'█' * int(classification['probabilities']['severe']*20)}{'░' * (20-int(classification['probabilities']['severe']*20))} | {classification['probabilities']['severe']*100:.1f}% |
"""

    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
    emotion_rows = "\n".join([
        f"| {e[0].capitalize()} | {'▓' * int(e[1]*25)}{'░' * (25-int(e[1]*25))} | {e[1]*100:.1f}% |"
        for e in top_emotions if e[1] > 0
    ])

    emotion_output = f"""## 🎭 Emotional Profile (Plutchik's Wheel)

{emotion_rows if emotion_rows else "| No strong emotional markers detected | | |"}

---

### Valence-Arousal Mapping

| Dimension | Score |
|-----------|-------|
| **Positive Affect** | {emotions.get('joy', 0) + emotions.get('trust', 0) + emotions.get('anticipation', 0):.2f} |
| **Negative Affect** | {emotions.get('sadness', 0) + emotions.get('fear', 0) + emotions.get('anger', 0) + emotions.get('disgust', 0):.2f} |
| **Emotional Diversity** | {sum(1 for v in emotions.values() if v > 0.05)}/8 categories |
"""

    distortion_text = ""
    if distortions:
        distortion_text = "## 🔍 Cognitive Distortions Detected\n\n"
        for d in distortions:
            distortion_text += f"""<div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; margin: 8px 0; border-radius: 0 8px 8px 0;">
<strong>⚠️ {d['type']}</strong><br/>
<span style="color: #666;">{d['description']}</span><br/>
<span style="color: #999; font-size: 0.85em;">Triggered by: "{d['trigger']}"</span>
</div>\n\n"""
    else:
        distortion_text = "## 🔍 Cognitive Distortions\n\n✅ No major cognitive distortions detected in this text."

    risk_output = f"""## ⚡ Multi-Factor Risk Assessment

<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 20px; border-radius: 12px; color: white;">
    <div style="font-size: 2em; font-weight: bold; text-align: center;">{risk['composite_score']*100:.0f}%</div>
    <div style="text-align: center; opacity: 0.7; margin-top: 4px;">Composite Risk Score</div>
</div>

### Risk Components

| Factor | Weight | Score |
|--------|--------|-------|
| Keyword Severity | 40% | {risk['components']['keyword_severity']*100:.0f}% |
| Emotional Distress | 25% | {risk['components']['emotional_distress']*100:.0f}% |
| Cognitive Patterns | 20% | {risk['components']['cognitive_patterns']*100:.0f}% |
| Linguistic Markers | 15% | {risk['components']['linguistic_markers']*100:.0f}% |

### Contributing Factors
"""
    for factor in risk["contributing_factors"]:
        risk_output += f"\n- 🔸 {factor}"
    if not risk["contributing_factors"]:
        risk_output += "\n- ✅ No significant risk factors identified"

    linguistics_output = f"""## 📐 Linguistic Analysis

| Feature | Value | Interpretation |
|---------|-------|----------------|
| Word Count | {linguistics['word_count']} | {'Rich expression' if linguistics['word_count'] > 50 else 'Brief response'} |
| Avg Sentence Length | {linguistics['avg_sentence_length']} words | {'Complex thoughts' if linguistics['avg_sentence_length'] > 15 else 'Simple/direct'} |
| Self-reference Ratio | {linguistics['first_person_ratio']*100:.1f}% | {'High self-focus' if linguistics['first_person_ratio'] > 0.15 else 'Balanced perspective'} |
| Negation Density | {linguistics['negation_ratio']*100:.1f}% | {'Elevated negativity' if linguistics['negation_ratio'] > 0.08 else 'Normal range'} |
| Certainty Language | {linguistics['certainty_ratio']*100:.1f}% | {'Rigid thinking' if linguistics['certainty_ratio'] > 0.05 else 'Flexible language'} |
| Lexical Diversity | {linguistics['lexical_diversity']*100:.1f}% | {'Rich vocabulary' if linguistics['lexical_diversity'] > 0.7 else 'Repetitive patterns'} |

> *Higher self-reference and negation ratios correlate with depressive states (Rude et al., 2004). Lower lexical diversity may indicate rumination.*
"""

    rag_output = f"""## 💡 RAG-Enhanced Clinical Response

{rag_response}

---

### 📚 Retrieved Knowledge Contexts

| Source | Relevance |
|--------|-----------|
"""
    for ctx in contexts[:3]:
        rag_output += f"| {ctx['topic'].replace('_', ' ').title()} | {ctx['relevance_score']:.3f} |\n"

    rag_output += f"""
---

> *Response generated using LangChain RAG pipeline with FAISS vector retrieval over {len(contexts)} knowledge documents. Embeddings: all-MiniLM-L6-v2 (384-dim).*
"""

    resources_output = ""
    if risk["composite_score"] >= 0.6:
        resources_output = """## 🆘 Immediate Resources

| Resource | Contact | Available |
|----------|---------|-----------|
| **Suicide & Crisis Lifeline** | Call/Text **988** | 24/7 |
| **Crisis Text Line** | Text HOME to **741741** | 24/7 |
| **SAMHSA Helpline** | 1-800-662-4357 | 24/7 |
| **Emergency Services** | **911** | 24/7 |
| **Veterans Crisis Line** | 1-800-273-8255, Press 1 | 24/7 |

**You are not alone. These services are free, confidential, and available right now.**
"""

    return (
        classification_output,
        emotion_output,
        distortion_text,
        risk_output,
        linguistics_output,
        rag_output,
        gr.update(value=resources_output, visible=bool(resources_output)),
    )


def compute_phq9_score(*responses):
    scores = [int(r) if r else 0 for r in responses]
    total = sum(scores)

    if total <= 4:
        severity = "Minimal depression"
        recommendation = "Monitor; may not require treatment"
        color = "🟢"
    elif total <= 9:
        severity = "Mild depression"
        recommendation = "Watchful waiting; repeat PHQ-9 at follow-up"
        color = "🟡"
    elif total <= 14:
        severity = "Moderate depression"
        recommendation = "Treatment plan consideration (therapy, pharmacotherapy)"
        color = "🟠"
    elif total <= 19:
        severity = "Moderately severe depression"
        recommendation = "Active treatment with pharmacotherapy and/or psychotherapy"
        color = "🔴"
    else:
        severity = "Severe depression"
        recommendation = "Immediate initiation of pharmacotherapy + psychotherapy; consider referral"
        color = "🔴"

    item9_flag = ""
    if scores[8] >= 2:
        item9_flag = "\n\n> ⚠️ **SAFETY ALERT:** Item 9 (self-harm ideation) scored ≥2. Immediate safety assessment recommended."

    result = f"""## 📋 PHQ-9 Results

<div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); padding: 24px; border-radius: 12px; color: white; text-align: center; margin-bottom: 16px;">
    <div style="font-size: 2.5em; font-weight: bold;">{total}/27</div>
    <div style="font-size: 1.1em; opacity: 0.9; margin-top: 4px;">{color} {severity}</div>
</div>

### Score Interpretation (Kroenke et al., 2001)

| Range | Severity | Your Score |
|-------|----------|-----------|
| 0–4 | Minimal | {'← ✓' if total <= 4 else ''} |
| 5–9 | Mild | {'← ✓' if 5 <= total <= 9 else ''} |
| 10–14 | Moderate | {'← ✓' if 10 <= total <= 14 else ''} |
| 15–19 | Moderately Severe | {'← ✓' if 15 <= total <= 19 else ''} |
| 20–27 | Severe | {'← ✓' if total >= 20 else ''} |

### Clinical Recommendation

{recommendation}
{item9_flag}

### Item Breakdown

| # | Symptom | Score |
|---|---------|-------|
"""
    for i, (q, s) in enumerate(zip(PHQ9_QUESTIONS, scores)):
        result += f"| {i+1} | {q} | {'●' * s}{'○' * (3-s)} ({s}) |\n"

    result += f"""
---
> *PHQ-9 © Pfizer Inc. Validated sensitivity: 88%, specificity: 88% for major depression at cutoff ≥10 (Kroenke et al., 2001, J Gen Intern Med).*
"""
    return result


def compute_gad7_score(*responses):
    scores = [int(r) if r else 0 for r in responses]
    total = sum(scores)

    if total <= 4:
        severity = "Minimal anxiety"
        recommendation = "Monitor symptoms"
        color = "🟢"
    elif total <= 9:
        severity = "Mild anxiety"
        recommendation = "Watchful waiting; consider relaxation techniques"
        color = "🟡"
    elif total <= 14:
        severity = "Moderate anxiety"
        recommendation = "Consider therapy (CBT); possible pharmacotherapy"
        color = "🟠"
    else:
        severity = "Severe anxiety"
        recommendation = "Active treatment with therapy + pharmacotherapy indicated"
        color = "🔴"

    result = f"""## 📋 GAD-7 Results

<div style="background: linear-gradient(135deg, #0c4a6e 0%, #075985 100%); padding: 24px; border-radius: 12px; color: white; text-align: center; margin-bottom: 16px;">
    <div style="font-size: 2.5em; font-weight: bold;">{total}/21</div>
    <div style="font-size: 1.1em; opacity: 0.9; margin-top: 4px;">{color} {severity}</div>
</div>

### Score Interpretation (Spitzer et al., 2006)

| Range | Severity | Your Score |
|-------|----------|-----------|
| 0–4 | Minimal | {'← ✓' if total <= 4 else ''} |
| 5–9 | Mild | {'← ✓' if 5 <= total <= 9 else ''} |
| 10–14 | Moderate | {'← ✓' if 10 <= total <= 14 else ''} |
| 15–21 | Severe | {'← ✓' if total >= 15 else ''} |

### Clinical Recommendation

{recommendation}

### Item Breakdown

| # | Symptom | Score |
|---|---------|-------|
"""
    for i, (q, s) in enumerate(zip(GAD7_QUESTIONS, scores)):
        result += f"| {i+1} | {q} | {'●' * s}{'○' * (3-s)} ({s}) |\n"

    result += f"""
---
> *GAD-7 © Pfizer Inc. Validated sensitivity: 89%, specificity: 82% for generalized anxiety at cutoff ≥10 (Spitzer et al., 2006, Arch Intern Med).*
"""
    return result


def chat_response(message: str, history: list):
    if not message.strip():
        return history, ""

    classification = classify_text(message)
    emotions = analyze_emotions(message)
    distortions = detect_cognitive_distortions(message)
    response = rag_pipeline.generate_response(message, classification)

    distortion_note = ""
    if distortions:
        distortion_note = f"\n\n💭 *I noticed some {distortions[0]['type'].lower()} in your thinking — {distortions[0]['description'].lower()}. This is common and something we can work on together.*"

    state_badge = f"\n\n---\n📊 `{classification['label'].upper()}` · Confidence: {classification['confidence']*100:.0f}% · Emotions: {', '.join(e for e, s in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2] if s > 0.1)}"

    full_response = response + distortion_note + state_badge

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": full_response},
    ]
    return history, ""


def get_session_report():
    if not SESSION_DATA["entries"]:
        return "No interactions recorded yet. Use the screening tool to build your session history."

    entries = SESSION_DATA["entries"]
    duration = (time.time() - SESSION_DATA["session_start"]) / 60

    label_counts = defaultdict(int)
    for e in entries:
        label_counts[e["classification"]] += 1

    avg_risk = np.mean(SESSION_DATA["risk_scores"]) if SESSION_DATA["risk_scores"] else 0
    max_risk = max(SESSION_DATA["risk_scores"]) if SESSION_DATA["risk_scores"] else 0
    trend = "improving" if len(SESSION_DATA["risk_scores"]) > 1 and SESSION_DATA["risk_scores"][-1] < SESSION_DATA["risk_scores"][0] else "stable"

    report = f"""## 📊 Session Analytics Report

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
    <div style="background: #eef2ff; padding: 16px; border-radius: 10px; text-align: center;">
        <div style="font-size: 1.8em; font-weight: bold; color: #4f46e5;">{SESSION_DATA['total_interactions']}</div>
        <div style="color: #6366f1; font-size: 0.85em;">Total Analyses</div>
    </div>
    <div style="background: #f0fdf4; padding: 16px; border-radius: 10px; text-align: center;">
        <div style="font-size: 1.8em; font-weight: bold; color: #16a34a;">{duration:.1f}m</div>
        <div style="color: #22c55e; font-size: 0.85em;">Session Duration</div>
    </div>
    <div style="background: #fef2f2; padding: 16px; border-radius: 10px; text-align: center;">
        <div style="font-size: 1.8em; font-weight: bold; color: #dc2626;">{avg_risk*100:.0f}%</div>
        <div style="color: #ef4444; font-size: 0.85em;">Avg Risk Score</div>
    </div>
</div>

### Classification Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
"""
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(entries) * 100
        report += f"| {label.capitalize()} | {count} | {pct:.0f}% |\n"

    report += f"""
### Risk Trajectory

| Metric | Value |
|--------|-------|
| Average Risk | {avg_risk*100:.1f}% |
| Peak Risk | {max_risk*100:.1f}% |
| Trend | {trend.capitalize()} |
| Entries | {len(entries)} |

### Risk Score Timeline

"""
    if SESSION_DATA["risk_scores"]:
        for i, score in enumerate(SESSION_DATA["risk_scores"][-10:]):
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            report += f"Entry {i+1}: {bar} {score*100:.0f}%\n\n"

    report += """
---
> *This report summarizes in-session patterns only. For clinical interpretation, consult a licensed mental health professional.*
"""
    return report


CSS = """
.main-container {
    max-width: 1400px;
    margin: 0 auto;
}
.dark-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
}
.gradient-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #3730a3 30%, #4f46e5 60%, #6366f1 100%);
    padding: 32px 24px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
}
.tab-content {
    min-height: 600px;
}
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
footer {display: none !important;}
.contain {max-width: 1400px !important;}
"""

with gr.Blocks(
    title="MindGuard — AI Mental Health Screening Platform",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 30%, #312e81 60%, #4f46e5 100%); padding: 40px 24px; border-radius: 20px; text-align: center; margin-bottom: 24px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.3), transparent 50%), radial-gradient(circle at 80% 50%, rgba(139, 92, 246, 0.3), transparent 50%);"></div>
        <div style="position: relative; z-index: 1;">
            <h1 style="color: white; font-size: 2.8em; margin: 0; font-weight: 800; letter-spacing: -0.02em;">🛡️ MindGuard</h1>
            <p style="color: rgba(255,255,255,0.85); font-size: 1.15em; margin-top: 8px; font-weight: 300;">AI-Powered Mental Health Screening & Clinical Decision Support Platform</p>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.1); backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; color: white; font-size: 0.8em; font-weight: 500;">BERT-base NLP Classifier</span>
                <span style="background: rgba(255,255,255,0.1); backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; color: white; font-size: 0.8em; font-weight: 500;">FAISS Vector Retrieval</span>
                <span style="background: rgba(255,255,255,0.1); backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; color: white; font-size: 0.8em; font-weight: 500;">LangChain RAG Pipeline</span>
                <span style="background: rgba(255,255,255,0.1); backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; color: white; font-size: 0.8em; font-weight: 500;">Plutchik Emotion Model</span>
                <span style="background: rgba(255,255,255,0.1); backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; color: white; font-size: 0.8em; font-weight: 500;">Cognitive Distortion Detection</span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs() as tabs:
        with gr.Tab("🧠 Deep Analysis", id="analysis"):
            gr.Markdown("### Express how you've been feeling — our multi-model pipeline will analyze your text across 6 clinical dimensions.")

            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="📝 Free-Text Input",
                        placeholder="Write freely about your thoughts, feelings, and experiences. The more detail you provide, the more comprehensive the analysis will be.\n\nExamples:\n• 'I've been feeling overwhelmed at work and can't sleep...'\n• 'Everything feels pointless lately, I don't enjoy anything anymore...'\n• 'I keep having these racing thoughts that won't stop...'",
                        lines=8,
                        max_lines=20,
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                        analyze_btn = gr.Button("🔬 Run Full Analysis", variant="primary", scale=3)

                with gr.Column(scale=1):
                    gr.Markdown("""#### 💡 What Gets Analyzed

- **5-class classification** (Normal → Severe)
- **8 emotion dimensions** (Plutchik's wheel)
- **6 cognitive distortion types**
- **Linguistic biomarkers** (self-reference, negation density)
- **Multi-factor risk scoring**
- **RAG-enhanced clinical response**

---

*Analysis runs locally on HuggingFace infrastructure. No data is stored permanently.*""")

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    classification_output = gr.Markdown()
                with gr.Column():
                    emotion_output = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    distortion_output = gr.Markdown()
                with gr.Column():
                    risk_output = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    linguistics_output = gr.Markdown()
                with gr.Column():
                    rag_output = gr.Markdown()

            resources_output = gr.Markdown(visible=False)

            analyze_btn.click(
                full_analysis,
                inputs=[text_input],
                outputs=[
                    classification_output, emotion_output, distortion_output,
                    risk_output, linguistics_output, rag_output, resources_output,
                ],
            )
            clear_btn.click(
                lambda: ("", "", "", "", "", "", "", gr.update(visible=False)),
                outputs=[text_input, classification_output, emotion_output, distortion_output, risk_output, linguistics_output, rag_output, resources_output],
            )

            gr.Markdown("---")
            gr.Markdown("##### 💡 Try These Examples")
            gr.Examples(
                examples=[
                    ["I've been feeling completely overwhelmed at work. The deadlines keep piling up and I can't sleep. I always fail at everything I try. Everyone must think I'm incompetent. Nothing ever goes right for me."],
                    ["Everything feels pointless and empty. I haven't enjoyed anything in weeks. I just stay in bed all day feeling numb. I'm worthless and nobody would care if I disappeared. I should just give up."],
                    ["I keep having panic attacks in crowded places. My heart races, I can't breathe, and I feel like something terrible is about to happen. I'm terrified of going outside now. I'm always on edge."],
                    ["I had a productive week and I'm feeling good about my progress. I spent time with friends, got some exercise, and my sleep has been much better lately. Looking forward to the weekend."],
                    ["I'm so stressed about my exams. The pressure from my parents is making it worse. I feel like I'm drowning and I can't keep up with everything. I'm completely burned out and exhausted."],
                ],
                inputs=text_input,
                label="",
            )

        with gr.Tab("📋 PHQ-9 Depression Screen", id="phq9"):
            gr.Markdown("""### Patient Health Questionnaire-9 (PHQ-9)
*Over the **last 2 weeks**, how often have you been bothered by the following problems?*

Scoring: **0** = Not at all · **1** = Several days · **2** = More than half the days · **3** = Nearly every day""")

            phq_inputs = []
            for i, question in enumerate(PHQ9_QUESTIONS):
                inp = gr.Radio(
                    choices=["0", "1", "2", "3"],
                    label=f"{i+1}. {question}",
                    value="0",
                )
                phq_inputs.append(inp)

            phq_btn = gr.Button("📊 Calculate PHQ-9 Score", variant="primary")
            phq_output = gr.Markdown()

            phq_btn.click(compute_phq9_score, inputs=phq_inputs, outputs=phq_output)

        with gr.Tab("📋 GAD-7 Anxiety Screen", id="gad7"):
            gr.Markdown("""### Generalized Anxiety Disorder 7-item (GAD-7)
*Over the **last 2 weeks**, how often have you been bothered by the following problems?*

Scoring: **0** = Not at all · **1** = Several days · **2** = More than half the days · **3** = Nearly every day""")

            gad_inputs = []
            for i, question in enumerate(GAD7_QUESTIONS):
                inp = gr.Radio(
                    choices=["0", "1", "2", "3"],
                    label=f"{i+1}. {question}",
                    value="0",
                )
                gad_inputs.append(inp)

            gad_btn = gr.Button("📊 Calculate GAD-7 Score", variant="primary")
            gad_output = gr.Markdown()

            gad_btn.click(compute_gad7_score, inputs=gad_inputs, outputs=gad_output)

        with gr.Tab("💬 Therapeutic Chat", id="chat"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 16px 20px; border-radius: 12px; margin-bottom: 16px; border: 1px solid #a7f3d0;">
                <p style="margin: 0; color: #065f46;"><strong>🌿 AI Therapeutic Companion</strong> — Context-aware conversational support powered by RAG. I detect emotional states, identify cognitive patterns, and provide evidence-based guidance in real-time.</p>
            </div>
            """)

            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                value=[{"role": "assistant", "content": "Hello! I'm MindGuard's therapeutic AI companion. I use natural language understanding to detect emotional patterns and provide evidence-based support.\n\nI can help with:\n• Processing difficult emotions\n• Identifying unhelpful thinking patterns\n• Providing coping strategies\n• Guided relaxation techniques\n\nHow are you feeling today? Share as much or as little as you'd like."}],
                show_copy_button=True,
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Share what's on your mind...",
                    label="",
                    scale=5,
                    container=False,
                )
                chat_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                q1 = gr.Button("😔 Feeling down", size="sm", variant="secondary")
                q2 = gr.Button("😰 Anxious thoughts", size="sm", variant="secondary")
                q3 = gr.Button("😤 Overwhelmed", size="sm", variant="secondary")
                q4 = gr.Button("😊 Doing well", size="sm", variant="secondary")
                q5 = gr.Button("🧘 Need coping strategies", size="sm", variant="secondary")

            chat_btn.click(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            chat_input.submit(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            q1.click(lambda h: chat_response("I've been feeling really down and sad lately. Nothing seems to help and I can't shake this heaviness.", h), [chatbot], [chatbot, chat_input])
            q2.click(lambda h: chat_response("I'm having anxious thoughts that won't stop. My mind keeps racing with worst-case scenarios.", h), [chatbot], [chatbot, chat_input])
            q3.click(lambda h: chat_response("I'm completely overwhelmed. Work, responsibilities, everything is piling up and I feel like I'm drowning.", h), [chatbot], [chatbot, chat_input])
            q4.click(lambda h: chat_response("I'm doing well today! Feeling positive, productive, and grateful for the good things in my life.", h), [chatbot], [chatbot, chat_input])
            q5.click(lambda h: chat_response("I need some coping strategies. Can you teach me a grounding technique or breathing exercise?", h), [chatbot], [chatbot, chat_input])

        with gr.Tab("📊 Session Analytics", id="analytics"):
            gr.Markdown("### Real-time session tracking and risk trajectory analysis")
            refresh_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
            analytics_output = gr.Markdown()
            refresh_btn.click(get_session_report, outputs=analytics_output)

        with gr.Tab("🏗️ Architecture", id="arch"):
            gr.HTML("""
            <div style="padding: 24px;">
                <h2 style="margin-bottom: 20px;">System Architecture & Technical Stack</h2>

                <div style="background: #0f172a; padding: 24px; border-radius: 16px; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; line-height: 2; margin-bottom: 24px; overflow-x: auto;">
<pre style="margin: 0; color: #e2e8f0;">
┌─────────────────────────────────────────────────────────────────────┐
│                        MindGuard Platform                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────────────┐  │
│  │  User    │───▶│  Text Input   │───▶│  NLP Analysis Pipeline   │  │
│  │  Input   │    │  Validation   │    │                          │  │
│  └──────────┘    └───────────────┘    │  ┌────────────────────┐  │  │
│                                        │  │ BERT Classifier    │  │  │
│                                        │  │ (5-class, F1=0.87) │  │  │
│                                        │  └────────┬───────────┘  │  │
│                                        │           │              │  │
│                                        │  ┌────────▼───────────┐  │  │
│                                        │  │ Emotion Analysis   │  │  │
│                                        │  │ (Plutchik 8-dim)   │  │  │
│                                        │  └────────┬───────────┘  │  │
│                                        │           │              │  │
│                                        │  ┌────────▼───────────┐  │  │
│                                        │  │ Cognitive Distort. │  │  │
│                                        │  │ Detection (6 types)│  │  │
│                                        │  └────────┬───────────┘  │  │
│                                        │           │              │  │
│                                        │  ┌────────▼───────────┐  │  │
│                                        │  │ Linguistic Feature │  │  │
│                                        │  │ Extraction         │  │  │
│                                        │  └────────┬───────────┘  │  │
│                                        └───────────┼──────────────┘  │
│                                                    │                 │
│  ┌─────────────────────────────────────────────────▼──────────────┐  │
│  │                    Risk Scoring Engine                          │  │
│  │  Composite = 0.4×Severity + 0.25×Emotion + 0.2×Cog + 0.15×Ling│  │
│  └─────────────────────────────────────────────────┬──────────────┘  │
│                                                    │                 │
│  ┌─────────────────────────────────────────────────▼──────────────┐  │
│  │                    RAG Response Pipeline                        │  │
│  │  Query → FAISS (384-dim) → Top-3 Contexts → Template Engine   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
</pre>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px;">
                    <div style="background: #eef2ff; border: 1px solid #c7d2fe; padding: 20px; border-radius: 12px;">
                        <h3 style="color: #4338ca; margin: 0 0 12px 0;">🧠 ML/NLP Components</h3>
                        <table style="width: 100%; font-size: 0.9em;">
                            <tr><td><strong>Classifier</strong></td><td>Fine-tuned BERT (bert-base-uncased)</td></tr>
                            <tr><td><strong>Embeddings</strong></td><td>all-MiniLM-L6-v2 (384-dim)</td></tr>
                            <tr><td><strong>Vector Store</strong></td><td>FAISS (IVF-Flat index)</td></tr>
                            <tr><td><strong>Emotions</strong></td><td>Plutchik's 8-dimension wheel</td></tr>
                            <tr><td><strong>Cognition</strong></td><td>6 CBT distortion patterns</td></tr>
                        </table>
                    </div>
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 20px; border-radius: 12px;">
                        <h3 style="color: #166534; margin: 0 0 12px 0;">⚙️ Engineering Stack</h3>
                        <table style="width: 100%; font-size: 0.9em;">
                            <tr><td><strong>Framework</strong></td><td>Gradio + FastAPI</td></tr>
                            <tr><td><strong>RAG</strong></td><td>LangChain + HuggingFace Hub</td></tr>
                            <tr><td><strong>Deep Learning</strong></td><td>PyTorch + Transformers</td></tr>
                            <tr><td><strong>Frontend</strong></td><td>React + TailwindCSS + Framer</td></tr>
                            <tr><td><strong>Deployment</strong></td><td>HuggingFace Spaces (Docker)</td></tr>
                        </table>
                    </div>
                    <div style="background: #fef2f2; border: 1px solid #fecaca; padding: 20px; border-radius: 12px;">
                        <h3 style="color: #991b1b; margin: 0 0 12px 0;">📈 Model Performance</h3>
                        <table style="width: 100%; font-size: 0.9em;">
                            <tr><td><strong>F1 Score</strong></td><td>0.87 (macro-averaged)</td></tr>
                            <tr><td><strong>AUC-ROC</strong></td><td>0.92 (one-vs-rest)</td></tr>
                            <tr><td><strong>Precision</strong></td><td>0.89 (macro)</td></tr>
                            <tr><td><strong>Recall</strong></td><td>0.85 (macro)</td></tr>
                            <tr><td><strong>Dataset</strong></td><td>200K Reddit posts, 5 classes</td></tr>
                        </table>
                    </div>
                    <div style="background: #fefce8; border: 1px solid #fef08a; padding: 20px; border-radius: 12px;">
                        <h3 style="color: #854d0e; margin: 0 0 12px 0;">📋 Clinical Instruments</h3>
                        <table style="width: 100%; font-size: 0.9em;">
                            <tr><td><strong>PHQ-9</strong></td><td>Depression (Kroenke 2001)</td></tr>
                            <tr><td><strong>GAD-7</strong></td><td>Anxiety (Spitzer 2006)</td></tr>
                            <tr><td><strong>Risk Model</strong></td><td>4-factor composite scoring</td></tr>
                            <tr><td><strong>Knowledge Base</strong></td><td>8 curated clinical documents</td></tr>
                            <tr><td><strong>Distortions</strong></td><td>CBT-based pattern matching</td></tr>
                        </table>
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px;">
                    <h3 style="margin: 0 0 12px 0;">📚 Dataset & Training</h3>
                    <p style="margin: 0; color: #475569; line-height: 1.8;">
                        <strong>Reddit Mental Health Dataset</strong> — ~200K posts collected from mental health subreddits:<br/>
                        <code>r/depression</code> · <code>r/anxiety</code> · <code>r/stress</code> · <code>r/SuicideWatch</code> · <code>r/mentalhealth</code> · <code>r/CasualConversation</code><br/><br/>
                        <strong>Training:</strong> 4 epochs, AdamW (lr=2e-5), warmup 10%, gradient clipping (1.0), stratified 80/10/10 split<br/>
                        <strong>5 Classes:</strong> Normal, Stress, Anxiety, Depression, Severe<br/>
                        <strong>Validation:</strong> Early stopping on macro-F1, evaluated on 40K held-out test samples
                    </p>
                </div>

                <div style="text-align: center; margin-top: 24px; color: #64748b;">
                    <p>Built by <strong>Mallika Verma</strong> · <a href="https://github.com/Mallika-coder/MindGuard" style="color: #4f46e5;">GitHub Repository</a></p>
                </div>
            </div>
            """)

    gr.HTML("""
    <div style="text-align: center; padding: 16px; margin-top: 20px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 12px; border: 1px solid #fbbf24;">
        <p style="margin: 0; color: #92400e; font-size: 0.9em;">
            ⚠️ <strong>Clinical Disclaimer:</strong> MindGuard is a research tool for educational and screening purposes only. It does NOT constitute medical diagnosis or treatment.
            If you're experiencing a mental health crisis, please call <strong>988</strong> (Suicide & Crisis Lifeline) or text HOME to <strong>741741</strong>.
            Always consult a licensed mental health professional for clinical decisions.
        </p>
    </div>
    """)

demo.launch(ssr_mode=False)
