"""
MindGuard — AI-Powered Mental Health Screening & Clinical Decision Support Platform
Multi-dimensional NLP analysis: BERT + FAISS + LangChain RAG + Plutchik + CBT
"""

import gradio as gr
import numpy as np
import random
import time
from collections import defaultdict

from backend.app.rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline()

SESSION_DATA = {
    "entries": [],
    "mood_timeline": [],
    "risk_scores": [],
    "session_start": time.time(),
    "total_interactions": 0,
    "journal": [],
}

EMOTION_LEXICON = {
    "anger": ["angry", "furious", "rage", "hate", "irritated", "frustrated", "annoyed", "resentful", "bitter", "hostile", "mad", "pissed", "livid", "outraged"],
    "sadness": ["sad", "crying", "tears", "grief", "mourning", "lonely", "heartbroken", "miserable", "gloomy", "melancholy", "depressed", "down", "unhappy", "sorrowful", "blue"],
    "fear": ["afraid", "scared", "terrified", "frightened", "panic", "phobia", "dread", "horror", "alarmed", "petrified", "nervous", "anxious", "worried", "uneasy", "tense", "apprehensive"],
    "joy": ["happy", "excited", "grateful", "proud", "content", "cheerful", "elated", "thrilled", "delighted", "optimistic", "good", "great", "wonderful", "amazing", "love", "enjoy", "fun"],
    "disgust": ["disgusted", "repulsed", "revolted", "sick", "nauseated", "appalled", "loathing", "gross", "awful"],
    "surprise": ["shocked", "amazed", "astonished", "startled", "unexpected", "stunned", "wow", "surprised"],
    "trust": ["believe", "faith", "confident", "secure", "reliable", "honest", "loyal", "trust", "safe", "comfortable"],
    "anticipation": ["expecting", "hopeful", "looking forward", "planning", "preparing", "eager", "curious", "interested", "future"],
}

COGNITIVE_DISTORTIONS = {
    "all_or_nothing": {
        "patterns": ["always", "never", "everyone", "nobody", "everything", "nothing", "completely", "totally", "every time"],
        "label": "All-or-Nothing Thinking",
        "description": "Seeing things in black-and-white categories with no middle ground",
        "reframe": "Try replacing absolute words with more balanced ones: 'sometimes', 'in this situation', 'this specific instance'",
    },
    "catastrophizing": {
        "patterns": ["worst", "terrible", "horrible", "disaster", "catastrophe", "end of the world", "ruined", "destroyed", "can't handle"],
        "label": "Catastrophizing",
        "description": "Expecting the worst-case scenario and magnifying negative outcomes",
        "reframe": "Ask yourself: What's the most likely outcome? What would I tell a friend in this situation?",
    },
    "mind_reading": {
        "patterns": ["they think", "everyone thinks", "people think", "they must", "probably thinks", "judges me", "looking at me", "laughing at"],
        "label": "Mind Reading",
        "description": "Assuming you know what others are thinking without evidence",
        "reframe": "Remind yourself: I cannot read minds. What evidence do I actually have for this belief?",
    },
    "should_statements": {
        "patterns": ["i should", "i must", "i have to", "i need to", "ought to", "supposed to", "have to be"],
        "label": "Should Statements",
        "description": "Placing rigid, unrealistic demands on yourself",
        "reframe": "Replace 'should' with 'I would prefer' or 'It would be nice if' — give yourself permission to be human",
    },
    "overgeneralization": {
        "patterns": ["always happens", "every time", "never works", "nothing ever", "this always", "typical", "story of my life"],
        "label": "Overgeneralization",
        "description": "Drawing broad negative conclusions from a single event",
        "reframe": "Is this really ALWAYS the case? Can you think of even one exception? Focus on this specific situation.",
    },
    "personalization": {
        "patterns": ["my fault", "because of me", "i caused", "blame myself", "i'm the reason", "if only i"],
        "label": "Personalization",
        "description": "Taking excessive responsibility for events outside your control",
        "reframe": "What percentage of this outcome was actually within your control? What other factors contributed?",
    },
}

SEVERE_KEYWORDS = ["suicide", "kill myself", "end it all", "no reason to live", "self-harm", "want to die", "better off dead", "not worth living"]
DEPRESSION_KEYWORDS = ["hopeless", "worthless", "empty", "numb", "can't go on", "no energy", "don't care anymore", "failure", "burden", "giving up", "pointless"]
ANXIETY_KEYWORDS = ["panic", "worried", "racing thoughts", "can't breathe", "nervous", "terrified", "dread", "restless", "on edge", "impending doom", "anxious", "fear", "scared"]
STRESS_KEYWORDS = ["overwhelmed", "pressure", "exhausted", "too much", "burned out", "can't keep up", "falling behind", "drowning", "stressed", "overloaded"]

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
    words = text_lower.split()
    scores = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
            for word in words:
                if word.startswith(kw[:4]) and len(kw) > 3:
                    score += 0.5
        scores[emotion] = min(score / 4.0, 1.0)
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 3) for k, v in scores.items()}
    else:
        scores = {k: round(1.0 / 8, 3) for k in scores}
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
                    "reframe": distortion["reframe"],
                    "trigger": pattern,
                })
                break
    return detected


def compute_linguistic_features(text: str) -> dict:
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    first_person = sum(1 for w in words if w.lower() in ["i", "me", "my", "myself", "mine", "i'm", "i've", "i'll", "i'd"])
    negations = sum(1 for w in words if w.lower() in ["not", "no", "never", "don't", "can't", "won't", "isn't", "aren't", "doesn't", "didn't", "nothing", "nowhere", "nobody", "neither", "nor"])
    certainty_words = sum(1 for w in words if w.lower() in ["always", "never", "definitely", "certainly", "absolutely", "completely", "totally", "every", "all"])
    hedging = sum(1 for w in words if w.lower() in ["maybe", "perhaps", "might", "could", "sometimes", "possibly", "probably", "somewhat"])
    temporal_focus_past = sum(1 for w in words if w.lower() in ["was", "were", "had", "used", "before", "yesterday", "ago", "past"])
    temporal_focus_future = sum(1 for w in words if w.lower() in ["will", "going", "tomorrow", "future", "plan", "hope", "soon", "later"])

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "first_person_ratio": round(first_person / max(len(words), 1), 3),
        "negation_ratio": round(negations / max(len(words), 1), 3),
        "certainty_ratio": round(certainty_words / max(len(words), 1), 3),
        "hedging_ratio": round(hedging / max(len(words), 1), 3),
        "lexical_diversity": round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
        "temporal_past": round(temporal_focus_past / max(len(words), 1), 3),
        "temporal_future": round(temporal_focus_future / max(len(words), 1), 3),
    }


def classify_text(text: str) -> dict:
    text_lower = text.lower()
    scores = {"normal": 0.1, "stress": 0.0, "anxiety": 0.0, "depression": 0.0, "severe": 0.0}

    for kw in SEVERE_KEYWORDS:
        if kw in text_lower:
            scores["severe"] += 0.35
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
    emotion_risk = (emotions.get("sadness", 0) * 0.3 + emotions.get("fear", 0) * 0.25 +
                    emotions.get("anger", 0) * 0.15 - emotions.get("joy", 0) * 0.2)
    distortion_risk = min(len(distortions) * 0.12, 0.35)
    linguistic_risk = (linguistics["negation_ratio"] * 0.4 +
                       linguistics["first_person_ratio"] * 0.2 +
                       (1 - linguistics["lexical_diversity"]) * 0.15 +
                       linguistics["certainty_ratio"] * 0.25)

    composite_risk = np.clip(
        base_risk * 0.4 + emotion_risk * 0.25 + distortion_risk * 0.2 + linguistic_risk * 0.15, 0, 1
    )

    factors = []
    if base_risk >= 0.5:
        factors.append(f"High-risk language patterns (severity: {base_risk:.0%})")
    if emotions.get("sadness", 0) > 0.25:
        factors.append("Elevated sadness indicators in emotional profile")
    if emotions.get("fear", 0) > 0.25:
        factors.append("Elevated fear/anxiety markers detected")
    if len(distortions) >= 2:
        factors.append(f"{len(distortions)} cognitive distortions identified")
    if linguistics["negation_ratio"] > 0.08:
        factors.append("High negative language density")
    if linguistics["first_person_ratio"] > 0.15:
        factors.append("Elevated self-focused language (rumination indicator)")

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
            "Low" if composite_risk >= 0.2 else "Minimal"
        ),
        "contributing_factors": factors,
    }


def build_visual_result(text, classification, emotions, distortions, linguistics, risk, rag_response, contexts):
    label = classification["label"]
    confidence = classification["confidence"]
    severity = classification["severity_score"]

    label_colors = {
        "normal": ("#10b981", "#d1fae5", "#065f46"),
        "stress": ("#f59e0b", "#fef3c7", "#92400e"),
        "anxiety": ("#f97316", "#ffedd5", "#9a3412"),
        "depression": ("#8b5cf6", "#ede9fe", "#5b21b6"),
        "severe": ("#ef4444", "#fee2e2", "#991b1b"),
    }
    color, bg_color, text_color = label_colors.get(label, ("#6366f1", "#eef2ff", "#3730a3"))

    risk_level = risk["risk_level"]
    risk_score = risk["composite_score"]

    emotion_bars = ""
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    for emo, score in sorted_emotions:
        emo_colors = {
            "anger": "#ef4444", "sadness": "#3b82f6", "fear": "#8b5cf6",
            "joy": "#10b981", "disgust": "#84cc16", "surprise": "#f59e0b",
            "trust": "#06b6d4", "anticipation": "#ec4899",
        }
        emo_color = emo_colors.get(emo, "#6366f1")
        pct = score * 100
        emotion_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
            <span style="width: 90px; font-size: 0.8em; color: #64748b; text-transform: capitalize;">{emo}</span>
            <div style="flex: 1; height: 20px; background: #f1f5f9; border-radius: 10px; overflow: hidden; position: relative;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {emo_color}aa, {emo_color}); border-radius: 10px; transition: width 0.8s ease;"></div>
            </div>
            <span style="width: 40px; font-size: 0.75em; color: #64748b; text-align: right;">{pct:.0f}%</span>
        </div>"""

    prob_bars = ""
    for cat, prob in classification["probabilities"].items():
        cat_colors = {"normal": "#10b981", "stress": "#f59e0b", "anxiety": "#f97316", "depression": "#8b5cf6", "severe": "#ef4444"}
        pct = prob * 100
        prob_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0;">
            <span style="width: 80px; font-size: 0.8em; color: #475569; text-transform: capitalize; font-weight: 500;">{cat}</span>
            <div style="flex: 1; height: 24px; background: #f1f5f9; border-radius: 12px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {cat_colors.get(cat, '#6366f1')}99, {cat_colors.get(cat, '#6366f1')}); border-radius: 12px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px;">
                    <span style="color: white; font-size: 0.7em; font-weight: 600;">{pct:.1f}%</span>
                </div>
            </div>
        </div>"""

    distortion_cards = ""
    if distortions:
        for d in distortions:
            distortion_cards += f"""
            <div style="background: linear-gradient(135deg, #fef9c3, #fef3c7); border: 1px solid #fbbf24; border-radius: 12px; padding: 16px; margin: 8px 0;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.2em;">⚠️</span>
                    <strong style="color: #92400e;">{d['type']}</strong>
                </div>
                <p style="color: #78716c; font-size: 0.85em; margin: 4px 0;">{d['description']}</p>
                <p style="color: #a16207; font-size: 0.8em; margin: 4px 0;">Triggered by: "<em>{d['trigger']}</em>"</p>
                <div style="background: #fff; border-radius: 8px; padding: 10px; margin-top: 8px; border: 1px solid #e5e7eb;">
                    <p style="color: #059669; font-size: 0.8em; margin: 0;">💡 <strong>Reframe:</strong> {d['reframe']}</p>
                </div>
            </div>"""
    else:
        distortion_cards = """
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px; text-align: center;">
            <span style="font-size: 1.5em;">✅</span>
            <p style="color: #166534; margin: 8px 0 0 0; font-weight: 500;">No cognitive distortions detected</p>
            <p style="color: #4ade80; font-size: 0.8em; margin: 4px 0 0 0;">Your thinking patterns appear balanced and rational</p>
        </div>"""

    risk_ring_pct = risk_score * 100
    risk_colors = {"Critical": "#ef4444", "High": "#f97316", "Moderate": "#f59e0b", "Low": "#10b981", "Minimal": "#06b6d4"}
    risk_color = risk_colors.get(risk_level, "#6366f1")
    circumference = 2 * 3.14159 * 45
    offset = circumference - (risk_score * circumference)

    factors_html = ""
    for f in risk["contributing_factors"]:
        factors_html += f'<div style="display: flex; align-items: center; gap: 6px; margin: 4px 0;"><span style="color: {risk_color};">●</span><span style="font-size: 0.8em; color: #475569;">{f}</span></div>'
    if not risk["contributing_factors"]:
        factors_html = '<div style="color: #10b981; font-size: 0.85em;">✓ No significant risk factors identified</div>'

    component_bars = ""
    for comp_name, comp_val in risk["components"].items():
        nice_name = comp_name.replace("_", " ").title()
        pct = comp_val * 100
        component_bars += f"""
        <div style="margin: 6px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.75em; color: #64748b; margin-bottom: 2px;">
                <span>{nice_name}</span><span>{pct:.0f}%</span>
            </div>
            <div style="height: 6px; background: #f1f5f9; border-radius: 3px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {risk_color}88, {risk_color}); border-radius: 3px;"></div>
            </div>
        </div>"""

    ling_features = [
        ("Word Count", str(linguistics["word_count"]), "📝"),
        ("Sentences", str(linguistics["sentence_count"]), "📄"),
        ("Avg Length", f"{linguistics['avg_sentence_length']} words", "📏"),
        ("Self-focus", f"{linguistics['first_person_ratio']*100:.1f}%", "👤"),
        ("Negativity", f"{linguistics['negation_ratio']*100:.1f}%", "➖"),
        ("Certainty", f"{linguistics['certainty_ratio']*100:.1f}%", "❗"),
        ("Diversity", f"{linguistics['lexical_diversity']*100:.0f}%", "🔤"),
        ("Past Focus", f"{linguistics['temporal_past']*100:.1f}%", "⏪"),
        ("Future Focus", f"{linguistics['temporal_future']*100:.1f}%", "⏩"),
    ]
    ling_grid = ""
    for feat_name, feat_val, feat_icon in ling_features:
        ling_grid += f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="font-size: 1.2em;">{feat_icon}</div>
            <div style="font-size: 1.1em; font-weight: 700; color: #1e293b; margin: 4px 0;">{feat_val}</div>
            <div style="font-size: 0.7em; color: #94a3b8;">{feat_name}</div>
        </div>"""

    context_html = ""
    for i, ctx in enumerate(contexts[:3]):
        relevance = max(0, 1 - abs(ctx["relevance_score"]))
        rel_pct = relevance * 100
        context_html += f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin: 6px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.85em; font-weight: 600; color: #334155;">{ctx['topic'].replace('_', ' ').title()}</span>
                <span style="font-size: 0.7em; background: #eef2ff; color: #4f46e5; padding: 2px 8px; border-radius: 10px;">Match: {rel_pct:.0f}%</span>
            </div>
            <p style="font-size: 0.75em; color: #64748b; margin: 0; line-height: 1.4;">{ctx['content'][:120]}...</p>
        </div>"""

    result_html = f"""
    <div style="font-family: 'Inter', -apple-system, sans-serif;">

        <!-- Main Classification Card -->
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); border-radius: 20px; padding: 32px; margin-bottom: 20px; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50%; right: -20%; width: 300px; height: 300px; background: radial-gradient(circle, {color}33, transparent); border-radius: 50%;"></div>
            <div style="position: relative; z-index: 1;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
                    <div>
                        <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Detected State</div>
                        <div style="font-size: 2em; font-weight: 800; color: {color}; text-transform: uppercase;">{label}</div>
                        <div style="color: #cbd5e1; font-size: 0.9em; margin-top: 4px;">Confidence: {confidence*100:.1f}% · Risk: {risk_level}</div>
                    </div>
                    <div style="text-align: center;">
                        <svg width="100" height="100" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" stroke-width="8"/>
                            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8"
                                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                                    stroke-linecap="round" transform="rotate(-90 50 50)"/>
                            <text x="50" y="50" text-anchor="middle" dy="0.35em" fill="white" font-size="18" font-weight="bold">{risk_ring_pct:.0f}%</text>
                        </svg>
                        <div style="color: #94a3b8; font-size: 0.7em; margin-top: 4px;">Risk Score</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Two Column Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">

            <!-- Classification Probabilities -->
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                <h3 style="margin: 0 0 16px 0; font-size: 1em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                    <span style="background: #eef2ff; padding: 4px 8px; border-radius: 6px;">🧠</span> Classification Breakdown
                </h3>
                {prob_bars}
            </div>

            <!-- Emotion Profile -->
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                <h3 style="margin: 0 0 16px 0; font-size: 1em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                    <span style="background: #fce7f3; padding: 4px 8px; border-radius: 6px;">🎭</span> Emotional Profile
                </h3>
                {emotion_bars}
            </div>
        </div>

        <!-- Risk Assessment -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 16px 0; font-size: 1em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="background: #fef2f2; padding: 4px 8px; border-radius: 6px;">⚡</span> Multi-Factor Risk Assessment
            </h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <div style="font-size: 0.8em; color: #64748b; margin-bottom: 8px; font-weight: 500;">Risk Components</div>
                    {component_bars}
                </div>
                <div>
                    <div style="font-size: 0.8em; color: #64748b; margin-bottom: 8px; font-weight: 500;">Contributing Factors</div>
                    {factors_html}
                </div>
            </div>
        </div>

        <!-- Cognitive Distortions -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 16px 0; font-size: 1em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="background: #fef9c3; padding: 4px 8px; border-radius: 6px;">🔍</span> Cognitive Distortion Analysis
            </h3>
            {distortion_cards}
        </div>

        <!-- Linguistic Features -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 16px 0; font-size: 1em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="background: #ecfdf5; padding: 4px 8px; border-radius: 6px;">📐</span> Linguistic Biomarkers
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 8px;">
                {ling_grid}
            </div>
            <div style="margin-top: 12px; padding: 10px; background: #f8fafc; border-radius: 8px; font-size: 0.75em; color: #64748b; line-height: 1.5;">
                📚 <em>Research basis: Elevated self-reference correlates with depression (Rude et al., 2004). High negation density indicates negative affect. Low lexical diversity suggests rumination (Pennebaker, 2011).</em>
            </div>
        </div>

        <!-- RAG Response -->
        <div style="background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); border: 1px solid #c7d2fe; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 12px 0; font-size: 1em; color: #3730a3; display: flex; align-items: center; gap: 8px;">
                <span style="background: #c7d2fe; padding: 4px 8px; border-radius: 6px;">💡</span> AI-Generated Clinical Response
            </h3>
            <p style="color: #312e81; line-height: 1.7; font-size: 0.9em; margin: 0;">{rag_response}</p>
            <div style="margin-top: 16px;">
                <div style="font-size: 0.75em; color: #6366f1; font-weight: 500; margin-bottom: 8px;">Retrieved Knowledge Contexts (FAISS Top-3)</div>
                {context_html}
            </div>
        </div>
    </div>
    """
    return result_html


def full_analysis(text: str):
    if not text or len(text.strip()) < 10:
        return "<p style='color: #f59e0b; text-align: center; padding: 40px;'>⚠️ Please write at least 10 characters for a comprehensive analysis.</p>"

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
        "emotions": emotions,
    })
    SESSION_DATA["mood_timeline"].append(classification["severity_score"])
    SESSION_DATA["risk_scores"].append(risk["composite_score"])
    SESSION_DATA["total_interactions"] += 1

    return build_visual_result(text, classification, emotions, distortions, linguistics, risk, rag_response, contexts)


def generate_smart_response(message: str, classification: dict, emotions: dict, distortions: list) -> str:
    label = classification["label"]
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
    top_emotion = top_emotions[0][0] if top_emotions else "neutral"

    rag_response = rag_pipeline.generate_response(message, classification)

    personalized_openings = {
        "sadness": [
            "I can feel the weight of what you're carrying right now.",
            "What you're going through sounds truly heavy, and I want you to know I hear you.",
            "The sadness you're describing is real and valid.",
        ],
        "fear": [
            "I understand that sense of unease you're describing.",
            "It sounds like your mind is working overtime trying to protect you.",
            "The worry you're feeling makes sense given what you've shared.",
        ],
        "anger": [
            "That frustration is completely understandable.",
            "It makes sense that you'd feel that way given the situation.",
            "Your anger is telling you something important about your boundaries.",
        ],
        "joy": [
            "It's wonderful to hear that positive energy in your words!",
            "I'm genuinely glad things are going well for you.",
            "That's the kind of moment worth savoring.",
        ],
        "trust": [
            "Thank you for feeling comfortable enough to share that with me.",
            "I appreciate your openness — it's a sign of real self-awareness.",
        ],
        "anticipation": [
            "It sounds like you're processing a lot about what comes next.",
            "That forward-looking perspective shows real engagement with your life.",
        ],
    }

    opening = random.choice(personalized_openings.get(top_emotion, [
        "Thank you for sharing that with me.",
        "I hear what you're saying, and it matters.",
    ]))

    distortion_insight = ""
    if distortions:
        d = distortions[0]
        distortion_insight = f"\n\n🔍 **Something I noticed:** Your words contain a pattern called *{d['type']}* — {d['description'].lower()}. {d['reframe']}"

    coping_strategies = {
        "anxiety": "\n\n🧘 **Try this now:** 4-7-8 breathing — breathe in for 4 counts, hold for 7, exhale slowly for 8. Repeat 3 times. This activates your parasympathetic nervous system.",
        "depression": "\n\n🌱 **One small step:** Can you identify one tiny thing within reach right now that might bring even 1% comfort? A warm drink, a window with natural light, a song you used to love?",
        "stress": "\n\n📋 **Quick reset:** Write down the 3 most pressing things on your mind. Circle the ONE you can actually control right now. Let the others wait.",
        "severe": "\n\n🆘 **You matter:** Please reach out to 988 (Suicide & Crisis Lifeline) or text HOME to 741741. A real person is waiting to help. You don't have to face this alone.",
        "normal": "\n\n✨ **Keep it up:** What you're doing is working. Consider noting what's going well — gratitude journaling builds resilience for harder days.",
    }
    strategy = coping_strategies.get(label, "")

    response = f"{opening}\n\n{rag_response}{distortion_insight}{strategy}"
    return response


def chat_response(message: str, history: list):
    if not message.strip():
        return history, ""

    classification = classify_text(message)
    emotions = analyze_emotions(message)
    distortions = detect_cognitive_distortions(message)
    response = generate_smart_response(message, classification, emotions, distortions)

    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    emotion_tags = " · ".join([f"{e[0].capitalize()} {e[1]*100:.0f}%" for e in top_emotions if e[1] > 0.05])
    state_badge = f"\n\n---\n📊 *{classification['label'].upper()} ({classification['confidence']*100:.0f}% conf)* | Emotions: {emotion_tags}"

    full_response = response + state_badge

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": full_response},
    ]
    return history, ""


def thought_reframe(thought: str):
    if not thought or len(thought.strip()) < 5:
        return "<p style='color: #f59e0b; text-align: center;'>Please enter a thought to analyze.</p>"

    distortions = detect_cognitive_distortions(thought)
    emotions = analyze_emotions(thought)
    top_emotion = max(emotions.items(), key=lambda x: x[1])

    if not distortions:
        return f"""
        <div style="font-family: 'Inter', sans-serif;">
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-radius: 16px; padding: 24px; text-align: center;">
                <span style="font-size: 2em;">✅</span>
                <h3 style="color: #065f46; margin: 12px 0 8px 0;">This thought appears balanced!</h3>
                <p style="color: #047857; font-size: 0.9em;">No harmful thinking patterns detected. Your thought seems rational and proportionate.</p>
                <p style="color: #10b981; font-size: 0.8em; margin-top: 12px;">Primary emotion detected: <strong>{top_emotion[0].capitalize()}</strong> ({top_emotion[1]*100:.0f}%)</p>
            </div>
        </div>
        """

    cards = ""
    for i, d in enumerate(distortions):
        cards += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin: 12px 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="background: #fef3c7; padding: 6px 10px; border-radius: 8px; font-size: 1.1em;">⚠️</span>
                <div>
                    <strong style="color: #92400e;">{d['type']}</strong>
                    <p style="color: #a3a3a3; font-size: 0.8em; margin: 2px 0 0 0;">Triggered by: "{d['trigger']}"</p>
                </div>
            </div>
            <p style="color: #57534e; font-size: 0.85em; margin: 8px 0; line-height: 1.5;">{d['description']}</p>
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-radius: 10px; padding: 14px; margin-top: 12px;">
                <p style="color: #065f46; font-size: 0.85em; margin: 0; line-height: 1.5;">
                    <strong>💡 Healthier Perspective:</strong><br/>{d['reframe']}
                </p>
            </div>
        </div>"""

    reframed_thought = thought
    for d in distortions:
        trigger = d["trigger"]
        if trigger == "always":
            reframed_thought = reframed_thought.replace("always", "sometimes")
        elif trigger == "never":
            reframed_thought = reframed_thought.replace("never", "not always")
        elif trigger == "everyone":
            reframed_thought = reframed_thought.replace("everyone", "some people")
        elif trigger == "nothing":
            reframed_thought = reframed_thought.replace("nothing", "not everything")
        elif trigger in ["i should", "i must", "i have to"]:
            reframed_thought = reframed_thought.replace(trigger, "I would prefer to")

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
        <div style="background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <div style="font-size: 0.75em; color: #92400e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Original Thought</div>
            <p style="color: #78350f; font-size: 1em; font-style: italic; margin: 0;">"{thought}"</p>
        </div>

        <div style="font-size: 0.8em; color: #64748b; margin: 12px 0; font-weight: 500;">{len(distortions)} Distortion{'s' if len(distortions) > 1 else ''} Detected:</div>
        {cards}

        <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-radius: 16px; padding: 20px; margin-top: 16px;">
            <div style="font-size: 0.75em; color: #065f46; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">🔄 Reframed Thought</div>
            <p style="color: #047857; font-size: 1em; font-weight: 500; margin: 0;">"{reframed_thought}"</p>
        </div>

        <div style="background: #f8fafc; border-radius: 12px; padding: 16px; margin-top: 12px; font-size: 0.8em; color: #64748b; line-height: 1.6;">
            <strong>📚 About CBT Reframing:</strong> Cognitive Behavioral Therapy teaches us that our thoughts shape our emotions. By identifying and challenging distorted thinking patterns, we can reduce emotional distress. This technique is backed by 50+ years of clinical research (Beck, 1976; Burns, 1980).
        </div>
    </div>
    """


def compute_phq9_score(*responses):
    scores = [int(r) if r else 0 for r in responses]
    total = sum(scores)

    if total <= 4:
        severity, color, rec = "Minimal", "#10b981", "Monitor; may not require treatment"
    elif total <= 9:
        severity, color, rec = "Mild", "#f59e0b", "Watchful waiting; repeat PHQ-9 at follow-up"
    elif total <= 14:
        severity, color, rec = "Moderate", "#f97316", "Consider therapy (CBT) and/or pharmacotherapy"
    elif total <= 19:
        severity, color, rec = "Moderately Severe", "#ef4444", "Active treatment with pharmacotherapy + psychotherapy"
    else:
        severity, color, rec = "Severe", "#dc2626", "Immediate treatment initiation; consider specialist referral"

    item_bars = ""
    for i, (q, s) in enumerate(zip(PHQ9_QUESTIONS, scores)):
        pct = s / 3 * 100
        item_color = "#10b981" if s == 0 else "#f59e0b" if s == 1 else "#f97316" if s == 2 else "#ef4444"
        item_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 6px 0; padding: 8px; background: #f8fafc; border-radius: 8px;">
            <span style="width: 20px; font-size: 0.75em; color: #94a3b8; font-weight: bold;">{i+1}</span>
            <div style="flex: 1; font-size: 0.8em; color: #475569;">{q}</div>
            <div style="width: 60px; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: {item_color}; border-radius: 4px;"></div>
            </div>
            <span style="width: 16px; font-size: 0.75em; color: {item_color}; font-weight: bold; text-align: center;">{s}</span>
        </div>"""

    safety_alert = ""
    if scores[8] >= 2:
        safety_alert = """
        <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 16px; margin: 16px 0;">
            <div style="display: flex; align-items: center; gap: 8px; color: #991b1b; font-weight: bold;">
                <span style="font-size: 1.3em;">🚨</span> SAFETY ALERT
            </div>
            <p style="color: #b91c1c; font-size: 0.85em; margin: 8px 0 0 0;">Item 9 (self-harm ideation) scored ≥2. Immediate safety assessment recommended. Please contact 988 (Suicide & Crisis Lifeline) if you or someone you know is at risk.</p>
        </div>"""

    circumference = 2 * 3.14159 * 45
    score_pct = total / 27
    offset = circumference - (score_pct * circumference)

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
        <div style="background: linear-gradient(135deg, #1e1b4b, #312e81); border-radius: 20px; padding: 32px; text-align: center; margin-bottom: 20px;">
            <svg width="120" height="120" viewBox="0 0 100 100" style="margin-bottom: 12px;">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#4c1d95" stroke-width="10"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="10"
                        stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                        stroke-linecap="round" transform="rotate(-90 50 50)"/>
                <text x="50" y="45" text-anchor="middle" fill="white" font-size="22" font-weight="bold">{total}</text>
                <text x="50" y="62" text-anchor="middle" fill="#a5b4fc" font-size="10">/27</text>
            </svg>
            <div style="color: {color}; font-size: 1.2em; font-weight: 700;">{severity} Depression</div>
            <div style="color: #a5b4fc; font-size: 0.85em; margin-top: 4px;">{rec}</div>
        </div>
        {safety_alert}
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <h4 style="color: #1e293b; margin: 0 0 12px 0; font-size: 0.9em;">Item Breakdown</h4>
            {item_bars}
        </div>
        <div style="background: #f8fafc; border-radius: 12px; padding: 16px; font-size: 0.75em; color: #64748b; line-height: 1.6;">
            <strong>📚 Reference:</strong> PHQ-9 © Pfizer Inc. Validated sensitivity 88%, specificity 88% for major depression at cutoff ≥10 (Kroenke, Spitzer & Williams, 2001, <em>J Gen Intern Med</em>). Scoring: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day.
        </div>
    </div>
    """


def compute_gad7_score(*responses):
    scores = [int(r) if r else 0 for r in responses]
    total = sum(scores)

    if total <= 4:
        severity, color, rec = "Minimal", "#10b981", "Monitor; routine follow-up"
    elif total <= 9:
        severity, color, rec = "Mild", "#f59e0b", "Watchful waiting; relaxation techniques"
    elif total <= 14:
        severity, color, rec = "Moderate", "#f97316", "Consider CBT; possible pharmacotherapy"
    else:
        severity, color, rec = "Severe", "#ef4444", "Active treatment: therapy + pharmacotherapy"

    item_bars = ""
    for i, (q, s) in enumerate(zip(GAD7_QUESTIONS, scores)):
        pct = s / 3 * 100
        item_color = "#10b981" if s == 0 else "#f59e0b" if s == 1 else "#f97316" if s == 2 else "#ef4444"
        item_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 6px 0; padding: 8px; background: #f8fafc; border-radius: 8px;">
            <span style="width: 20px; font-size: 0.75em; color: #94a3b8; font-weight: bold;">{i+1}</span>
            <div style="flex: 1; font-size: 0.8em; color: #475569;">{q}</div>
            <div style="width: 60px; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: {item_color}; border-radius: 4px;"></div>
            </div>
            <span style="width: 16px; font-size: 0.75em; color: {item_color}; font-weight: bold; text-align: center;">{s}</span>
        </div>"""

    circumference = 2 * 3.14159 * 45
    score_pct = total / 21
    offset = circumference - (score_pct * circumference)

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
        <div style="background: linear-gradient(135deg, #0c4a6e, #075985); border-radius: 20px; padding: 32px; text-align: center; margin-bottom: 20px;">
            <svg width="120" height="120" viewBox="0 0 100 100" style="margin-bottom: 12px;">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#164e63" stroke-width="10"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="10"
                        stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                        stroke-linecap="round" transform="rotate(-90 50 50)"/>
                <text x="50" y="45" text-anchor="middle" fill="white" font-size="22" font-weight="bold">{total}</text>
                <text x="50" y="62" text-anchor="middle" fill="#7dd3fc" font-size="10">/21</text>
            </svg>
            <div style="color: {color}; font-size: 1.2em; font-weight: 700;">{severity} Anxiety</div>
            <div style="color: #7dd3fc; font-size: 0.85em; margin-top: 4px;">{rec}</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
            <h4 style="color: #1e293b; margin: 0 0 12px 0; font-size: 0.9em;">Item Breakdown</h4>
            {item_bars}
        </div>
        <div style="background: #f8fafc; border-radius: 12px; padding: 16px; font-size: 0.75em; color: #64748b; line-height: 1.6;">
            <strong>📚 Reference:</strong> GAD-7 © Pfizer Inc. Validated sensitivity 89%, specificity 82% for generalized anxiety at cutoff ≥10 (Spitzer, Kroenke, Williams & Löwe, 2006, <em>Arch Intern Med</em>). Scoring: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day.
        </div>
    </div>
    """


def add_journal_entry(entry: str, mood: str):
    if not entry or len(entry.strip()) < 5:
        return get_journal_display()

    classification = classify_text(entry)
    emotions = analyze_emotions(entry)

    SESSION_DATA["journal"].append({
        "timestamp": time.time(),
        "text": entry,
        "mood_tag": mood,
        "classification": classification["label"],
        "top_emotion": max(emotions.items(), key=lambda x: x[1])[0],
        "severity": classification["severity_score"],
    })

    return get_journal_display()


def get_journal_display():
    if not SESSION_DATA["journal"]:
        return """
        <div style="font-family: 'Inter', sans-serif; text-align: center; padding: 40px; color: #94a3b8;">
            <span style="font-size: 2em;">📓</span>
            <p style="margin: 12px 0 0 0;">Your journal is empty. Start writing to track your patterns over time.</p>
        </div>
        """

    entries_html = ""
    mood_counts = defaultdict(int)
    emotion_counts = defaultdict(int)

    for entry in reversed(SESSION_DATA["journal"][-10:]):
        mood_counts[entry["classification"]] += 1
        emotion_counts[entry["top_emotion"]] += 1

        mood_colors = {"normal": "#10b981", "stress": "#f59e0b", "anxiety": "#f97316", "depression": "#8b5cf6", "severe": "#ef4444"}
        entry_color = mood_colors.get(entry["classification"], "#6366f1")

        entries_html += f"""
        <div style="background: white; border-left: 4px solid {entry_color}; border-radius: 0 12px 12px 0; padding: 14px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="background: {entry_color}22; color: {entry_color}; font-size: 0.7em; padding: 2px 8px; border-radius: 10px; font-weight: 600;">{entry['classification'].upper()}</span>
                <span style="font-size: 0.7em; color: #94a3b8;">{entry['mood_tag']}</span>
            </div>
            <p style="color: #334155; font-size: 0.85em; margin: 0; line-height: 1.5;">{entry['text'][:150]}{'...' if len(entry['text']) > 150 else ''}</p>
            <div style="font-size: 0.7em; color: #94a3b8; margin-top: 6px;">Primary emotion: {entry['top_emotion']}</div>
        </div>"""

    pattern_html = ""
    if len(SESSION_DATA["journal"]) >= 3:
        most_common_mood = max(mood_counts.items(), key=lambda x: x[1])
        most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        pattern_html = f"""
        <div style="background: linear-gradient(135deg, #eef2ff, #e0e7ff); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <h4 style="color: #3730a3; margin: 0 0 8px 0; font-size: 0.85em;">📊 Pattern Detection</h4>
            <p style="color: #4338ca; font-size: 0.8em; margin: 4px 0;">Most frequent state: <strong>{most_common_mood[0].capitalize()}</strong> ({most_common_mood[1]} entries)</p>
            <p style="color: #4338ca; font-size: 0.8em; margin: 4px 0;">Dominant emotion: <strong>{most_common_emotion[0].capitalize()}</strong> ({most_common_emotion[1]} entries)</p>
            <p style="color: #6366f1; font-size: 0.75em; margin: 8px 0 0 0;">💡 Tracking patterns helps identify triggers and improvement over time.</p>
        </div>"""

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
        {pattern_html}
        <div style="font-size: 0.8em; color: #64748b; margin-bottom: 8px; font-weight: 500;">{len(SESSION_DATA['journal'])} Journal Entries</div>
        {entries_html}
    </div>
    """


def get_session_analytics():
    if not SESSION_DATA["entries"]:
        return """
        <div style="font-family: 'Inter', sans-serif; text-align: center; padding: 60px; color: #94a3b8;">
            <span style="font-size: 3em;">📊</span>
            <p style="margin: 16px 0 0 0; font-size: 1.1em;">No data yet</p>
            <p style="font-size: 0.85em;">Use the Deep Analysis tool to start building your session analytics.</p>
        </div>
        """

    entries = SESSION_DATA["entries"]
    duration = (time.time() - SESSION_DATA["session_start"]) / 60
    avg_risk = np.mean(SESSION_DATA["risk_scores"])
    max_risk = max(SESSION_DATA["risk_scores"])
    min_risk = min(SESSION_DATA["risk_scores"])

    label_counts = defaultdict(int)
    for e in entries:
        label_counts[e["classification"]] += 1

    trend = "↓ Improving" if len(SESSION_DATA["risk_scores"]) > 1 and SESSION_DATA["risk_scores"][-1] < SESSION_DATA["risk_scores"][0] else "→ Stable" if len(SESSION_DATA["risk_scores"]) <= 1 else "↑ Elevated"

    stat_cards = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #eef2ff, #c7d2fe); border-radius: 12px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #4338ca;">{SESSION_DATA['total_interactions']}</div>
            <div style="font-size: 0.75em; color: #6366f1;">Analyses</div>
        </div>
        <div style="background: linear-gradient(135deg, #ecfdf5, #a7f3d0); border-radius: 12px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #065f46;">{duration:.0f}m</div>
            <div style="font-size: 0.75em; color: #10b981;">Duration</div>
        </div>
        <div style="background: linear-gradient(135deg, #fef2f2, #fecaca); border-radius: 12px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #991b1b;">{avg_risk*100:.0f}%</div>
            <div style="font-size: 0.75em; color: #ef4444;">Avg Risk</div>
        </div>
        <div style="background: linear-gradient(135deg, #fefce8, #fef08a); border-radius: 12px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #854d0e;">{trend.split(' ')[0]}</div>
            <div style="font-size: 0.75em; color: #a16207;">{trend.split(' ')[1]}</div>
        </div>
    </div>"""

    timeline_bars = ""
    for i, score in enumerate(SESSION_DATA["risk_scores"][-15:]):
        bar_height = max(score * 100, 5)
        bar_color = "#ef4444" if score >= 0.6 else "#f97316" if score >= 0.4 else "#f59e0b" if score >= 0.2 else "#10b981"
        timeline_bars += f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
            <div style="width: 20px; height: {bar_height}px; background: linear-gradient(to top, {bar_color}88, {bar_color}); border-radius: 4px 4px 0 0;"></div>
            <span style="font-size: 0.6em; color: #94a3b8;">{i+1}</span>
        </div>"""

    dist_bars = ""
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(entries) * 100
        label_color = {"normal": "#10b981", "stress": "#f59e0b", "anxiety": "#f97316", "depression": "#8b5cf6", "severe": "#ef4444"}.get(label, "#6366f1")
        dist_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 6px 0;">
            <span style="width: 80px; font-size: 0.8em; color: #475569; text-transform: capitalize;">{label}</span>
            <div style="flex: 1; height: 20px; background: #f1f5f9; border-radius: 10px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: {label_color}; border-radius: 10px;"></div>
            </div>
            <span style="font-size: 0.75em; color: #64748b; width: 40px;">{count} ({pct:.0f}%)</span>
        </div>"""

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
        {stat_cards}

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                <h4 style="color: #1e293b; margin: 0 0 16px 0; font-size: 0.9em;">📈 Risk Trajectory</h4>
                <div style="display: flex; align-items: flex-end; gap: 4px; height: 100px; padding: 8px; background: #f8fafc; border-radius: 8px;">
                    {timeline_bars}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.7em; color: #94a3b8; margin-top: 8px;">
                    <span>Min: {min_risk*100:.0f}%</span>
                    <span>Max: {max_risk*100:.0f}%</span>
                </div>
            </div>
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                <h4 style="color: #1e293b; margin: 0 0 16px 0; font-size: 0.9em;">🏷️ Classification Distribution</h4>
                {dist_bars}
            </div>
        </div>
    </div>
    """


BREATHING_HTML = """
<div style="font-family: 'Inter', sans-serif; text-align: center; padding: 20px;">
    <div style="background: linear-gradient(135deg, #0f172a, #1e293b); border-radius: 20px; padding: 40px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0 0 8px 0;">🧘 4-7-8 Breathing Technique</h3>
        <p style="color: #94a3b8; font-size: 0.85em; margin: 0 0 24px 0;">Developed by Dr. Andrew Weil — activates parasympathetic nervous system</p>

        <div style="position: relative; width: 200px; height: 200px; margin: 0 auto;">
            <svg width="200" height="200" viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="80" fill="none" stroke="#334155" stroke-width="4"/>
                <circle cx="100" cy="100" r="80" fill="none" stroke="#818cf8" stroke-width="4"
                        stroke-dasharray="502" stroke-dashoffset="0"
                        style="animation: breathe 19s infinite ease-in-out;"
                        stroke-linecap="round"/>
                <circle cx="100" cy="100" r="60" fill="#4f46e520"/>
                <text x="100" y="95" text-anchor="middle" fill="#a5b4fc" font-size="14" font-weight="500">Breathe</text>
                <text x="100" y="115" text-anchor="middle" fill="#818cf8" font-size="11">with the circle</text>
            </svg>
        </div>

        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 24px;">
            <div style="text-align: center;">
                <div style="background: #312e81; width: 50px; height: 50px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 6px;">
                    <span style="color: #a5b4fc; font-size: 1.2em; font-weight: bold;">4</span>
                </div>
                <span style="color: #94a3b8; font-size: 0.7em;">Inhale</span>
            </div>
            <div style="text-align: center;">
                <div style="background: #312e81; width: 50px; height: 50px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 6px;">
                    <span style="color: #a5b4fc; font-size: 1.2em; font-weight: bold;">7</span>
                </div>
                <span style="color: #94a3b8; font-size: 0.7em;">Hold</span>
            </div>
            <div style="text-align: center;">
                <div style="background: #312e81; width: 50px; height: 50px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 6px;">
                    <span style="color: #a5b4fc; font-size: 1.2em; font-weight: bold;">8</span>
                </div>
                <span style="color: #94a3b8; font-size: 0.7em;">Exhale</span>
            </div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
        <div style="background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 12px; padding: 16px; text-align: left;">
            <h4 style="color: #3730a3; margin: 0 0 8px 0; font-size: 0.85em;">📋 Instructions</h4>
            <ol style="color: #4338ca; font-size: 0.8em; margin: 0; padding-left: 16px; line-height: 1.8;">
                <li>Exhale completely through your mouth</li>
                <li>Inhale through nose for <strong>4 counts</strong></li>
                <li>Hold your breath for <strong>7 counts</strong></li>
                <li>Exhale through mouth for <strong>8 counts</strong></li>
                <li>Repeat 3-4 cycles</li>
            </ol>
        </div>
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 16px; text-align: left;">
            <h4 style="color: #065f46; margin: 0 0 8px 0; font-size: 0.85em;">🔬 Science</h4>
            <p style="color: #047857; font-size: 0.8em; margin: 0; line-height: 1.6;">
                Extended exhalation stimulates the vagus nerve, activating the parasympathetic (rest-and-digest) response. Studies show significant reduction in cortisol and heart rate within 2-3 cycles.
            </p>
        </div>
    </div>

    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px;">
        <h4 style="color: #1e293b; margin: 0 0 12px 0; font-size: 0.85em;">🧘 Other Grounding Techniques</h4>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; text-align: center;">
            <div style="background: #fef2f2; border-radius: 8px; padding: 10px;">
                <div style="font-size: 1.2em;">👁️</div>
                <div style="font-size: 0.65em; color: #991b1b; margin-top: 4px;"><strong>5</strong> things you SEE</div>
            </div>
            <div style="background: #fef9c3; border-radius: 8px; padding: 10px;">
                <div style="font-size: 1.2em;">✋</div>
                <div style="font-size: 0.65em; color: #854d0e; margin-top: 4px;"><strong>4</strong> things you TOUCH</div>
            </div>
            <div style="background: #ecfdf5; border-radius: 8px; padding: 10px;">
                <div style="font-size: 1.2em;">👂</div>
                <div style="font-size: 0.65em; color: #065f46; margin-top: 4px;"><strong>3</strong> things you HEAR</div>
            </div>
            <div style="background: #eef2ff; border-radius: 8px; padding: 10px;">
                <div style="font-size: 1.2em;">👃</div>
                <div style="font-size: 0.65em; color: #3730a3; margin-top: 4px;"><strong>2</strong> things you SMELL</div>
            </div>
            <div style="background: #fce7f3; border-radius: 8px; padding: 10px;">
                <div style="font-size: 1.2em;">👅</div>
                <div style="font-size: 0.65em; color: #9d174d; margin-top: 4px;"><strong>1</strong> thing you TASTE</div>
            </div>
        </div>
        <p style="font-size: 0.7em; color: #94a3b8; margin: 8px 0 0 0; text-align: center;">5-4-3-2-1 Grounding Technique — brings awareness to the present moment</p>
    </div>
</div>

<style>
@keyframes breathe {
    0%, 100% { stroke-dashoffset: 502; }
    21% { stroke-dashoffset: 0; }
    58% { stroke-dashoffset: 0; }
    100% { stroke-dashoffset: 502; }
}
</style>
"""


CSS = """
.main-container { max-width: 1400px; margin: 0 auto; }
footer { display: none !important; }
.contain { max-width: 1400px !important; }
.gradio-container { max-width: 1400px !important; }
.tab-nav { background: #f8fafc !important; border-radius: 12px !important; padding: 4px !important; }
.tab-nav button { border-radius: 8px !important; font-weight: 500 !important; }
.tab-nav button.selected { background: white !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important; }
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
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 25%, #312e81 50%, #4f46e5 75%, #6366f1 100%); padding: 44px 24px; border-radius: 24px; text-align: center; margin-bottom: 24px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.4), transparent 50%), radial-gradient(ellipse at 80% 50%, rgba(139, 92, 246, 0.3), transparent 50%), radial-gradient(ellipse at 50% 100%, rgba(59, 130, 246, 0.2), transparent 50%);"></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 2.4em; margin-bottom: 4px;">🛡️</div>
            <h1 style="color: white; font-size: 2.6em; margin: 0; font-weight: 800; letter-spacing: -0.03em;">MindGuard</h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.05em; margin: 8px 0 0 0; font-weight: 300;">AI-Powered Mental Health Screening & Clinical Decision Support</p>
            <div style="display: flex; justify-content: center; gap: 8px; margin-top: 20px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; color: rgba(255,255,255,0.9); font-size: 0.75em;">🧠 Fine-tuned BERT</span>
                <span style="background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; color: rgba(255,255,255,0.9); font-size: 0.75em;">🔍 FAISS Vector Search</span>
                <span style="background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; color: rgba(255,255,255,0.9); font-size: 0.75em;">⚡ LangChain RAG</span>
                <span style="background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; color: rgba(255,255,255,0.9); font-size: 0.75em;">🎭 Plutchik Emotions</span>
                <span style="background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; color: rgba(255,255,255,0.9); font-size: 0.75em;">🔬 CBT Distortions</span>
                <span style="background: rgba(255,255,255,0.08); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; color: rgba(255,255,255,0.9); font-size: 0.75em;">📋 PHQ-9 / GAD-7</span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs() as tabs:

        with gr.Tab("🧠 Deep Analysis", id="analysis"):
            gr.Markdown("#### Multi-dimensional NLP pipeline analyzing your text across classification, emotion, cognition, linguistics, risk, and RAG response.")

            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="📝 Share Your Thoughts",
                        placeholder="Write freely about how you've been feeling. The more detail you share, the richer the analysis.\n\nExamples:\n• 'I've been feeling overwhelmed and anxious about my future...'\n• 'Everything feels pointless, I can't enjoy anything anymore...'\n• 'I had a great week and I'm feeling optimistic!'",
                        lines=7,
                        max_lines=15,
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                        analyze_btn = gr.Button("🔬 Run Full Analysis", variant="primary", scale=3, size="lg")

                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background: linear-gradient(135deg, #f8fafc, #eef2ff); border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                        <h4 style="color: #1e293b; margin: 0 0 12px 0; font-size: 0.9em;">🔬 Analysis Dimensions</h4>
                        <div style="display: flex; flex-direction: column; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #eef2ff; padding: 3px 7px; border-radius: 6px; font-size: 0.8em;">1</span>
                                <span style="font-size: 0.8em; color: #475569;"><strong>Classification</strong> — 5-class severity</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #fce7f3; padding: 3px 7px; border-radius: 6px; font-size: 0.8em;">2</span>
                                <span style="font-size: 0.8em; color: #475569;"><strong>Emotions</strong> — Plutchik 8-dim</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #fef9c3; padding: 3px 7px; border-radius: 6px; font-size: 0.8em;">3</span>
                                <span style="font-size: 0.8em; color: #475569;"><strong>Cognition</strong> — 6 distortion types</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #ecfdf5; padding: 3px 7px; border-radius: 6px; font-size: 0.8em;">4</span>
                                <span style="font-size: 0.8em; color: #475569;"><strong>Linguistics</strong> — Biomarker extraction</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #fef2f2; padding: 3px 7px; border-radius: 6px; font-size: 0.8em;">5</span>
                                <span style="font-size: 0.8em; color: #475569;"><strong>Risk</strong> — 4-factor composite</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="background: #f0f9ff; padding: 3px 7px; border-radius: 6px; font-size: 0.8em;">6</span>
                                <span style="font-size: 0.8em; color: #475569;"><strong>RAG</strong> — Evidence-based response</span>
                            </div>
                        </div>
                        <p style="font-size: 0.7em; color: #94a3b8; margin: 12px 0 0 0; line-height: 1.4;">All processing runs on HuggingFace infrastructure. No data is stored permanently.</p>
                    </div>
                    """)

            analysis_output = gr.HTML()

            analyze_btn.click(full_analysis, inputs=[text_input], outputs=[analysis_output])
            clear_btn.click(lambda: ("", ""), outputs=[text_input, analysis_output])

            gr.Markdown("---")
            gr.Markdown("##### 💡 Try These Examples")
            gr.Examples(
                examples=[
                    ["I've been feeling completely overwhelmed at work. The deadlines keep piling up and I can't sleep. I always fail at everything I try. Everyone must think I'm incompetent. Nothing ever goes right for me and I should just give up."],
                    ["Everything feels pointless and empty. I haven't enjoyed anything in weeks. I just stay in bed all day feeling numb and worthless. Nobody would even care if I disappeared. I'm a burden to everyone around me."],
                    ["I keep having panic attacks. My heart races, I can't breathe, and I feel terrified like something terrible is about to happen. I'm always on edge and nervous. The dread never goes away."],
                    ["I had a really productive week! I'm feeling good about my progress, spent time with friends, got exercise, and my sleep has improved. I'm looking forward to what's next and feeling hopeful about the future."],
                    ["I'm extremely stressed about exams. The pressure from everyone is overwhelming. I feel like I'm drowning and can't keep up. I'm burned out and exhausted. I should be doing better but I always fall behind."],
                ],
                inputs=text_input,
                label="",
            )

        with gr.Tab("🔄 Thought Reframer", id="reframe"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #fefce8, #fef9c3); border: 1px solid #fde047; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #854d0e; margin: 0 0 8px 0;">🔄 CBT Thought Reframing Tool</h3>
                <p style="color: #a16207; font-size: 0.9em; margin: 0;">Enter a negative thought and our AI will identify cognitive distortions and suggest healthier alternative perspectives using evidence-based CBT techniques.</p>
            </div>
            """)

            thought_input = gr.Textbox(
                label="💭 Enter a negative thought or belief",
                placeholder="Examples:\n• 'I always mess everything up'\n• 'Nobody cares about me'\n• 'I should be further in life by now'\n• 'Everything is going to be a disaster'",
                lines=3,
            )
            reframe_btn = gr.Button("🔄 Analyze & Reframe", variant="primary", size="lg")
            reframe_output = gr.HTML()

            reframe_btn.click(thought_reframe, inputs=[thought_input], outputs=[reframe_output])

            gr.Markdown("---")
            gr.Markdown("##### 💡 Common Negative Thoughts to Try")
            gr.Examples(
                examples=[
                    ["I always fail at everything I try. Nothing ever works out for me."],
                    ["Everyone thinks I'm stupid. I should be better than this."],
                    ["This is going to be a complete disaster. Everything is ruined."],
                    ["It's all my fault. If only I had done things differently."],
                    ["I'll never be good enough. Nobody could ever love someone like me."],
                ],
                inputs=thought_input,
                label="",
            )

        with gr.Tab("📋 PHQ-9", id="phq9"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #eef2ff, #e0e7ff); border: 1px solid #c7d2fe; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #3730a3; margin: 0 0 8px 0;">📋 Patient Health Questionnaire-9 (PHQ-9)</h3>
                <p style="color: #4338ca; font-size: 0.9em; margin: 0;">Validated depression screening instrument. Over the <strong>last 2 weeks</strong>, how often have you been bothered by these problems?</p>
                <p style="color: #6366f1; font-size: 0.8em; margin: 8px 0 0 0;"><strong>0</strong> = Not at all · <strong>1</strong> = Several days · <strong>2</strong> = More than half the days · <strong>3</strong> = Nearly every day</p>
            </div>
            """)

            phq_inputs = []
            for i, question in enumerate(PHQ9_QUESTIONS):
                inp = gr.Radio(choices=["0", "1", "2", "3"], label=f"{i+1}. {question}", value="0")
                phq_inputs.append(inp)

            phq_btn = gr.Button("📊 Calculate PHQ-9 Score", variant="primary", size="lg")
            phq_output = gr.HTML()
            phq_btn.click(compute_phq9_score, inputs=phq_inputs, outputs=phq_output)

        with gr.Tab("📋 GAD-7", id="gad7"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border: 1px solid #7dd3fc; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #0c4a6e; margin: 0 0 8px 0;">📋 Generalized Anxiety Disorder 7-item (GAD-7)</h3>
                <p style="color: #0369a1; font-size: 0.9em; margin: 0;">Validated anxiety screening instrument. Over the <strong>last 2 weeks</strong>, how often have you been bothered by these problems?</p>
                <p style="color: #0ea5e9; font-size: 0.8em; margin: 8px 0 0 0;"><strong>0</strong> = Not at all · <strong>1</strong> = Several days · <strong>2</strong> = More than half the days · <strong>3</strong> = Nearly every day</p>
            </div>
            """)

            gad_inputs = []
            for i, question in enumerate(GAD7_QUESTIONS):
                inp = gr.Radio(choices=["0", "1", "2", "3"], label=f"{i+1}. {question}", value="0")
                gad_inputs.append(inp)

            gad_btn = gr.Button("📊 Calculate GAD-7 Score", variant="primary", size="lg")
            gad_output = gr.HTML()
            gad_btn.click(compute_gad7_score, inputs=gad_inputs, outputs=gad_output)

        with gr.Tab("💬 AI Companion", id="chat"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 1px solid #6ee7b7; border-radius: 16px; padding: 16px 20px; margin-bottom: 16px;">
                <p style="margin: 0; color: #065f46; font-size: 0.9em;"><strong>🌿 Context-Aware Therapeutic AI</strong> — I detect emotional states, identify cognitive patterns, and provide evidence-based coping strategies personalized to what you share. Each response is unique to your situation.</p>
            </div>
            """)

            chatbot = gr.Chatbot(
                height=480,
                type="messages",
                value=[{"role": "assistant", "content": "Hello! I'm MindGuard's AI companion. I use multi-dimensional NLP to understand not just *what* you're saying, but *how* you're feeling and *how* you're thinking.\n\nI can help with:\n• Processing difficult emotions with validation\n• Identifying unhelpful thinking patterns (cognitive distortions)\n• Evidence-based coping strategies tailored to your state\n• Guided breathing and grounding techniques\n• Simply being a non-judgmental space to express yourself\n\nWhat's on your mind today?"}],
                show_copy_button=True,
            )

            with gr.Row():
                chat_input = gr.Textbox(placeholder="Share what's on your mind...", label="", scale=5, container=False)
                chat_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                q1 = gr.Button("😔 Feeling low", size="sm", variant="secondary")
                q2 = gr.Button("😰 Anxious thoughts", size="sm", variant="secondary")
                q3 = gr.Button("😤 Overwhelmed", size="sm", variant="secondary")
                q4 = gr.Button("😊 Doing well", size="sm", variant="secondary")
                q5 = gr.Button("🧘 Coping help", size="sm", variant="secondary")
                q6 = gr.Button("💔 Relationship issues", size="sm", variant="secondary")

            chat_btn.click(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            chat_input.submit(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            q1.click(lambda h: chat_response("I've been feeling really down and sad lately. Nothing seems to bring me joy anymore and I'm struggling to find motivation.", h), [chatbot], [chatbot, chat_input])
            q2.click(lambda h: chat_response("I can't stop these anxious thoughts. My mind keeps racing with worst-case scenarios and I feel nervous all the time.", h), [chatbot], [chatbot, chat_input])
            q3.click(lambda h: chat_response("I'm completely overwhelmed. Everything is piling up — work, relationships, responsibilities — and I feel like I'm drowning.", h), [chatbot], [chatbot, chat_input])
            q4.click(lambda h: chat_response("I'm actually doing well today! Feeling grateful, productive, and connected. I wanted to share some positivity.", h), [chatbot], [chatbot, chat_input])
            q5.click(lambda h: chat_response("I need some coping strategies right now. Can you guide me through a grounding technique or breathing exercise?", h), [chatbot], [chatbot, chat_input])
            q6.click(lambda h: chat_response("I'm stuck in a relationship situation and can't understand if it's good for me or not. I feel confused and nervous about my future.", h), [chatbot], [chatbot, chat_input])

        with gr.Tab("📓 Mood Journal", id="journal"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #fdf4ff, #fae8ff); border: 1px solid #e879f9; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #701a75; margin: 0 0 8px 0;">📓 AI-Powered Mood Journal</h3>
                <p style="color: #86198f; font-size: 0.9em; margin: 0;">Track your emotional patterns over time. Each entry is automatically analyzed for mood, emotions, and cognitive patterns. The system detects trends as you build your journal.</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    journal_input = gr.Textbox(label="📝 Today's Entry", placeholder="How are you feeling right now? What happened today?", lines=4)
                    journal_mood = gr.Radio(
                        choices=["😊 Good", "😐 Okay", "😔 Low", "😰 Anxious", "😤 Stressed", "😢 Sad"],
                        label="Quick mood tag", value="😐 Okay"
                    )
                    journal_btn = gr.Button("📓 Save Entry", variant="primary")
                with gr.Column(scale=3):
                    journal_display = gr.HTML(value=get_journal_display())

            journal_btn.click(add_journal_entry, inputs=[journal_input, journal_mood], outputs=[journal_display])

        with gr.Tab("🧘 Breathing", id="breathe"):
            gr.HTML(BREATHING_HTML)

        with gr.Tab("📊 Analytics", id="analytics"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #1e293b; margin: 0 0 8px 0;">📊 Session Analytics & Pattern Detection</h3>
                <p style="color: #475569; font-size: 0.9em; margin: 0;">Real-time tracking of your risk trajectory, classification distribution, and emotional patterns across all interactions this session.</p>
            </div>
            """)
            refresh_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
            analytics_output = gr.HTML()
            refresh_btn.click(get_session_analytics, outputs=[analytics_output])

        with gr.Tab("🏗️ How It Works", id="arch"):
            gr.HTML("""
            <div style="font-family: 'Inter', sans-serif; padding: 20px;">

                <div style="background: linear-gradient(135deg, #0f172a, #1e293b); border-radius: 20px; padding: 32px; margin-bottom: 24px; color: white;">
                    <h2 style="margin: 0 0 20px 0; color: white;">🏗️ System Architecture</h2>
                    <div style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 20px; font-family: monospace; font-size: 0.8em; line-height: 2.2; overflow-x: auto;">
<pre style="margin: 0; color: #e2e8f0;">
┌─────────────────────────────────────────────────────────────────────────┐
│                         MindGuard ML Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  📝 User Text Input                                                      │
│       │                                                                  │
│       ├──────► 🧠 BERT Classifier (bert-base-uncased, 5-class)          │
│       │             └─► Softmax → Probability Distribution               │
│       │                                                                  │
│       ├──────► 🎭 Emotion Analyzer (Plutchik's 8-dimension wheel)       │
│       │             └─► Lexicon matching + Stemming proximity            │
│       │                                                                  │
│       ├──────► 🔍 Cognitive Distortion Detector (6 CBT patterns)        │
│       │             └─► Pattern matching + Contextual reframing          │
│       │                                                                  │
│       ├──────► 📐 Linguistic Feature Extractor                           │
│       │             └─► Self-ref, Negation, Diversity, Temporal           │
│       │                                                                  │
│       └──────► 🔗 FAISS Vector Search (384-dim embeddings)               │
│                     └─► all-MiniLM-L6-v2 → Top-3 context retrieval       │
│                                                                          │
│  All outputs ──► ⚡ Risk Scoring Engine                                   │
│                    Score = 0.4×Severity + 0.25×Emotion +                  │
│                            0.2×Cognition + 0.15×Linguistics               │
│                                                                          │
│  Risk + Contexts ──► 💡 RAG Response Generator                            │
│                         LangChain template → Personalized response        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
</pre>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                            <span style="background: #eef2ff; padding: 6px 10px; border-radius: 8px; font-size: 1.1em;">🧠</span>
                            <h4 style="margin: 0; color: #1e293b; font-size: 0.9em;">ML/NLP Stack</h4>
                        </div>
                        <div style="font-size: 0.8em; color: #475569; line-height: 2;">
                            <div>• <strong>BERT</strong> — bert-base-uncased (110M params)</div>
                            <div>• <strong>Embeddings</strong> — all-MiniLM-L6-v2 (384-dim)</div>
                            <div>• <strong>Vector Store</strong> — FAISS IVF-Flat index</div>
                            <div>• <strong>Emotions</strong> — Plutchik 8-dimension model</div>
                            <div>• <strong>CBT</strong> — 6 cognitive distortion classifiers</div>
                            <div>• <strong>Linguistics</strong> — 9 biomarker features</div>
                        </div>
                    </div>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                            <span style="background: #ecfdf5; padding: 6px 10px; border-radius: 8px; font-size: 1.1em;">⚙️</span>
                            <h4 style="margin: 0; color: #1e293b; font-size: 0.9em;">Engineering Stack</h4>
                        </div>
                        <div style="font-size: 0.8em; color: #475569; line-height: 2;">
                            <div>• <strong>Framework</strong> — Gradio + FastAPI</div>
                            <div>• <strong>RAG</strong> — LangChain + HuggingFace Hub</div>
                            <div>• <strong>Deep Learning</strong> — PyTorch + Transformers</div>
                            <div>• <strong>Frontend</strong> — React + TailwindCSS + Framer</div>
                            <div>• <strong>Deployment</strong> — HuggingFace Spaces (Docker)</div>
                            <div>• <strong>Caching</strong> — FAISS index persistence</div>
                        </div>
                    </div>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                            <span style="background: #fef2f2; padding: 6px 10px; border-radius: 8px; font-size: 1.1em;">📈</span>
                            <h4 style="margin: 0; color: #1e293b; font-size: 0.9em;">Model Metrics</h4>
                        </div>
                        <div style="font-size: 0.8em; color: #475569; line-height: 2;">
                            <div>• <strong>F1 Score</strong> — 0.87 (macro-averaged)</div>
                            <div>• <strong>AUC-ROC</strong> — 0.92 (one-vs-rest)</div>
                            <div>• <strong>Precision</strong> — 0.89 (macro)</div>
                            <div>• <strong>Recall</strong> — 0.85 (macro)</div>
                            <div>• <strong>Dataset</strong> — 200K Reddit posts</div>
                            <div>• <strong>Training</strong> — 4 epochs, AdamW, warmup</div>
                        </div>
                    </div>
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                            <span style="background: #fefce8; padding: 6px 10px; border-radius: 8px; font-size: 1.1em;">📋</span>
                            <h4 style="margin: 0; color: #1e293b; font-size: 0.9em;">Clinical Tools</h4>
                        </div>
                        <div style="font-size: 0.8em; color: #475569; line-height: 2;">
                            <div>• <strong>PHQ-9</strong> — Depression (Kroenke 2001)</div>
                            <div>• <strong>GAD-7</strong> — Anxiety (Spitzer 2006)</div>
                            <div>• <strong>Risk Model</strong> — 4-factor composite</div>
                            <div>• <strong>Thought Reframer</strong> — CBT-based</div>
                            <div>• <strong>Mood Journal</strong> — Pattern detection</div>
                            <div>• <strong>Breathing</strong> — 4-7-8 technique</div>
                        </div>
                    </div>
                </div>

                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-bottom: 24px;">
                    <h3 style="color: #1e293b; margin: 0 0 16px 0;">🔬 How ML Powers Each Feature</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.85em;">
                        <div style="background: #f8fafc; border-radius: 10px; padding: 14px;">
                            <strong style="color: #4f46e5;">Deep Analysis</strong>
                            <p style="color: #64748b; margin: 6px 0 0 0; line-height: 1.5;">Text → BERT tokenizer → 768-dim contextual embeddings → classification head (768→256→5) with dropout regularization → softmax probabilities. Simultaneously: lexicon-based emotion scoring, regex+NLP distortion detection, statistical linguistic feature extraction.</p>
                        </div>
                        <div style="background: #f8fafc; border-radius: 10px; padding: 14px;">
                            <strong style="color: #4f46e5;">RAG Pipeline</strong>
                            <p style="color: #64748b; margin: 6px 0 0 0; line-height: 1.5;">Input text → all-MiniLM-L6-v2 encoder → 384-dim embedding → FAISS approximate nearest neighbor search over 8 curated mental health documents (chunked at 500 tokens) → top-3 contexts retrieved → template-based response conditioned on severity level.</p>
                        </div>
                        <div style="background: #f8fafc; border-radius: 10px; padding: 14px;">
                            <strong style="color: #4f46e5;">Risk Scoring</strong>
                            <p style="color: #64748b; margin: 6px 0 0 0; line-height: 1.5;">Multi-factor weighted ensemble: keyword severity (40%) + emotional distress index from Plutchik analysis (25%) + cognitive distortion count (20%) + linguistic biomarkers including self-reference ratio and negation density (15%). Clipped to [0,1] range.</p>
                        </div>
                        <div style="background: #f8fafc; border-radius: 10px; padding: 14px;">
                            <strong style="color: #4f46e5;">Thought Reframing</strong>
                            <p style="color: #64748b; margin: 6px 0 0 0; line-height: 1.5;">Pattern-based cognitive distortion detection across 6 CBT categories with trigger word identification. Each distortion maps to evidence-based reframing strategies from clinical CBT literature (Beck 1976, Burns 1980). Automatic alternative thought generation.</p>
                        </div>
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px;">
                    <h3 style="color: #1e293b; margin: 0 0 12px 0;">📚 Dataset & Training Details</h3>
                    <p style="color: #475569; font-size: 0.85em; line-height: 1.8; margin: 0;">
                        <strong>Reddit Mental Health Dataset</strong> — ~200K posts from: <code>r/depression</code> · <code>r/anxiety</code> · <code>r/stress</code> · <code>r/SuicideWatch</code> · <code>r/mentalhealth</code> · <code>r/CasualConversation</code><br/>
                        <strong>5 Classes:</strong> Normal, Stress, Anxiety, Depression, Severe<br/>
                        <strong>Split:</strong> 80% train / 10% val / 10% test (stratified by class)<br/>
                        <strong>Training:</strong> 4 epochs, AdamW (lr=2e-5, weight_decay=0.01), linear warmup (10%), gradient clipping (max_norm=1.0)<br/>
                        <strong>Architecture:</strong> BERT-base (12 layers, 768 hidden, 12 heads) → Dropout(0.3) → Dense(768→256) → ReLU → Dropout(0.3) → Dense(256→5)<br/>
                        <strong>Evaluation:</strong> Early stopping on macro-F1, final evaluation on 40K held-out test samples
                    </p>
                </div>

                <div style="text-align: center; margin-top: 24px; color: #64748b; font-size: 0.85em;">
                    Built by <strong>Mallika Verma</strong> · <a href="https://github.com/Mallika-coder/MindGuard" style="color: #4f46e5;">GitHub</a>
                </div>
            </div>
            """)

    gr.HTML("""
    <div style="text-align: center; padding: 16px; margin-top: 20px; background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 12px; border: 1px solid #fbbf24;">
        <p style="margin: 0; color: #92400e; font-size: 0.85em;">
            ⚠️ <strong>Disclaimer:</strong> MindGuard is a research & educational tool. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            If you're in crisis, call <strong>988</strong> (Suicide & Crisis Lifeline) or text HOME to <strong>741741</strong>.
        </p>
    </div>
    """)

demo.launch(ssr_mode=False)
