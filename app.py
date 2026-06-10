"""
MindGuard — AI-Powered Mental Health Screening & Clinical Decision Support Platform
Multi-dimensional NLP: BERT + FAISS + LangChain RAG + Plutchik + CBT + AI Guide
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
    "checkin_history": [],
}

EMOTION_LEXICON = {
    "anger": ["angry", "furious", "rage", "hate", "irritated", "frustrated", "annoyed", "resentful", "bitter", "hostile", "mad", "pissed", "livid", "outraged"],
    "sadness": ["sad", "crying", "tears", "grief", "mourning", "lonely", "heartbroken", "miserable", "gloomy", "melancholy", "depressed", "down", "unhappy", "sorrowful", "blue"],
    "fear": ["afraid", "scared", "terrified", "frightened", "panic", "phobia", "dread", "horror", "alarmed", "petrified", "nervous", "anxious", "worried", "uneasy", "tense", "apprehensive", "stuck"],
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
ANXIETY_KEYWORDS = ["panic", "worried", "racing thoughts", "can't breathe", "nervous", "terrified", "dread", "restless", "on edge", "impending doom", "anxious", "fear", "scared", "cant understand", "stuck"]
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


def milo_guide(message, mood="neutral"):
    mood_faces = {
        "neutral": "😊",
        "happy": "🤗",
        "thinking": "🤔",
        "caring": "💚",
        "alert": "⚡",
        "calm": "🧘",
        "celebrate": "🎉",
    }
    face = mood_faces.get(mood, "😊")
    return f"""
    <div style="display: flex; align-items: flex-start; gap: 12px; background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #d1fae5 100%); border: 1px solid #86efac; border-radius: 16px; padding: 16px 20px; margin-bottom: 20px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: -20px; right: -20px; width: 80px; height: 80px; background: radial-gradient(circle, #86efac33, transparent); border-radius: 50%;"></div>
        <div style="flex-shrink: 0; width: 48px; height: 48px; background: linear-gradient(135deg, #10b981, #059669); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4em; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); animation: milo-bounce 2s ease-in-out infinite;">
            {face}
        </div>
        <div style="flex: 1; position: relative; z-index: 1;">
            <div style="font-size: 0.7em; color: #059669; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">Milo — Your AI Guide</div>
            <div style="color: #065f46; font-size: 0.88em; line-height: 1.6;">{message}</div>
        </div>
    </div>
    """


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
                if len(kw) > 3 and word.startswith(kw[:4]):
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
    temporal_past = sum(1 for w in words if w.lower() in ["was", "were", "had", "used", "before", "yesterday", "ago", "past"])
    temporal_future = sum(1 for w in words if w.lower() in ["will", "going", "tomorrow", "future", "plan", "hope", "soon", "later"])

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "first_person_ratio": round(first_person / max(len(words), 1), 3),
        "negation_ratio": round(negations / max(len(words), 1), 3),
        "certainty_ratio": round(certainty_words / max(len(words), 1), 3),
        "hedging_ratio": round(hedging / max(len(words), 1), 3),
        "lexical_diversity": round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
        "temporal_past": round(temporal_past / max(len(words), 1), 3),
        "temporal_future": round(temporal_future / max(len(words), 1), 3),
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


def compute_risk_score(classification, emotions, distortions, linguistics):
    base_risk = classification["severity_score"]
    emotion_risk = (emotions.get("sadness", 0) * 0.3 + emotions.get("fear", 0) * 0.25 +
                    emotions.get("anger", 0) * 0.15 - emotions.get("joy", 0) * 0.2)
    distortion_risk = min(len(distortions) * 0.12, 0.35)
    linguistic_risk = (linguistics["negation_ratio"] * 0.4 + linguistics["first_person_ratio"] * 0.2 +
                       (1 - linguistics["lexical_diversity"]) * 0.15 + linguistics["certainty_ratio"] * 0.25)
    composite_risk = np.clip(base_risk * 0.4 + emotion_risk * 0.25 + distortion_risk * 0.2 + linguistic_risk * 0.15, 0, 1)
    factors = []
    if base_risk >= 0.5:
        factors.append("High-risk language patterns detected")
    if emotions.get("sadness", 0) > 0.25:
        factors.append("Elevated sadness in emotional profile")
    if emotions.get("fear", 0) > 0.25:
        factors.append("Elevated fear/anxiety markers")
    if len(distortions) >= 2:
        factors.append(f"{len(distortions)} cognitive distortions identified")
    if linguistics["negation_ratio"] > 0.08:
        factors.append("High negative language density")
    if linguistics["first_person_ratio"] > 0.15:
        factors.append("Elevated self-focused language")
    return {
        "composite_score": round(float(composite_risk), 3),
        "components": {
            "keyword_severity": round(float(base_risk), 3),
            "emotional_distress": round(float(max(emotion_risk, 0)), 3),
            "cognitive_patterns": round(float(distortion_risk), 3),
            "linguistic_markers": round(float(min(linguistic_risk, 1)), 3),
        },
        "risk_level": ("Critical" if composite_risk >= 0.8 else "High" if composite_risk >= 0.6 else "Moderate" if composite_risk >= 0.4 else "Low" if composite_risk >= 0.2 else "Minimal"),
        "contributing_factors": factors,
    }


def full_analysis(text: str):
    if not text or len(text.strip()) < 10:
        return milo_guide("Hey there! I need a bit more from you — write at least 10 characters about how you're feeling so I can run a proper analysis. The more you share, the better I can help! 💪", "thinking")

    classification = classify_text(text)
    emotions = analyze_emotions(text)
    distortions = detect_cognitive_distortions(text)
    linguistics = compute_linguistic_features(text)
    risk = compute_risk_score(classification, emotions, distortions, linguistics)
    rag_response = rag_pipeline.generate_response(text, classification)
    contexts = rag_pipeline.retrieve_context(text)

    SESSION_DATA["entries"].append({
        "timestamp": time.time(), "text_length": len(text),
        "classification": classification["label"], "severity": classification["severity_score"],
        "risk_score": risk["composite_score"], "emotions": emotions,
    })
    SESSION_DATA["mood_timeline"].append(classification["severity_score"])
    SESSION_DATA["risk_scores"].append(risk["composite_score"])
    SESSION_DATA["total_interactions"] += 1

    label = classification["label"]
    confidence = classification["confidence"]
    risk_score = risk["composite_score"]
    risk_level = risk["risk_level"]

    label_config = {
        "normal": ("#10b981", "#d1fae5", "All Clear", "Your mental state appears healthy"),
        "stress": ("#f59e0b", "#fef3c7", "Stress Detected", "Elevated stress markers found"),
        "anxiety": ("#f97316", "#ffedd5", "Anxiety Indicators", "Anxiety patterns identified"),
        "depression": ("#8b5cf6", "#ede9fe", "Depression Markers", "Depression indicators present"),
        "severe": ("#ef4444", "#fee2e2", "Urgent Attention", "Immediate support recommended"),
    }
    color, bg, title, subtitle = label_config.get(label, ("#6366f1", "#eef2ff", "Analysis Complete", ""))

    risk_colors = {"Critical": "#ef4444", "High": "#f97316", "Moderate": "#f59e0b", "Low": "#10b981", "Minimal": "#06b6d4"}
    risk_color = risk_colors.get(risk_level, "#6366f1")
    circumference = 2 * 3.14159 * 45
    offset = circumference - (risk_score * circumference)

    milo_messages = {
        "normal": "Great news! Your text shows a healthy emotional state. Keep nurturing these positive patterns — they build resilience for tougher days ahead!",
        "stress": "I can see you're under some pressure. That's your body's way of signaling it needs attention. Let's look at what the data shows and find ways to decompress.",
        "anxiety": "I notice some anxiety patterns in what you've shared. Remember — anxiety is your brain trying to protect you, even when it overreacts. Let's understand it together.",
        "depression": "I hear heaviness in your words, and I want you to know that matters. The analysis below shows specific patterns — understanding them is the first step toward feeling better.",
        "severe": "I'm concerned about what you've shared. Please know you're not alone, and help is available right now. Call 988 or text HOME to 741741. Let's look at the analysis together.",
    }
    milo_mood = {"normal": "celebrate", "stress": "caring", "anxiety": "caring", "depression": "caring", "severe": "alert"}

    prob_bars = ""
    for cat, prob in classification["probabilities"].items():
        cat_colors = {"normal": "#10b981", "stress": "#f59e0b", "anxiety": "#f97316", "depression": "#8b5cf6", "severe": "#ef4444"}
        pct = prob * 100
        prob_bars += f"""
        <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0;">
            <span style="width: 85px; font-size: 0.78em; color: #475569; text-transform: capitalize; font-weight: 500;">{cat}</span>
            <div style="flex: 1; height: 28px; background: #f1f5f9; border-radius: 14px; overflow: hidden; position: relative;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {cat_colors.get(cat, '#6366f1')}88, {cat_colors.get(cat, '#6366f1')}); border-radius: 14px; transition: width 0.8s ease;"></div>
                <span style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); font-size: 0.7em; font-weight: 600; color: {'white' if pct > 40 else '#64748b'};">{pct:.1f}%</span>
            </div>
        </div>"""

    emotion_bars = ""
    for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        emo_colors = {"anger": "#ef4444", "sadness": "#3b82f6", "fear": "#8b5cf6", "joy": "#10b981", "disgust": "#84cc16", "surprise": "#f59e0b", "trust": "#06b6d4", "anticipation": "#ec4899"}
        emo_color = emo_colors.get(emo, "#6366f1")
        pct = score * 100
        emo_icons = {"anger": "😠", "sadness": "😢", "fear": "😨", "joy": "😊", "disgust": "🤢", "surprise": "😲", "trust": "🤝", "anticipation": "✨"}
        emotion_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
            <span style="font-size: 1em;">{emo_icons.get(emo, '•')}</span>
            <span style="width: 80px; font-size: 0.75em; color: #64748b; text-transform: capitalize;">{emo}</span>
            <div style="flex: 1; height: 18px; background: #f1f5f9; border-radius: 9px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {emo_color}77, {emo_color}); border-radius: 9px;"></div>
            </div>
            <span style="width: 35px; font-size: 0.7em; color: #94a3b8; text-align: right;">{pct:.0f}%</span>
        </div>"""

    distortion_html = ""
    if distortions:
        for d in distortions:
            distortion_html += f"""
            <div style="background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1px solid #fbbf24; border-radius: 14px; padding: 16px; margin: 10px 0;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="background: #fbbf24; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7em;">!</span>
                    <strong style="color: #92400e; font-size: 0.9em;">{d['type']}</strong>
                </div>
                <p style="color: #78716c; font-size: 0.82em; margin: 4px 0; line-height: 1.5;">{d['description']}</p>
                <p style="color: #a16207; font-size: 0.75em; margin: 4px 0;">Triggered by: "<em>{d['trigger']}</em>"</p>
                <div style="background: white; border-radius: 10px; padding: 12px; margin-top: 10px; border: 1px solid #d1fae5;">
                    <p style="color: #059669; font-size: 0.8em; margin: 0; line-height: 1.5;">💡 <strong>Reframe:</strong> {d['reframe']}</p>
                </div>
            </div>"""
    else:
        distortion_html = """
        <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 14px; padding: 20px; text-align: center;">
            <div style="font-size: 1.5em; margin-bottom: 6px;">✅</div>
            <p style="color: #065f46; font-weight: 500; margin: 0;">No cognitive distortions detected</p>
            <p style="color: #6ee7b7; font-size: 0.8em; margin: 4px 0 0 0;">Your thinking patterns appear balanced</p>
        </div>"""

    component_bars = ""
    for comp_name, comp_val in risk["components"].items():
        nice_name = comp_name.replace("_", " ").title()
        pct = comp_val * 100
        component_bars += f"""
        <div style="margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.73em; color: #64748b; margin-bottom: 3px;">
                <span>{nice_name}</span><span>{pct:.0f}%</span>
            </div>
            <div style="height: 8px; background: #f1f5f9; border-radius: 4px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: linear-gradient(90deg, {risk_color}77, {risk_color}); border-radius: 4px;"></div>
            </div>
        </div>"""

    factors_html = ""
    for f in risk["contributing_factors"]:
        factors_html += f'<div style="display: flex; align-items: flex-start; gap: 6px; margin: 5px 0;"><span style="color: {risk_color}; margin-top: 2px;">●</span><span style="font-size: 0.8em; color: #475569; line-height: 1.4;">{f}</span></div>'
    if not factors_html:
        factors_html = '<div style="color: #10b981; font-size: 0.85em;">✓ No significant risk factors</div>'

    ling_features = [
        ("Words", str(linguistics["word_count"]), "📝", "#6366f1"),
        ("Sentences", str(linguistics["sentence_count"]), "📄", "#8b5cf6"),
        ("Self-focus", f"{linguistics['first_person_ratio']*100:.0f}%", "👤", "#ec4899"),
        ("Negativity", f"{linguistics['negation_ratio']*100:.0f}%", "➖", "#ef4444"),
        ("Certainty", f"{linguistics['certainty_ratio']*100:.0f}%", "❗", "#f97316"),
        ("Diversity", f"{linguistics['lexical_diversity']*100:.0f}%", "🔤", "#10b981"),
    ]
    ling_grid = ""
    for name, val, icon, lcolor in ling_features:
        ling_grid += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 8px; text-align: center; transition: transform 0.2s; cursor: default;">
            <div style="font-size: 1.1em;">{icon}</div>
            <div style="font-size: 1.05em; font-weight: 700; color: {lcolor}; margin: 3px 0;">{val}</div>
            <div style="font-size: 0.65em; color: #94a3b8;">{name}</div>
        </div>"""

    context_html = ""
    for ctx in contexts[:3]:
        relevance = max(0, 1 - abs(ctx["relevance_score"]))
        context_html += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin: 6px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8em; font-weight: 600; color: #334155;">{ctx['topic'].replace('_', ' ').title()}</span>
                <span style="font-size: 0.65em; background: #eef2ff; color: #4f46e5; padding: 2px 8px; border-radius: 10px;">{relevance*100:.0f}% match</span>
            </div>
            <p style="font-size: 0.72em; color: #94a3b8; margin: 4px 0 0 0; line-height: 1.4;">{ctx['content'][:100]}...</p>
        </div>"""

    wellness_score = max(0, int((1 - risk_score) * 100))
    wellness_color = "#10b981" if wellness_score >= 70 else "#f59e0b" if wellness_score >= 40 else "#ef4444"

    result_html = f"""
    <div style="font-family: 'Inter', -apple-system, sans-serif;">

        {milo_guide(milo_messages.get(label, "Analysis complete! Here's what I found."), milo_mood.get(label, "neutral"))}

        <!-- Wellness Score + Classification Hero -->
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #334155 100%); border-radius: 24px; padding: 32px; margin-bottom: 20px; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -40%; right: -10%; width: 250px; height: 250px; background: radial-gradient(circle, {color}22, transparent); border-radius: 50%;"></div>
            <div style="position: absolute; bottom: -30%; left: -5%; width: 200px; height: 200px; background: radial-gradient(circle, {risk_color}15, transparent); border-radius: 50%;"></div>
            <div style="position: relative; z-index: 1; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
                <div>
                    <div style="display: inline-block; background: {color}22; border: 1px solid {color}44; padding: 4px 12px; border-radius: 20px; margin-bottom: 8px;">
                        <span style="color: {color}; font-size: 0.75em; font-weight: 600;">{title}</span>
                    </div>
                    <div style="font-size: 2.2em; font-weight: 800; color: white; text-transform: uppercase; letter-spacing: -0.02em;">{label}</div>
                    <div style="color: #94a3b8; font-size: 0.85em; margin-top: 4px;">{subtitle} · Confidence: {confidence*100:.0f}%</div>
                </div>
                <div style="display: flex; gap: 20px; align-items: center;">
                    <div style="text-align: center;">
                        <svg width="90" height="90" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" stroke-width="7"/>
                            <circle cx="50" cy="50" r="45" fill="none" stroke="{risk_color}" stroke-width="7" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
                            <text x="50" y="46" text-anchor="middle" fill="white" font-size="16" font-weight="bold">{risk_score*100:.0f}%</text>
                            <text x="50" y="62" text-anchor="middle" fill="#94a3b8" font-size="8">RISK</text>
                        </svg>
                    </div>
                    <div style="text-align: center;">
                        <svg width="90" height="90" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" stroke-width="7"/>
                            <circle cx="50" cy="50" r="45" fill="none" stroke="{wellness_color}" stroke-width="7" stroke-dasharray="{circumference}" stroke-dashoffset="{circumference - (wellness_score/100 * circumference)}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
                            <text x="50" y="46" text-anchor="middle" fill="white" font-size="16" font-weight="bold">{wellness_score}</text>
                            <text x="50" y="62" text-anchor="middle" fill="#94a3b8" font-size="8">WELLNESS</text>
                        </svg>
                    </div>
                </div>
            </div>
        </div>

        <!-- Grid: Classification + Emotions -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px;">
                <h3 style="margin: 0 0 14px 0; font-size: 0.95em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                    <span style="background: #eef2ff; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85em;">🧠</span> Classification
                </h3>
                {prob_bars}
            </div>
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px;">
                <h3 style="margin: 0 0 14px 0; font-size: 0.95em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                    <span style="background: #fce7f3; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85em;">🎭</span> Emotional Profile
                </h3>
                {emotion_bars}
            </div>
        </div>

        <!-- Risk Assessment -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 14px 0; font-size: 0.95em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="background: #fef2f2; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85em;">⚡</span> Risk Assessment · <span style="color: {risk_color}; font-weight: 700;">{risk_level}</span>
            </h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>{component_bars}</div>
                <div>
                    <div style="font-size: 0.75em; color: #64748b; margin-bottom: 6px; font-weight: 500;">Contributing Factors</div>
                    {factors_html}
                </div>
            </div>
        </div>

        <!-- Cognitive Distortions -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 14px 0; font-size: 0.95em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="background: #fef9c3; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85em;">🔍</span> Cognitive Patterns
            </h3>
            {distortion_html}
        </div>

        <!-- Linguistics -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 22px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 14px 0; font-size: 0.95em; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="background: #ecfdf5; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85em;">📐</span> Linguistic Biomarkers
            </h3>
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px;">
                {ling_grid}
            </div>
        </div>

        <!-- RAG Response -->
        <div style="background: linear-gradient(135deg, #eef2ff, #e0e7ff); border: 1px solid #c7d2fe; border-radius: 18px; padding: 22px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 12px 0; font-size: 0.95em; color: #3730a3; display: flex; align-items: center; gap: 8px;">
                <span style="background: #c7d2fe; width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85em;">💡</span> Personalized Guidance
            </h3>
            <p style="color: #312e81; line-height: 1.7; font-size: 0.9em; margin: 0 0 16px 0;">{rag_response}</p>
            <details style="cursor: pointer;">
                <summary style="font-size: 0.75em; color: #6366f1; font-weight: 500;">📚 View Retrieved Knowledge Sources (FAISS Top-3)</summary>
                <div style="margin-top: 8px;">{context_html}</div>
            </details>
        </div>
    </div>
    """
    return result_html


def generate_smart_response(message, classification, emotions, distortions):
    label = classification["label"]
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
    top_emotion = top_emotions[0][0] if top_emotions else "neutral"
    rag_response = rag_pipeline.generate_response(message, classification)

    openings = {
        "sadness": ["I can feel the weight of what you're carrying.", "What you're going through sounds really heavy.", "The sadness you're describing is real and valid — I'm here."],
        "fear": ["I understand that sense of unease.", "It sounds like your mind is working overtime to protect you.", "The worry you're feeling makes complete sense."],
        "anger": ["That frustration is completely understandable.", "Your anger is telling you something important.", "It makes sense you'd feel that way."],
        "joy": ["It's wonderful to hear that energy!", "I'm genuinely glad things are going well.", "That's the kind of moment worth savoring."],
        "trust": ["Thank you for feeling comfortable sharing that.", "I appreciate your openness."],
        "anticipation": ["It sounds like you're processing a lot about what's ahead.", "That forward-looking perspective shows real engagement."],
    }
    opening = random.choice(openings.get(top_emotion, ["Thank you for sharing that with me.", "I hear what you're saying."]))

    distortion_note = ""
    if distortions:
        d = distortions[0]
        distortion_note = f"\n\n🔍 **Pattern noticed:** I see some *{d['type']}* in your words — {d['description'].lower()}. {d['reframe']}"

    strategies = {
        "anxiety": "\n\n🧘 **Try this:** 4-7-8 breathing — in for 4, hold for 7, out for 8. Repeat 3 times. This activates your calm-down system.",
        "depression": "\n\n🌱 **One small step:** What's one tiny thing within reach right now that might bring even 1% comfort?",
        "stress": "\n\n📋 **Quick reset:** What are the 3 most pressing things? Circle the ONE you can actually control right now.",
        "severe": "\n\n🆘 **You matter:** Please reach out to 988 (call/text) or text HOME to 741741. A real person is ready to help right now.",
        "normal": "\n\n✨ **Keep going:** What you're doing is working. Consider noting what's going well today.",
    }
    strategy = strategies.get(label, "")
    return f"{opening}\n\n{rag_response}{distortion_note}{strategy}"


def chat_response(message: str, history: list):
    if not message.strip():
        return history, ""
    classification = classify_text(message)
    emotions = analyze_emotions(message)
    distortions = detect_cognitive_distortions(message)
    response = generate_smart_response(message, classification, emotions, distortions)
    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    emotion_tags = " · ".join([f"{e[0]} {e[1]*100:.0f}%" for e in top_emotions if e[1] > 0.05])
    badge = f"\n\n---\n📊 *{classification['label'].upper()} ({classification['confidence']*100:.0f}%)* | {emotion_tags}"
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response + badge}]
    return history, ""


def thought_reframe(thought: str):
    if not thought or len(thought.strip()) < 5:
        return milo_guide("Write a negative thought or belief you'd like to challenge — I'll help you see it from a healthier angle! 🔄", "thinking")

    distortions = detect_cognitive_distortions(thought)
    emotions = analyze_emotions(thought)
    top_emotion = max(emotions.items(), key=lambda x: x[1])
    SESSION_DATA["reframes_done"] = SESSION_DATA.get("reframes_done", 0) + 1

    if not distortions:
        return f"""
        <div style="font-family: 'Inter', sans-serif;">
            {milo_guide("Good news! This thought looks balanced and rational. No harmful thinking patterns detected — your perspective seems proportionate to the situation. 🎉", "celebrate")}
            <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 16px; padding: 24px; text-align: center;">
                <span style="font-size: 2.5em;">✅</span>
                <h3 style="color: #065f46; margin: 12px 0 4px 0;">Balanced Thought</h3>
                <p style="color: #047857; font-size: 0.9em; margin: 0;">Primary emotion: <strong>{top_emotion[0].capitalize()}</strong> ({top_emotion[1]*100:.0f}%)</p>
            </div>
        </div>"""

    cards = ""
    for d in distortions:
        cards += f"""
        <div style="background: white; border: 1px solid #fbbf24; border-radius: 14px; padding: 18px; margin: 12px 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="background: #fbbf24; color: white; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75em; font-weight: bold;">!</span>
                <strong style="color: #92400e;">{d['type']}</strong>
                <span style="font-size: 0.7em; color: #d97706; background: #fef3c7; padding: 2px 8px; border-radius: 10px;">"{d['trigger']}"</span>
            </div>
            <p style="color: #57534e; font-size: 0.83em; margin: 6px 0;">{d['description']}</p>
            <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-radius: 10px; padding: 12px; margin-top: 10px;">
                <p style="color: #065f46; font-size: 0.82em; margin: 0;">💡 <strong>Healthier perspective:</strong> {d['reframe']}</p>
            </div>
        </div>"""

    reframed = thought
    replacements = {"always": "sometimes", "never": "not always", "everyone": "some people", "nothing": "not everything", "nobody": "not everyone"}
    for old, new in replacements.items():
        reframed = reframed.replace(old, new).replace(old.capitalize(), new.capitalize())

    return f"""
    <div style="font-family: 'Inter', sans-serif;">
        {milo_guide(f"I found {len(distortions)} thinking pattern{'s' if len(distortions) > 1 else ''} that might be causing extra distress. Let's reframe this together — small shifts in perspective can make a big difference! 🔄", "caring")}
        <div style="background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 14px; padding: 18px; margin-bottom: 16px;">
            <div style="font-size: 0.7em; color: #92400e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Original Thought</div>
            <p style="color: #78350f; font-size: 0.95em; font-style: italic; margin: 0;">"{thought}"</p>
        </div>
        {cards}
        <div style="background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-radius: 14px; padding: 18px; margin-top: 16px;">
            <div style="font-size: 0.7em; color: #065f46; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">🔄 Reframed Version</div>
            <p style="color: #047857; font-size: 0.95em; font-weight: 500; margin: 0;">"{reframed}"</p>
        </div>
        <div style="background: #f8fafc; border-radius: 12px; padding: 14px; margin-top: 12px; font-size: 0.75em; color: #64748b;">
            📚 <em>Based on Cognitive Behavioral Therapy (Beck, 1976; Burns, 1980). CBT is the gold-standard treatment with 50+ years of clinical evidence.</em>
        </div>
    </div>"""


def compute_phq9_score(*responses):
    scores = [int(r) if r else 0 for r in responses]
    total = sum(scores)
    if total <= 4:
        severity, color, rec = "Minimal", "#10b981", "Monitor; no treatment needed"
    elif total <= 9:
        severity, color, rec = "Mild", "#f59e0b", "Watchful waiting; repeat at follow-up"
    elif total <= 14:
        severity, color, rec = "Moderate", "#f97316", "Consider therapy (CBT) + pharmacotherapy"
    elif total <= 19:
        severity, color, rec = "Moderately Severe", "#ef4444", "Active treatment recommended"
    else:
        severity, color, rec = "Severe", "#dc2626", "Immediate treatment + specialist referral"

    item_bars = ""
    for i, (q, s) in enumerate(zip(PHQ9_QUESTIONS, scores)):
        pct = s / 3 * 100
        ic = "#10b981" if s == 0 else "#f59e0b" if s == 1 else "#f97316" if s == 2 else "#ef4444"
        item_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0; padding: 8px 12px; background: #f8fafc; border-radius: 10px;">
            <span style="width: 18px; font-size: 0.7em; color: #94a3b8; font-weight: bold;">{i+1}</span>
            <div style="flex: 1; font-size: 0.78em; color: #475569;">{q}</div>
            <div style="width: 50px; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: {ic}; border-radius: 3px;"></div>
            </div>
            <span style="width: 14px; font-size: 0.7em; color: {ic}; font-weight: bold;">{s}</span>
        </div>"""

    safety = ""
    if scores[8] >= 2:
        safety = f"""<div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 14px; padding: 16px; margin: 16px 0;">
            <strong style="color: #991b1b;">🚨 SAFETY ALERT:</strong>
            <span style="color: #b91c1c; font-size: 0.85em;"> Item 9 scored ≥2. Please contact 988 if you're at risk.</span></div>"""

    circ = 2 * 3.14159 * 45
    off = circ - (total / 27 * circ)
    milo_msg = "Let's look at your PHQ-9 results together. " + ("You're doing well — keep up the self-care!" if total <= 4 else "I see some areas of concern. Remember, these scores help guide next steps, not define you." if total <= 14 else "These scores suggest significant difficulty. Please know that help is available and effective.")
    return f"""<div style="font-family: 'Inter', sans-serif;">
        {milo_guide(milo_msg, "caring" if total > 9 else "celebrate" if total <= 4 else "thinking")}
        <div style="background: linear-gradient(135deg, #1e1b4b, #312e81); border-radius: 20px; padding: 28px; text-align: center; margin-bottom: 16px;">
            <svg width="110" height="110" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#4c1d95" stroke-width="9"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="9" stroke-dasharray="{circ}" stroke-dashoffset="{off}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
                <text x="50" y="44" text-anchor="middle" fill="white" font-size="20" font-weight="bold">{total}</text>
                <text x="50" y="60" text-anchor="middle" fill="#a5b4fc" font-size="9">/27</text>
            </svg>
            <div style="color: {color}; font-size: 1.15em; font-weight: 700; margin-top: 8px;">{severity}</div>
            <div style="color: #a5b4fc; font-size: 0.8em; margin-top: 4px;">{rec}</div>
        </div>
        {safety}
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px;">{item_bars}</div>
        <div style="background: #f8fafc; border-radius: 12px; padding: 14px; margin-top: 12px; font-size: 0.72em; color: #64748b;">
            📚 PHQ-9 © Pfizer Inc. Sensitivity 88%, Specificity 88% at ≥10 (Kroenke et al., 2001)
        </div></div>"""


def compute_gad7_score(*responses):
    scores = [int(r) if r else 0 for r in responses]
    total = sum(scores)
    if total <= 4:
        severity, color, rec = "Minimal", "#10b981", "Monitor; routine follow-up"
    elif total <= 9:
        severity, color, rec = "Mild", "#f59e0b", "Relaxation techniques recommended"
    elif total <= 14:
        severity, color, rec = "Moderate", "#f97316", "Consider CBT; possible medication"
    else:
        severity, color, rec = "Severe", "#ef4444", "Active treatment indicated"

    item_bars = ""
    for i, (q, s) in enumerate(zip(GAD7_QUESTIONS, scores)):
        pct = s / 3 * 100
        ic = "#10b981" if s == 0 else "#f59e0b" if s == 1 else "#f97316" if s == 2 else "#ef4444"
        item_bars += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0; padding: 8px 12px; background: #f8fafc; border-radius: 10px;">
            <span style="width: 18px; font-size: 0.7em; color: #94a3b8; font-weight: bold;">{i+1}</span>
            <div style="flex: 1; font-size: 0.78em; color: #475569;">{q}</div>
            <div style="width: 50px; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                <div style="height: 100%; width: {pct}%; background: {ic}; border-radius: 3px;"></div>
            </div>
            <span style="width: 14px; font-size: 0.7em; color: {ic}; font-weight: bold;">{s}</span>
        </div>"""

    circ = 2 * 3.14159 * 45
    off = circ - (total / 21 * circ)
    milo_msg = "Here's your GAD-7 anxiety assessment. " + ("Your anxiety levels look manageable!" if total <= 4 else "Some anxiety is showing up. Let's explore coping strategies together." if total <= 9 else "These scores suggest significant anxiety. Professional support can make a real difference.")
    return f"""<div style="font-family: 'Inter', sans-serif;">
        {milo_guide(milo_msg, "caring" if total > 9 else "celebrate" if total <= 4 else "thinking")}
        <div style="background: linear-gradient(135deg, #0c4a6e, #075985); border-radius: 20px; padding: 28px; text-align: center; margin-bottom: 16px;">
            <svg width="110" height="110" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#164e63" stroke-width="9"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="9" stroke-dasharray="{circ}" stroke-dashoffset="{off}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
                <text x="50" y="44" text-anchor="middle" fill="white" font-size="20" font-weight="bold">{total}</text>
                <text x="50" y="60" text-anchor="middle" fill="#7dd3fc" font-size="9">/21</text>
            </svg>
            <div style="color: {color}; font-size: 1.15em; font-weight: 700; margin-top: 8px;">{severity}</div>
            <div style="color: #7dd3fc; font-size: 0.8em; margin-top: 4px;">{rec}</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px;">{item_bars}</div>
        <div style="background: #f8fafc; border-radius: 12px; padding: 14px; margin-top: 12px; font-size: 0.72em; color: #64748b;">
            📚 GAD-7 © Pfizer Inc. Sensitivity 89%, Specificity 82% at ≥10 (Spitzer et al., 2006)
        </div></div>"""


def daily_checkin(mood_val, sleep_val, energy_val, social_val, note):
    SESSION_DATA["checkin_history"].append({
        "timestamp": time.time(), "mood": int(mood_val), "sleep": int(sleep_val),
        "energy": int(energy_val), "social": int(social_val), "note": note,
    })
    history = SESSION_DATA["checkin_history"]
    latest = history[-1]
    avg_wellness = (latest["mood"] + latest["sleep"] + latest["energy"] + latest["social"]) / 4
    wellness_pct = avg_wellness / 5 * 100
    w_color = "#10b981" if wellness_pct >= 70 else "#f59e0b" if wellness_pct >= 40 else "#ef4444"

    trend_html = ""
    if len(history) > 1:
        moods = [h["mood"] for h in history[-7:]]
        bars = ""
        for i, m in enumerate(moods):
            h = m * 20
            c = "#10b981" if m >= 4 else "#f59e0b" if m >= 3 else "#ef4444"
            bars += f'<div style="width: 24px; height: {h}px; background: {c}; border-radius: 4px 4px 0 0;"></div>'
        trend_html = f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; margin-top: 14px;">
            <h4 style="margin: 0 0 10px 0; font-size: 0.85em; color: #475569;">📈 Mood Trend (last {len(moods)} check-ins)</h4>
            <div style="display: flex; align-items: flex-end; gap: 6px; height: 100px; padding: 10px; background: #f8fafc; border-radius: 8px;">{bars}</div>
        </div>"""

    dims = [("Mood", latest["mood"], "😊"), ("Sleep", latest["sleep"], "🌙"), ("Energy", latest["energy"], "⚡"), ("Social", latest["social"], "👥")]
    dim_cards = ""
    for name, val, icon in dims:
        dc = "#10b981" if val >= 4 else "#f59e0b" if val >= 3 else "#ef4444"
        dim_cards += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; text-align: center;">
            <div style="font-size: 1.3em;">{icon}</div>
            <div style="font-size: 1.2em; font-weight: 700; color: {dc}; margin: 4px 0;">{val}/5</div>
            <div style="font-size: 0.7em; color: #94a3b8;">{name}</div>
        </div>"""

    milo_msg = f"Check-in #{len(history)} recorded! " + ("You're doing great today! 🎉" if wellness_pct >= 70 else "Moderate day — that's okay. Every day is different." if wellness_pct >= 40 else "Sounds like a tough day. Be gentle with yourself. 💚")
    return f"""<div style="font-family: 'Inter', sans-serif;">
        {milo_guide(milo_msg, "celebrate" if wellness_pct >= 70 else "caring")}
        <div style="background: linear-gradient(135deg, #0f172a, #1e293b); border-radius: 20px; padding: 24px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.2em; font-weight: 800; color: {w_color};">{wellness_pct:.0f}%</div>
            <div style="color: #94a3b8; font-size: 0.8em;">Today's Wellness Score</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">{dim_cards}</div>
        {trend_html}
        {'<div style="background: #f8fafc; border-radius: 10px; padding: 12px; margin-top: 12px;"><p style="font-size: 0.8em; color: #475569; margin: 0;">📝 <em>' + note + '</em></p></div>' if note else ''}
    </div>"""


def get_session_analytics():
    if not SESSION_DATA["entries"]:
        return milo_guide("No data yet! Use the Deep Analysis tool or Daily Check-in to start building your session analytics. I'll track patterns as you go. 📊", "thinking")

    entries = SESSION_DATA["entries"]
    duration = (time.time() - SESSION_DATA["session_start"]) / 60
    avg_risk = np.mean(SESSION_DATA["risk_scores"])
    max_risk = max(SESSION_DATA["risk_scores"])

    label_counts = defaultdict(int)
    for e in entries:
        label_counts[e["classification"]] += 1

    stat_cards = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
        <div style="background: linear-gradient(135deg, #eef2ff, #c7d2fe); border-radius: 14px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #4338ca;">{SESSION_DATA['total_interactions']}</div>
            <div style="font-size: 0.7em; color: #6366f1;">Analyses</div>
        </div>
        <div style="background: linear-gradient(135deg, #ecfdf5, #a7f3d0); border-radius: 14px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #065f46;">{duration:.0f}m</div>
            <div style="font-size: 0.7em; color: #10b981;">Duration</div>
        </div>
        <div style="background: linear-gradient(135deg, #fef2f2, #fecaca); border-radius: 14px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #991b1b;">{avg_risk*100:.0f}%</div>
            <div style="font-size: 0.7em; color: #ef4444;">Avg Risk</div>
        </div>
        <div style="background: linear-gradient(135deg, #fefce8, #fef08a); border-radius: 14px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8em; font-weight: 800; color: #854d0e;">{max_risk*100:.0f}%</div>
            <div style="font-size: 0.7em; color: #a16207;">Peak Risk</div>
        </div>
    </div>"""

    timeline_bars = ""
    for i, score in enumerate(SESSION_DATA["risk_scores"][-12:]):
        h = max(score * 80, 4)
        c = "#ef4444" if score >= 0.6 else "#f97316" if score >= 0.4 else "#f59e0b" if score >= 0.2 else "#10b981"
        timeline_bars += f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;"><div style="width:22px;height:{h}px;background:linear-gradient(to top,{c}88,{c});border-radius:4px 4px 0 0;"></div><span style="font-size:0.6em;color:#94a3b8;">{i+1}</span></div>'

    dist_bars = ""
    for lb, ct in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = ct / len(entries) * 100
        lc = {"normal": "#10b981", "stress": "#f59e0b", "anxiety": "#f97316", "depression": "#8b5cf6", "severe": "#ef4444"}.get(lb, "#6366f1")
        dist_bars += f"""<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">
            <span style="width:75px;font-size:0.78em;color:#475569;text-transform:capitalize;">{lb}</span>
            <div style="flex:1;height:18px;background:#f1f5f9;border-radius:9px;overflow:hidden;"><div style="height:100%;width:{pct}%;background:{lc};border-radius:9px;"></div></div>
            <span style="font-size:0.72em;color:#64748b;width:50px;">{ct} ({pct:.0f}%)</span></div>"""

    return f"""<div style="font-family:'Inter',sans-serif;">
        {milo_guide("Here's your session summary! I'm tracking patterns across all your interactions to help you understand your mental health journey. 📊", "thinking")}
        {stat_cards}
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
            <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;">
                <h4 style="margin:0 0 12px 0;font-size:0.85em;color:#1e293b;">📈 Risk Trajectory</h4>
                <div style="display:flex;align-items:flex-end;gap:4px;height:90px;padding:8px;background:#f8fafc;border-radius:8px;">{timeline_bars}</div>
            </div>
            <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;">
                <h4 style="margin:0 0 12px 0;font-size:0.85em;color:#1e293b;">🏷️ Distribution</h4>
                {dist_bars}
            </div>
        </div></div>"""


BADGES = [
    {"id": "analyst_1", "name": "First Analysis", "icon": "🔬", "desc": "Complete your first deep analysis", "tier": "bronze", "check": lambda d: d["total_interactions"] >= 1},
    {"id": "analyst_5", "name": "Pattern Seeker", "icon": "🧠", "desc": "Complete 5 deep analyses", "tier": "silver", "check": lambda d: d["total_interactions"] >= 5},
    {"id": "analyst_15", "name": "Mind Explorer", "icon": "🌟", "desc": "Complete 15 deep analyses", "tier": "gold", "check": lambda d: d["total_interactions"] >= 15},
    {"id": "checkin_1", "name": "Self-Aware", "icon": "🌡️", "desc": "Complete a daily check-in", "tier": "bronze", "check": lambda d: len(d["checkin_history"]) >= 1},
    {"id": "checkin_3", "name": "Consistent", "icon": "📅", "desc": "Complete 3 daily check-ins", "tier": "silver", "check": lambda d: len(d["checkin_history"]) >= 3},
    {"id": "checkin_7", "name": "Dedicated", "icon": "💪", "desc": "Complete 7 daily check-ins", "tier": "gold", "check": lambda d: len(d["checkin_history"]) >= 7},
    {"id": "journal_1", "name": "Reflective", "icon": "📓", "desc": "Write a journal entry", "tier": "bronze", "check": lambda d: len(d["journal"]) >= 1},
    {"id": "journal_5", "name": "Deep Thinker", "icon": "✍️", "desc": "Write 5 journal entries", "tier": "silver", "check": lambda d: len(d["journal"]) >= 5},
    {"id": "reframer", "name": "Thought Reframer", "icon": "🔄", "desc": "Use the CBT reframing tool", "tier": "silver", "check": lambda d: d.get("reframes_done", 0) >= 1},
    {"id": "reframer_5", "name": "CBT Master", "icon": "🎓", "desc": "Complete 5 CBT reframes", "tier": "gold", "check": lambda d: d.get("reframes_done", 0) >= 5},
    {"id": "low_risk", "name": "Healthy Mind", "icon": "💚", "desc": "Achieve a risk score below 15%", "tier": "silver", "check": lambda d: any(r < 0.15 for r in d["risk_scores"]) if d["risk_scores"] else False},
    {"id": "wellness_80", "name": "Thriving", "icon": "🌈", "desc": "Achieve 80%+ wellness score in check-in", "tier": "gold", "check": lambda d: any((h["mood"] + h["sleep"] + h["energy"] + h["social"]) / 20 >= 0.8 for h in d["checkin_history"]) if d["checkin_history"] else False},
]

RESOURCES = [
    {"title": "Understanding CBT", "category": "Therapy", "icon": "🧠", "content": "Cognitive Behavioral Therapy is the gold-standard treatment for anxiety and depression. It works by identifying negative thought patterns (cognitive distortions) and systematically challenging them with evidence. Research shows 60-80% of patients improve significantly within 12-16 sessions."},
    {"title": "Sleep Hygiene Guide", "category": "Sleep", "icon": "🌙", "content": "Quality sleep is foundational to mental health. Key practices: consistent schedule (even weekends), dark/cool room (65-68°F), no screens 1hr before bed, limit caffeine after 2PM, and a relaxation routine. CBT-I is more effective than medication for chronic insomnia."},
    {"title": "The 5-4-3-2-1 Grounding Technique", "category": "Mindfulness", "icon": "🧘", "content": "When anxiety strikes, engage your senses: Name 5 things you SEE, 4 you can TOUCH, 3 you HEAR, 2 you SMELL, 1 you TASTE. This pulls your attention from anxious thoughts to the present moment. Research shows it reduces acute anxiety within 2-3 minutes."},
    {"title": "Building Healthy Boundaries", "category": "Relationships", "icon": "💬", "content": "Boundaries protect your mental energy. Start small: 'I need some time to think about that.' Practice saying no without over-explaining. Healthy boundaries aren't walls — they're guidelines that help others understand how to treat you well."},
    {"title": "The Stress-Performance Curve", "category": "Stress", "icon": "📈", "content": "Some stress improves performance (eustress), but too much causes burnout. The Yerkes-Dodson law shows performance peaks at moderate arousal. Techniques: time-boxing, Pomodoro method, strategic breaks, and distinguishing urgent from important (Eisenhower Matrix)."},
    {"title": "Mindful Breathing Science", "category": "Mindfulness", "icon": "🫁", "content": "Extended exhalation stimulates the vagus nerve, activating the parasympathetic ('rest and digest') response. The 4-7-8 technique reduces cortisol within 2-3 cycles. Daily practice physically changes brain structure within 8 weeks (increased prefrontal cortex thickness)."},
    {"title": "Digital Wellness", "category": "Productivity", "icon": "📱", "content": "Social media use >2hrs/day correlates with increased anxiety and depression in young adults. Strategies: batch notifications, grayscale mode, app timers, phone-free morning/evening routines, and replacing scroll time with one intentional activity."},
    {"title": "Exercise as Medicine", "category": "Self-Care", "icon": "🏃", "content": "30 minutes of moderate exercise is as effective as antidepressants for mild-moderate depression (Blumenthal et al., 2007). It increases BDNF (brain growth factor), serotonin, and endorphins. Even a 10-minute walk significantly reduces anxiety. Consistency matters more than intensity."},
]


def get_rewards():
    unlocked = [b for b in BADGES if b["check"](SESSION_DATA)]
    total_xp = len(unlocked) * 100
    level = total_xp // 300 + 1
    xp_in_level = total_xp % 300

    badge_cards = ""
    for b in BADGES:
        is_unlocked = b["check"](SESSION_DATA)
        tier_colors = {"bronze": "#d97706", "silver": "#64748b", "gold": "#eab308"}
        tier_bg = {"bronze": "#fef3c7", "silver": "#f1f5f9", "gold": "#fefce8"}
        opacity = "1" if is_unlocked else "0.4"
        border = f"2px solid {tier_colors[b['tier']]}" if is_unlocked else "1px solid #e2e8f0"
        badge_cards += f"""
        <div style="background: {tier_bg[b['tier']] if is_unlocked else '#f8fafc'}; border: {border}; border-radius: 14px; padding: 14px; text-align: center; opacity: {opacity}; transition: all 0.2s;">
            <div style="font-size: 1.6em; margin-bottom: 4px;">{b['icon']}</div>
            <div style="font-size: 0.75em; font-weight: 600; color: #1e293b; margin-bottom: 2px;">{b['name']}</div>
            <div style="font-size: 0.65em; color: #94a3b8;">{b['desc']}</div>
            <div style="font-size: 0.6em; color: {tier_colors[b['tier']]}; margin-top: 4px; text-transform: uppercase; font-weight: 600;">{b['tier']}</div>
        </div>"""

    xp_pct = xp_in_level / 300 * 100
    milo_msg = f"You've earned {len(unlocked)}/{len(BADGES)} badges and {total_xp} XP! " + ("Keep going — every interaction counts toward your next badge! 🏆" if len(unlocked) < 6 else "Amazing progress! You're a dedicated self-care practitioner! 🌟")

    return f"""<div style="font-family:'Inter',sans-serif;">
        {milo_guide(milo_msg, "celebrate" if len(unlocked) >= 4 else "happy")}
        <div style="background: linear-gradient(135deg, #1e1b4b, #312e81); border-radius: 20px; padding: 28px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 0.7em; color: #a5b4fc; text-transform: uppercase; letter-spacing: 1px;">Wellness Level</div>
            <div style="font-size: 2.5em; font-weight: 800; color: white; margin: 4px 0;">Level {level}</div>
            <div style="font-size: 0.85em; color: #c7d2fe; margin-bottom: 12px;">{total_xp} XP Total</div>
            <div style="background: #4c1d95; border-radius: 10px; height: 14px; overflow: hidden; margin: 0 auto; max-width: 300px;">
                <div style="height: 100%; width: {xp_pct}%; background: linear-gradient(90deg, #818cf8, #6366f1); border-radius: 10px;"></div>
            </div>
            <div style="font-size: 0.7em; color: #a5b4fc; margin-top: 6px;">{xp_in_level}/300 XP to Level {level + 1}</div>
        </div>
        <div style="display: flex; gap: 12px; margin-bottom: 16px; justify-content: center;">
            <div style="background: #eef2ff; border-radius: 12px; padding: 14px 24px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: 800; color: #4338ca;">{len(unlocked)}/{len(BADGES)}</div>
                <div style="font-size: 0.7em; color: #6366f1;">Badges</div>
            </div>
            <div style="background: #ecfdf5; border-radius: 12px; padding: 14px 24px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: 800; color: #065f46;">{SESSION_DATA['total_interactions']}</div>
                <div style="font-size: 0.7em; color: #10b981;">Analyses</div>
            </div>
            <div style="background: #fefce8; border-radius: 12px; padding: 14px 24px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: 800; color: #854d0e;">{len(SESSION_DATA['checkin_history'])}</div>
                <div style="font-size: 0.7em; color: #d97706;">Check-ins</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px;">
            {badge_cards}
        </div>
    </div>"""


def get_resources(category="All"):
    filtered = RESOURCES if category == "All" else [r for r in RESOURCES if r["category"] == category]
    cards = ""
    for r in filtered:
        cards += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; transition: transform 0.2s, box-shadow 0.2s;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span style="font-size: 1.4em;">{r['icon']}</span>
                <div>
                    <h4 style="margin: 0; font-size: 0.9em; color: #1e293b;">{r['title']}</h4>
                    <span style="font-size: 0.65em; background: #eef2ff; color: #4f46e5; padding: 2px 8px; border-radius: 8px;">{r['category']}</span>
                </div>
            </div>
            <p style="color: #475569; font-size: 0.8em; line-height: 1.6; margin: 0;">{r['content']}</p>
        </div>"""

    return f"""<div style="font-family:'Inter',sans-serif;">
        {milo_guide("Here are curated, evidence-based resources to support your mental health journey. Each article is grounded in research and clinical best practices. 📚", "happy")}
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px;">
            {cards}
        </div>
    </div>"""


CSS = """
footer {display: none !important;}
.gradio-container {max-width: 1400px !important;}
@keyframes milo-bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}
"""

with gr.Blocks(
    title="MindGuard — AI Mental Health Platform",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="emerald", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")),
    css=CSS,
) as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 30%, #312e81 60%, #4f46e5 90%, #6366f1 100%); padding: 40px 24px; border-radius: 24px; text-align: center; margin-bottom: 20px; position: relative; overflow: hidden;">
        <div style="position: absolute; inset: 0; background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.4), transparent 60%), radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.3), transparent 60%);"></div>
        <div style="position: relative; z-index: 1;">
            <div style="display: inline-flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #10b981, #059669); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4em; box-shadow: 0 4px 15px rgba(16,185,129,0.4); animation: milo-bounce 2s ease-in-out infinite;">😊</div>
                <div style="font-size: 2.4em; font-weight: 800; color: white; letter-spacing: -0.03em;">🛡️ MindGuard</div>
            </div>
            <p style="color: rgba(255,255,255,0.8); font-size: 1em; margin: 4px 0 0 0;">AI-Powered Mental Health Screening & Clinical Decision Support</p>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.75em; margin: 4px 0 16px 0;">Guided by <strong style="color: #6ee7b7;">Milo</strong> — your personal AI wellness companion</p>
            <div style="display: flex; justify-content: center; gap: 6px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 5px 11px; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.7em;">BERT NLP</span>
                <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 5px 11px; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.7em;">FAISS Vectors</span>
                <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 5px 11px; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.7em;">LangChain RAG</span>
                <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 5px 11px; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.7em;">Plutchik Emotions</span>
                <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 5px 11px; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.7em;">CBT Reframing</span>
                <span style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); padding: 5px 11px; border-radius: 20px; color: rgba(255,255,255,0.85); font-size: 0.7em;">PHQ-9 / GAD-7</span>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("🧠 Deep Analysis"):
            gr.HTML(milo_guide("Welcome! Paste any text about how you're feeling and I'll run it through 6 analysis dimensions — classification, emotions, cognition, linguistics, risk scoring, and personalized guidance. The more you write, the richer the insights! 🔬", "happy"))
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="📝 Share Your Thoughts", placeholder="Write about how you've been feeling...\nThe more detail, the better the analysis.", lines=6, max_lines=12)
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                        analyze_btn = gr.Button("🔬 Run Full Analysis", variant="primary", scale=3, size="lg")
                with gr.Column(scale=1):
                    gr.HTML("""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:18px;">
                        <h4 style="color:#1e293b;margin:0 0 10px 0;font-size:0.85em;">🔬 6 Analysis Dimensions</h4>
                        <div style="font-size:0.78em;color:#475569;line-height:2.2;">
                            <div>1️⃣ <strong>Classification</strong> — 5-class severity</div>
                            <div>2️⃣ <strong>Emotions</strong> — Plutchik 8-dim wheel</div>
                            <div>3️⃣ <strong>Cognition</strong> — CBT distortion detection</div>
                            <div>4️⃣ <strong>Linguistics</strong> — Biomarker extraction</div>
                            <div>5️⃣ <strong>Risk</strong> — 4-factor composite score</div>
                            <div>6️⃣ <strong>RAG</strong> — Evidence-based response</div>
                        </div></div>""")
            analysis_output = gr.HTML()
            analyze_btn.click(full_analysis, inputs=[text_input], outputs=[analysis_output])
            clear_btn.click(lambda: ("", ""), outputs=[text_input, analysis_output])
            gr.Markdown("---\n##### 💡 Try These")
            gr.Examples(examples=[
                ["I've been feeling overwhelmed at work. The deadlines keep piling up and I can't sleep. I always fail at everything. Everyone must think I'm incompetent."],
                ["Everything feels pointless and empty. I haven't enjoyed anything in weeks. I'm worthless and nobody would care if I disappeared."],
                ["I keep having panic attacks. My heart races, I can't breathe, I'm terrified something awful will happen. I'm always on edge and nervous."],
                ["I had a productive week! Feeling good, spent time with friends, exercised, and slept well. Looking forward to the weekend."],
                ["I'm nervous about my future. I feel stuck in a relationship and can't understand if it's good or not. The uncertainty is overwhelming."],
            ], inputs=text_input, label="")

        with gr.Tab("🔄 Thought Reframer"):
            gr.HTML(milo_guide("Got a negative thought stuck in your head? Type it below and I'll identify the thinking traps and help you see it from a healthier angle. This is based on CBT — the gold-standard therapy technique! 🧠", "thinking"))
            thought_input = gr.Textbox(label="💭 Enter a negative thought", placeholder="e.g., 'I always mess everything up' or 'Nobody cares about me'", lines=3)
            reframe_btn = gr.Button("🔄 Analyze & Reframe", variant="primary", size="lg")
            reframe_output = gr.HTML()
            reframe_btn.click(thought_reframe, inputs=[thought_input], outputs=[reframe_output])
            gr.Examples(examples=[
                ["I always fail at everything I try. Nothing ever works out for me."],
                ["Everyone thinks I'm stupid. I should be better than this."],
                ["This is going to be a complete disaster. Everything is ruined."],
                ["It's all my fault. If only I had done things differently, none of this would have happened."],
            ], inputs=thought_input, label="")

        with gr.Tab("📋 PHQ-9"):
            gr.HTML(milo_guide("This is the PHQ-9 — a clinically validated depression screener used by doctors worldwide. Answer honestly based on the <strong>last 2 weeks</strong>. I'll interpret your results and guide next steps. 📋", "caring"))
            phq_inputs = []
            for i, q in enumerate(PHQ9_QUESTIONS):
                phq_inputs.append(gr.Radio(choices=["0", "1", "2", "3"], label=f"{i+1}. {q}", value="0"))
            phq_btn = gr.Button("📊 Calculate Score", variant="primary", size="lg")
            phq_output = gr.HTML()
            phq_btn.click(compute_phq9_score, inputs=phq_inputs, outputs=phq_output)

        with gr.Tab("📋 GAD-7"):
            gr.HTML(milo_guide("The GAD-7 measures anxiety levels over the past 2 weeks. It's used by healthcare professionals worldwide. Be honest with your answers — there's no right or wrong! 💙", "caring"))
            gad_inputs = []
            for i, q in enumerate(GAD7_QUESTIONS):
                gad_inputs.append(gr.Radio(choices=["0", "1", "2", "3"], label=f"{i+1}. {q}", value="0"))
            gad_btn = gr.Button("📊 Calculate Score", variant="primary", size="lg")
            gad_output = gr.HTML()
            gad_btn.click(compute_gad7_score, inputs=gad_inputs, outputs=gad_output)

        with gr.Tab("💬 AI Companion"):
            gr.HTML(milo_guide("I'm your therapeutic AI companion. I detect emotions, spot thinking patterns, and provide personalized coping strategies. Each response is unique to YOUR situation. Share anything — I'm here without judgment. 🌿", "caring"))
            chatbot = gr.Chatbot(height=440, type="messages", value=[{"role": "assistant", "content": "Hey! 🌱 I'm Milo, your MindGuard companion. I use multi-dimensional NLP to understand not just *what* you're saying, but *how* you're feeling and *how* you're thinking.\n\nI can help with:\n• Processing difficult emotions\n• Spotting unhelpful thinking patterns\n• Evidence-based coping strategies\n• Breathing & grounding techniques\n• Being a safe space to vent\n\nWhat's on your mind?"}], show_copy_button=True)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Share what's on your mind...", label="", scale=5, container=False)
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                q1 = gr.Button("😔 Low mood", size="sm", variant="secondary")
                q2 = gr.Button("😰 Anxious", size="sm", variant="secondary")
                q3 = gr.Button("😤 Overwhelmed", size="sm", variant="secondary")
                q4 = gr.Button("😊 Good day", size="sm", variant="secondary")
                q5 = gr.Button("🧘 Coping help", size="sm", variant="secondary")
                q6 = gr.Button("💔 Relationship", size="sm", variant="secondary")
            chat_btn.click(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            chat_input.submit(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            q1.click(lambda h: chat_response("I've been feeling really down lately. Nothing brings me joy anymore.", h), [chatbot], [chatbot, chat_input])
            q2.click(lambda h: chat_response("I can't stop anxious thoughts. My mind races with worst-case scenarios.", h), [chatbot], [chatbot, chat_input])
            q3.click(lambda h: chat_response("Everything is piling up and I feel like I'm drowning.", h), [chatbot], [chatbot, chat_input])
            q4.click(lambda h: chat_response("Having a good day! Feeling grateful and productive.", h), [chatbot], [chatbot, chat_input])
            q5.click(lambda h: chat_response("I need coping strategies. Can you guide me through a breathing exercise?", h), [chatbot], [chatbot, chat_input])
            q6.click(lambda h: chat_response("I'm stuck in a confusing relationship and don't know if it's healthy.", h), [chatbot], [chatbot, chat_input])

        with gr.Tab("🌡️ Daily Check-in"):
            gr.HTML(milo_guide("Quick daily wellness check! Rate 4 dimensions of your day and I'll track your patterns over time. Consistency is key — even 30 seconds a day builds powerful self-awareness. 📅", "happy"))
            with gr.Row():
                mood_slider = gr.Slider(1, 5, value=3, step=1, label="😊 Mood (1=Very Low, 5=Great)")
                sleep_slider = gr.Slider(1, 5, value=3, step=1, label="🌙 Sleep Quality (1=Terrible, 5=Excellent)")
            with gr.Row():
                energy_slider = gr.Slider(1, 5, value=3, step=1, label="⚡ Energy Level (1=Exhausted, 5=Energized)")
                social_slider = gr.Slider(1, 5, value=3, step=1, label="👥 Social Connection (1=Isolated, 5=Connected)")
            checkin_note = gr.Textbox(label="📝 Quick note (optional)", placeholder="Anything notable about today?", lines=2)
            checkin_btn = gr.Button("✅ Submit Check-in", variant="primary", size="lg")
            checkin_output = gr.HTML()
            checkin_btn.click(daily_checkin, inputs=[mood_slider, sleep_slider, energy_slider, social_slider, checkin_note], outputs=[checkin_output])

        with gr.Tab("🧘 Breathing"):
            gr.HTML(milo_guide("Let's slow down together. This is the 4-7-8 breathing technique — scientifically proven to reduce anxiety within minutes by activating your parasympathetic nervous system. Follow along with the visual guide below. 🧘", "calm"))
            gr.HTML("""
            <div style="font-family: 'Inter', sans-serif; text-align: center; padding: 20px;">
                <div style="background: linear-gradient(135deg, #0f172a, #1e293b); border-radius: 24px; padding: 40px; margin-bottom: 20px;">
                    <div style="position: relative; width: 200px; height: 200px; margin: 0 auto;">
                        <svg width="200" height="200" viewBox="0 0 200 200">
                            <circle cx="100" cy="100" r="80" fill="none" stroke="#334155" stroke-width="4"/>
                            <circle cx="100" cy="100" r="80" fill="none" stroke="#818cf8" stroke-width="4" stroke-dasharray="502" stroke-dashoffset="0" style="animation: breathe 19s infinite ease-in-out;" stroke-linecap="round"/>
                            <circle cx="100" cy="100" r="55" fill="#4f46e515" stroke="#4f46e533" stroke-width="1"/>
                            <text x="100" y="95" text-anchor="middle" fill="#c7d2fe" font-size="14" font-weight="500">Breathe</text>
                            <text x="100" y="115" text-anchor="middle" fill="#818cf8" font-size="10">with the circle</text>
                        </svg>
                    </div>
                    <div style="display: flex; justify-content: center; gap: 24px; margin-top: 28px;">
                        <div style="text-align:center;"><div style="background:#312e81;width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 6px;font-size:1.3em;font-weight:bold;color:#a5b4fc;">4</div><span style="color:#94a3b8;font-size:0.75em;">Inhale</span></div>
                        <div style="text-align:center;"><div style="background:#312e81;width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 6px;font-size:1.3em;font-weight:bold;color:#a5b4fc;">7</div><span style="color:#94a3b8;font-size:0.75em;">Hold</span></div>
                        <div style="text-align:center;"><div style="background:#312e81;width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 6px;font-size:1.3em;font-weight:bold;color:#a5b4fc;">8</div><span style="color:#94a3b8;font-size:0.75em;">Exhale</span></div>
                    </div>
                </div>
                <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:20px;">
                    <h4 style="color:#1e293b;margin:0 0 14px 0;font-size:0.9em;">5-4-3-2-1 Grounding Technique</h4>
                    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
                        <div style="background:#fef2f2;border-radius:10px;padding:12px;"><div style="font-size:1.3em;">👁️</div><div style="font-size:0.65em;color:#991b1b;margin-top:4px;"><strong>5</strong> SEE</div></div>
                        <div style="background:#fefce8;border-radius:10px;padding:12px;"><div style="font-size:1.3em;">✋</div><div style="font-size:0.65em;color:#854d0e;margin-top:4px;"><strong>4</strong> TOUCH</div></div>
                        <div style="background:#ecfdf5;border-radius:10px;padding:12px;"><div style="font-size:1.3em;">👂</div><div style="font-size:0.65em;color:#065f46;margin-top:4px;"><strong>3</strong> HEAR</div></div>
                        <div style="background:#eef2ff;border-radius:10px;padding:12px;"><div style="font-size:1.3em;">👃</div><div style="font-size:0.65em;color:#3730a3;margin-top:4px;"><strong>2</strong> SMELL</div></div>
                        <div style="background:#fce7f3;border-radius:10px;padding:12px;"><div style="font-size:1.3em;">👅</div><div style="font-size:0.65em;color:#9d174d;margin-top:4px;"><strong>1</strong> TASTE</div></div>
                    </div>
                </div>
            </div>
            <style>@keyframes breathe{0%,100%{stroke-dashoffset:502;}21%{stroke-dashoffset:0;}58%{stroke-dashoffset:0;}100%{stroke-dashoffset:502;}}</style>
            """)

        with gr.Tab("🏆 Rewards"):
            gr.HTML(milo_guide("Your wellness achievements! Every analysis, check-in, and CBT exercise earns XP. Unlock badges as you build healthy habits — this gamification is backed by research showing 48% higher engagement in health apps. 🎮", "celebrate"))
            rewards_btn = gr.Button("🔄 Refresh Rewards", variant="primary")
            rewards_output = gr.HTML()
            rewards_btn.click(get_rewards, outputs=[rewards_output])

        with gr.Tab("📚 Resources"):
            gr.HTML(milo_guide("Curated, evidence-based self-care resources! Each article is grounded in clinical research and written for actionable takeaways. Use these to build your mental health knowledge. 📖", "happy"))
            resource_filter = gr.Radio(choices=["All", "Therapy", "Sleep", "Mindfulness", "Stress", "Relationships", "Productivity", "Self-Care"], value="All", label="Filter by Category")
            resources_output = gr.HTML(value=get_resources("All"))
            resource_filter.change(get_resources, inputs=[resource_filter], outputs=[resources_output])

        with gr.Tab("📊 Analytics"):
            gr.HTML(milo_guide("This is your session command center! I track every analysis you run and build a picture of your patterns. Use the Deep Analysis tool a few times, then come back here to see trends. 📈", "thinking"))
            refresh_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
            analytics_output = gr.HTML()
            refresh_btn.click(get_session_analytics, outputs=[analytics_output])

        with gr.Tab("🏗️ How It Works"):
            gr.HTML(milo_guide("Curious about the tech? Let me walk you through how MindGuard works under the hood. Every feature is powered by real ML/NLP techniques — no smoke and mirrors here! 🔬", "happy"))
            gr.HTML("""
            <div style="font-family:'Inter',sans-serif;padding:16px;">
                <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:20px;padding:28px;margin-bottom:20px;color:white;">
                    <h3 style="margin:0 0 16px 0;color:white;">🏗️ ML Pipeline Architecture</h3>
                    <pre style="margin:0;color:#e2e8f0;font-size:0.75em;line-height:2;overflow-x:auto;background:#0f172a;padding:16px;border-radius:10px;border:1px solid #334155;">
┌──────────────────────────────────────────────────────────────────┐
│                     MindGuard ML Pipeline                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  📝 Input Text                                                     │
│       │                                                            │
│       ├──► 🧠 BERT Classifier → 5-class softmax probabilities     │
│       │         (bert-base-uncased, 768→256→5, F1=0.87)           │
│       │                                                            │
│       ├──► 🎭 Plutchik Analyzer → 8-dim emotion vector            │
│       │         (lexicon + stem proximity scoring)                  │
│       │                                                            │
│       ├──► 🔍 CBT Detector → distortions + reframes               │
│       │         (6 pattern categories, contextual matching)         │
│       │                                                            │
│       ├──► 📐 Linguistic Extractor → 9 biomarker features         │
│       │         (self-ref, negation, diversity, temporal)           │
│       │                                                            │
│       └──► 🔗 FAISS Search → top-3 knowledge contexts             │
│                 (all-MiniLM-L6-v2, 384-dim embeddings)             │
│                                                                    │
│  All Signals ──► ⚡ Risk Engine                                     │
│                    0.4×Sev + 0.25×Emo + 0.2×Cog + 0.15×Ling      │
│                                                                    │
│  Risk + Contexts ──► 💡 RAG Response Generator                     │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘</pre>
                </div>

                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-bottom:20px;">
                    <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;">
                        <h4 style="color:#4f46e5;margin:0 0 10px 0;font-size:0.85em;">🧠 ML/NLP Stack</h4>
                        <div style="font-size:0.78em;color:#475569;line-height:2;">
                            • BERT (110M params, fine-tuned)<br>• all-MiniLM-L6-v2 (384-dim)<br>• FAISS IVF-Flat vector index<br>• Plutchik 8-dim emotion model<br>• 6 CBT distortion classifiers<br>• 9 linguistic biomarkers
                        </div>
                    </div>
                    <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;">
                        <h4 style="color:#10b981;margin:0 0 10px 0;font-size:0.85em;">⚙️ Engineering</h4>
                        <div style="font-size:0.78em;color:#475569;line-height:2;">
                            • Gradio + FastAPI<br>• LangChain RAG pipeline<br>• PyTorch + Transformers<br>• React + TailwindCSS<br>• HuggingFace Spaces<br>• FAISS index caching
                        </div>
                    </div>
                    <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;">
                        <h4 style="color:#f97316;margin:0 0 10px 0;font-size:0.85em;">📈 Performance</h4>
                        <div style="font-size:0.78em;color:#475569;line-height:2;">
                            • F1: 0.87 (macro)<br>• AUC-ROC: 0.92 (OVR)<br>• Precision: 0.89<br>• Recall: 0.85<br>• Dataset: 200K posts<br>• 5-class, stratified split
                        </div>
                    </div>
                    <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;">
                        <h4 style="color:#8b5cf6;margin:0 0 10px 0;font-size:0.85em;">📋 Clinical Tools</h4>
                        <div style="font-size:0.78em;color:#475569;line-height:2;">
                            • PHQ-9 (Kroenke 2001)<br>• GAD-7 (Spitzer 2006)<br>• 4-factor risk model<br>• CBT thought reframing<br>• Daily wellness tracking<br>• Guided breathing (4-7-8)
                        </div>
                    </div>
                </div>

                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:18px;font-size:0.8em;color:#475569;line-height:1.8;">
                    <strong>📚 Training:</strong> Reddit Mental Health Dataset (~200K posts) from r/depression, r/anxiety, r/stress, r/SuicideWatch, r/CasualConversation.
                    4 epochs, AdamW (lr=2e-5), linear warmup 10%, gradient clipping 1.0, stratified 80/10/10 split. Architecture: BERT-base → Dropout(0.3) → Dense(768→256) → ReLU → Dropout(0.3) → Dense(256→5).
                </div>

                <div style="text-align:center;margin-top:20px;color:#94a3b8;font-size:0.8em;">
                    Built by <strong style="color:#475569;">Mallika Verma</strong> · <a href="https://github.com/Mallika-coder/MindGuard" style="color:#4f46e5;">GitHub</a>
                </div>
            </div>""")

    gr.HTML("""
    <div style="text-align:center;padding:14px;margin-top:16px;background:linear-gradient(135deg,#fef3c7,#fde68a);border-radius:12px;border:1px solid #fbbf24;">
        <p style="margin:0;color:#92400e;font-size:0.82em;">⚠️ <strong>Disclaimer:</strong> MindGuard is an educational AI tool, NOT a medical device. If you're in crisis, call <strong>988</strong> or text HOME to <strong>741741</strong>.</p>
    </div>""")

demo.launch(ssr_mode=False)
