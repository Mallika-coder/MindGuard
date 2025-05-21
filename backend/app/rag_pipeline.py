import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


KNOWLEDGE_BASE = [
    {
        "topic": "depression_support",
        "content": """Depression is a common mental health condition affecting over 280 million people worldwide.
        Key signs include persistent sadness, loss of interest, fatigue, and changes in sleep patterns.
        Evidence-based approaches include Cognitive Behavioral Therapy (CBT), regular physical activity,
        maintaining social connections, and establishing daily routines. If symptoms persist for more than
        two weeks, professional help is strongly recommended. Remember: depression is treatable, and
        seeking help is a sign of strength, not weakness.""",
    },
    {
        "topic": "anxiety_management",
        "content": """Anxiety disorders are the most common mental health conditions, affecting 301 million people.
        Symptoms include excessive worry, restlessness, difficulty concentrating, and physical tension.
        Effective coping strategies include deep breathing exercises (4-7-8 technique), progressive muscle
        relaxation, mindfulness meditation, and grounding techniques (5-4-3-2-1 method). Regular exercise,
        limiting caffeine, and maintaining a consistent sleep schedule can significantly reduce symptoms.
        Professional therapy, particularly CBT and exposure therapy, shows high success rates.""",
    },
    {
        "topic": "stress_reduction",
        "content": """Chronic stress affects physical and mental health, contributing to cardiovascular disease,
        weakened immunity, and mental health disorders. The body's stress response (fight-or-flight) was
        designed for short-term threats, not prolonged activation. Effective stress management includes
        time management, boundary setting, regular breaks, nature exposure, and social support.
        The Pomodoro Technique, journaling, and creative activities are proven stress reducers.
        If stress is overwhelming daily functioning, professional support can help develop personalized strategies.""",
    },
    {
        "topic": "crisis_resources",
        "content": """If you or someone you know is in immediate danger, please contact emergency services (911).
        National Suicide Prevention Lifeline: 988 (call or text). Crisis Text Line: Text HOME to 741741.
        International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
        Remember: a crisis is temporary, and help is available 24/7. You are not alone, and your life matters.
        Reaching out is the bravest thing you can do.""",
    },
    {
        "topic": "sleep_hygiene",
        "content": """Poor sleep is both a symptom and contributor to mental health issues. Adults need 7-9 hours
        of quality sleep. Sleep hygiene practices include: maintaining a consistent schedule, keeping the
        bedroom dark and cool (65-68°F), avoiding screens 1 hour before bed, limiting caffeine after 2 PM,
        and creating a relaxing bedtime routine. Cognitive Behavioral Therapy for Insomnia (CBT-I) is the
        first-line treatment for chronic insomnia and is more effective long-term than medication.""",
    },
    {
        "topic": "mindfulness_practices",
        "content": """Mindfulness meditation has strong scientific evidence for reducing anxiety, depression, and
        stress. Start with 5 minutes daily and gradually increase. Key practices include body scan meditation,
        loving-kindness meditation, and mindful breathing. Apps like Headspace and Calm can guide beginners.
        Research shows 8 weeks of regular practice can physically change brain structure, increasing gray
        matter in areas associated with emotional regulation and decreasing amygdala reactivity.""",
    },
    {
        "topic": "professional_help",
        "content": """Seeking professional help is appropriate when symptoms interfere with daily life, relationships,
        or work for more than two weeks. Types of mental health professionals include psychiatrists (can prescribe
        medication), psychologists (therapy and testing), licensed counselors, and social workers. Many offer
        telehealth options. Insurance often covers mental health services. Community mental health centers
        provide sliding-scale fees. Employee Assistance Programs (EAPs) offer free short-term counseling.""",
    },
    {
        "topic": "self_care_strategies",
        "content": """Self-care is not selfish — it's essential for mental health maintenance. The pillars include:
        physical (exercise, nutrition, sleep), emotional (journaling, therapy, boundaries), social (meaningful
        connections, community), and spiritual (purpose, values, mindfulness). Start small: one 10-minute walk,
        one healthy meal, one conversation with a friend. Build gradually. Track what works for you.
        Self-care looks different for everyone — find what recharges YOUR specific batteries.""",
    },
]


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self):
        documents = [
            Document(page_content=item["content"], metadata={"topic": item["topic"]})
            for item in KNOWLEDGE_BASE
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store

    def retrieve_context(self, query: str, k: int = 3) -> list[dict]:
        results = self.vector_store.similarity_search_with_score(query, k=k)
        contexts = []
        for doc, score in results:
            contexts.append({
                "content": doc.page_content,
                "topic": doc.metadata.get("topic", "general"),
                "relevance_score": round(1 - score, 4),
            })
        return contexts

    def generate_response(self, user_text: str, classification: dict) -> str:
        contexts = self.retrieve_context(user_text)
        context_text = "\n".join([c["content"] for c in contexts])

        label = classification["label"]
        severity = classification["severity_score"]

        if severity >= 0.75:
            tone = "urgent and compassionate"
            action = "strongly recommend seeking professional help immediately"
        elif severity >= 0.5:
            tone = "warm and supportive"
            action = "suggest considering professional support alongside self-help strategies"
        elif severity >= 0.25:
            tone = "encouraging and informative"
            action = "share evidence-based coping strategies"
        else:
            tone = "positive and affirming"
            action = "reinforce healthy habits and resilience"

        response = self._compose_response(
            user_text, label, severity, tone, action, context_text
        )
        return response

    def _compose_response(self, user_text, label, severity, tone, action, context):
        if label == "severe":
            return (
                "I hear you, and I want you to know that what you're feeling matters deeply. "
                "Based on what you've shared, I'm concerned about your wellbeing. "
                "Please reach out to a crisis helpline — you can call or text 988 (Suicide & Crisis Lifeline) "
                "or text HOME to 741741 (Crisis Text Line). You don't have to face this alone. "
                "A trained counselor is available 24/7 and can provide immediate support. "
                "Your life has value, and help is available right now."
            )

        if label == "depression":
            return (
                "Thank you for sharing how you're feeling. What you're experiencing sounds really difficult, "
                "and it takes courage to express it. Depression can make everything feel heavier, "
                "but please know that it is treatable and things can get better. "
                f"Based on current research: {context[:300]} "
                "Consider reaching out to a mental health professional who can provide personalized support. "
                "In the meantime, small steps like a short walk, connecting with someone you trust, "
                "or maintaining a simple routine can help. You deserve support."
            )

        if label == "anxiety":
            return (
                "I can sense the worry in your words, and I want you to know that anxiety is "
                "your brain's way of trying to protect you — even when it overreacts. "
                "Let's try something together: take a slow breath in for 4 counts, "
                "hold for 7, and exhale for 8. "
                f"Here's what research shows helps: {context[:300]} "
                "Grounding techniques like the 5-4-3-2-1 method can bring you back to the present moment. "
                "If anxiety is affecting your daily life, a therapist specializing in CBT can be very effective."
            )

        if label == "stress":
            return (
                "It sounds like you're carrying a heavy load right now. "
                "Stress is your body's signal that something needs attention. "
                f"Evidence-based strategies that may help: {context[:300]} "
                "Remember that it's okay to set boundaries and ask for help. "
                "Try breaking overwhelming tasks into smaller steps, and give yourself permission to rest. "
                "If stress is becoming chronic, talking to a professional can help you develop "
                "personalized coping strategies."
            )

        return (
            "It's great that you're checking in with yourself — self-awareness is a powerful tool "
            "for maintaining mental wellness. Keep nurturing your mental health through regular "
            "self-care, meaningful connections, and activities that bring you joy. "
            f"Some tips to maintain your wellbeing: {context[:300]} "
            "Remember, taking care of your mental health isn't just for difficult times — "
            "it's an ongoing practice that builds resilience."
        )
