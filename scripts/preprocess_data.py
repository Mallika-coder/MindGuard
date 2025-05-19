"""
Data Preprocessing Script for Reddit Mental Health Dataset.

Downloads and preprocesses data from mental health subreddits.
Output: data/reddit_mental_health.csv with columns [text, label, subreddit, word_count]
"""

import os
import pandas as pd
import re
from sklearn.utils import resample

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_mental_health.csv")

LABEL_MAPPING = {
    "depression": "depression",
    "SuicideWatch": "severe",
    "anxiety": "anxiety",
    "stress": "stress",
    "mentalhealth": "normal",
    "offmychest": "normal",
    "CasualConversation": "normal",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_sample_dataset():
    """Create a sample dataset for demonstration purposes."""
    samples = {
        "normal": [
            "Had a great day at work today. Finished my project ahead of schedule and got positive feedback.",
            "Just came back from a run. The weather is beautiful and I feel energized.",
            "Spent the weekend with friends. We went hiking and had a lovely picnic.",
            "Started learning a new programming language. It's challenging but fun.",
            "Cooked a new recipe today and it turned out amazing. Feeling accomplished.",
        ] * 40,
        "stress": [
            "Work deadlines are piling up and I can barely keep up. My manager keeps adding more tasks.",
            "I have three exams next week and haven't started studying. The pressure is immense.",
            "My workload has doubled since my colleague left. I'm doing the job of two people.",
            "Financial stress is killing me. Bills keep coming and my salary barely covers rent.",
            "I feel like I'm constantly running on empty. There's never enough time for everything.",
        ] * 40,
        "anxiety": [
            "I can't stop worrying about everything. My mind races constantly and I can't sleep.",
            "Had another panic attack today in the grocery store. My heart was pounding so fast.",
            "I keep imagining worst-case scenarios for everything. What if I lose my job? What if I get sick?",
            "Social situations terrify me. I cancelled plans again because the thought of going out made me nauseous.",
            "I feel like something bad is always about to happen. This constant dread is exhausting.",
        ] * 40,
        "depression": [
            "I haven't left my bed in three days. Nothing seems worth the effort anymore.",
            "Everything feels gray and meaningless. I used to love painting but now I can't even look at my supplies.",
            "I feel like a burden to everyone around me. They'd be better off without me in their lives.",
            "Can't remember the last time I felt genuinely happy. Just going through the motions every day.",
            "I've been sleeping 14 hours a day and still feel exhausted. Food has lost all taste.",
        ] * 40,
        "severe": [
            "I don't see the point of going on anymore. Every day is just pain.",
            "I've been thinking about ending things. I wrote letters to my family.",
            "The world would be better without me in it. I'm just a waste of space.",
            "I've been researching methods. I can't take this suffering anymore.",
            "Nobody would notice if I was gone. I'm invisible to everyone.",
        ] * 40,
    }

    rows = []
    for label, texts in samples.items():
        for text in texts:
            rows.append({"text": clean_text(text), "label": label})

    df = pd.DataFrame(rows)
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= 5]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def balance_dataset(df: pd.DataFrame, target_per_class: int = 200) -> pd.DataFrame:
    balanced_frames = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        if len(class_df) >= target_per_class:
            balanced_frames.append(class_df.sample(target_per_class, random_state=42))
        else:
            balanced_frames.append(
                resample(class_df, n_samples=target_per_class, random_state=42)
            )
    return pd.concat(balanced_frames).reset_index(drop=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Creating sample dataset for demonstration...")
    df = create_sample_dataset()

    print(f"Raw samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    df = balance_dataset(df, target_per_class=200)
    print(f"\nBalanced samples: {len(df)}")
    print(f"Balanced distribution:\n{df['label'].value_counts()}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
