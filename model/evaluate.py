"""Standalone evaluation script for the trained model."""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from config import MODEL_SAVE_DIR, BATCH_SIZE, NUM_CLASSES, LABELS, LABEL2ID, DATA_DIR
from dataset import MentalHealthDataset
from classifier import MentalHealthClassifier
from transformers import BertTokenizer


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MentalHealthClassifier()
    model.load_state_dict(
        torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pt"), map_location=device)
    )
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_SAVE_DIR, "tokenizer"))

    df = pd.read_csv(os.path.join(DATA_DIR, "reddit_mental_health.csv"))
    df["label_id"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    test_df = df.sample(frac=0.2, random_state=42)
    test_dataset = MentalHealthDataset(
        test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    f1 = f1_score(all_labels, all_preds, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )
    labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")

    print(f"\nMacro F1 Score:  {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro AUC-ROC:   {auc:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=LABELS)}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate_model()
