import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from config import (
    MODEL_NAME, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO, LABELS, LABEL2ID, DATA_DIR, MODEL_SAVE_DIR,
)
from dataset import MentalHealthDataset
from classifier import MentalHealthClassifier


def load_data():
    """Load and preprocess the Reddit Mental Health dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, "reddit_mental_health.csv"))
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 20]
    df["label_id"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    f1 = f1_score(all_labels, all_preds, average="macro")
    labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")

    report = classification_report(all_labels, all_preds, target_names=LABELS)
    return f1, auc, report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_data()
    print(f"Dataset size: {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label_id"]
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = MentalHealthDataset(
        train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer
    )
    val_dataset = MentalHealthDataset(
        val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer
    )
    test_dataset = MentalHealthDataset(
        test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MentalHealthClassifier().to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print("\nStarting training...")
    best_f1 = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_f1, val_auc, val_report = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f} | Val AUC-ROC: {val_auc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pt"))
            tokenizer.save_pretrained(os.path.join(MODEL_SAVE_DIR, "tokenizer"))
            print(f"  ✓ Saved best model (F1: {best_f1:.4f})")

    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pt")))
    test_f1, test_auc, test_report = evaluate(model, test_loader, device)

    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")
    print(f"\nClassification Report:\n{test_report}")


if __name__ == "__main__":
    main()
