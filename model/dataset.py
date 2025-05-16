import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from .config import MODEL_NAME, MAX_LENGTH


class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
