import torch
import numpy as np
from transformers import BertTokenizer
from .classifier import MentalHealthClassifier
from .config import MODEL_SAVE_DIR, MAX_LENGTH, ID2LABEL, NUM_CLASSES
import os


class MentalHealthPredictor:
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = model_dir or MODEL_SAVE_DIR

        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer")
        )
        self.model = MentalHealthClassifier()
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "best_model.pt"),
                map_location=self.device,
            )
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        severity_map = {
            "normal": 0,
            "stress": 1,
            "anxiety": 2,
            "depression": 3,
            "severe": 4,
        }
        label = ID2LABEL[predicted_class]
        severity_score = severity_map.get(label, 0) / 4.0

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "severity_score": round(severity_score, 2),
            "probabilities": {
                ID2LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)
            },
        }
