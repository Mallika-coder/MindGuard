import os

MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 5
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
DROPOUT_RATE = 0.3
WARMUP_RATIO = 0.1

LABELS = ["normal", "depression", "anxiety", "stress", "severe"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
