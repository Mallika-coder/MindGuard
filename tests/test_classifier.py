"""Tests for the BERT classifier model."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.classifier import MentalHealthClassifier
from model.config import NUM_CLASSES, MAX_LENGTH


def test_model_output_shape():
    model = MentalHealthClassifier()
    model.eval()

    batch_size = 4
    input_ids = torch.randint(0, 30000, (batch_size, MAX_LENGTH))
    attention_mask = torch.ones(batch_size, MAX_LENGTH, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (batch_size, NUM_CLASSES)


def test_model_probabilities():
    model = MentalHealthClassifier()
    model.eval()

    input_ids = torch.randint(0, 30000, (1, MAX_LENGTH))
    attention_mask = torch.ones(1, MAX_LENGTH, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.softmax(output, dim=1)

    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_model_gradient_flow():
    model = MentalHealthClassifier()
    model.train()

    input_ids = torch.randint(0, 30000, (2, MAX_LENGTH))
    attention_mask = torch.ones(2, MAX_LENGTH, dtype=torch.long)
    labels = torch.tensor([0, 1])

    output = model(input_ids, attention_mask)
    loss = torch.nn.CrossEntropyLoss()(output, labels)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


if __name__ == "__main__":
    test_model_output_shape()
    test_model_probabilities()
    test_model_gradient_flow()
    print("All classifier tests passed!")
