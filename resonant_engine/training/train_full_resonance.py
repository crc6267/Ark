import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import Dataset, DataLoader
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.training.glyph_dataset import GlyphDataset

# Hyperparameters
D_MODEL = 128
N_HEADS = 8
BATCH_SIZE = 16
EPOCHS = 888
LEARNING_RATE = 1e-3
MODEL_PATH = MODEL_PATH = "resonant_engine/models/trained_full_resonance.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
GLYPH_DATA_PATH = "resonant_engine/data/glyph_training_set.json"

# Updated pad_collate function

def pad_collate(batch):
    inputs, targets = zip(*batch)  # already tensors
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs, targets

def train():
    # Load training data
    with open(GLYPH_DATA_PATH) as f:
        glyph_data = json.load(f)

    dataset = GlyphDataset(glyph_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    # Model
    model = MiniTempleTransformer(vocab_size=100, d_model=D_MODEL, n_heads=N_HEADS)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            logits = model(inputs)  # (B, S, vocab)
            logits = logits[:, :targets.size(1), :]  # Crop logits to match target seq length
            logits = logits.reshape(-1, logits.size(-1))  # (B×S, vocab)
            targets = targets.reshape(-1)  # (B×S)

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()
