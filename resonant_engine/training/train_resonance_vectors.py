# train_resonance_vectors.py

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
import json
import os
import math

from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.glyphs.glyph_vectorizer import vectorize_glyph

# ─── Sacred Parameters ─────────────────────────────────────────
D_MODEL = 128
N_HEADS = 8
EPOCHS = 20
BATCH_SIZE = 8
INITIAL_LR = 1e-3
DECAY_LAMBDA = 1 / 40
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/trained_resonant_vector.pth')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/glyph_training_set.json')

# ─── Dataset for Resonance Vectors ─────────────────────────────
class GlyphVectorDataset(Dataset):
    def __init__(self, glyph_data):
        self.inputs = []
        self.vectors = []

        for glyph in glyph_data:
            name = glyph["name"]
            tokens = glyph["tokens"]
            if len(tokens) != 4:
                continue
            tensor = torch.tensor(tokens, dtype=torch.long)
            vec = vectorize_glyph(name)
            self.inputs.append(tensor)
            self.vectors.append(vec)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.vectors[idx]

# ─── Cosine Similarity Loss ─────────────────────────────────────
def cosine_loss(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()

# ─── Training Loop ──────────────────────────────────────────────
def train():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        glyph_data = json.load(f)

    dataset = GlyphVectorDataset(glyph_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MiniTempleTransformer(vocab_size=100, d_model=D_MODEL, n_heads=N_HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: math.exp(-DECAY_LAMBDA * epoch))

    for epoch in range(EPOCHS):
        total_loss = 0.0
        model.train()

        for inputs, targets in dataloader:
            inputs = inputs
            targets = targets

            outputs = model(inputs, mode="resonance")
            loss = cosine_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"🜔 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - LR: {current_lr:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Saved to {MODEL_PATH}")

# ─── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    train()
