import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.training.glyph_dataset import GlyphSequenceDataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/glyph_training_set.jsonl")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../core/model_weights/temple_transformer.pt")

# ----------------------------
# 🧪 Training Loop
# ----------------------------
def train_transformer():
    dataset = GlyphSequenceDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MiniTempleTransformer(vocab_size=100, d_model=64, n_heads=4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            logits = model(x, mode="logits")  # [B, 2T, V] due to mirroring
            logits = logits[:, :x.shape[1], :]  # ✨ slice to match original T

            logits = logits.contiguous().view(-1, logits.size(-1))  # [B*T, V]
            y = y.contiguous().view(-1)  # [B*T]

            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10 | Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Temple Transformer saved to {SAVE_PATH}")


if __name__ == "__main__":
    train_transformer()
