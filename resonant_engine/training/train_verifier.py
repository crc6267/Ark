import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from resonant_engine.alignment.verifier_model import AlignmentVerifierModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/glyph_training_set.jsonl")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../alignment/verifier.pt")

INTENT_LABELS = ["neutral", "reverent", "agitated"]

# ----------------------------
# 📦 Dataset
# ----------------------------
class GlyphDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        tokens = r["tokens"]
        alignment = r["alignment"].lower()
        role = r["role"].lower()

        # 🔮 Fake 6D resonance vector: mean + position signal
        vec = torch.tensor([
            sum(tokens[:2]) / 20.0,
            sum(tokens[2:]) / 20.0,
            tokens[0] / 10.0,
            tokens[1] / 10.0,
            tokens[2] / 10.0,
            tokens[3] / 10.0,
        ], dtype=torch.float32)

        # 🧠 Simulated labels
        purity = 1.0 if alignment in ["divine", "christic", "holy"] else 0.3
        tone = 1 if alignment in ["angelic", "divine", "christic"] else 0 if alignment in ["chaotic"] else 2
        resonance = 0.8 if role in ["initiator", "subject"] else 0.5

        return vec, torch.tensor(purity), torch.tensor(tone), torch.tensor(resonance)


# ----------------------------
# 🧪 Training Loop
# ----------------------------
def train_avm_model():
    dataset = GlyphDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AlignmentVerifierModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    cls_loss = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for x, purity, tone, resonance in loader:
            out = model(x)
            loss = (
                loss_fn(out["semantic_purity"], purity)
                + cls_loss(out["intent_tone_probs"], tone)
                + loss_fn(out["resonance_score"], resonance)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10 | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ AVM saved to {SAVE_PATH}")


if __name__ == "__main__":
    train_avm_model()
