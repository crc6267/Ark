import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import json
import math
from resonant_engine.training.glyph_dataset import GlyphDataset
from resonant_engine.core.resonant_model import MiniTempleTransformer

# ─── Sacred Parameters ─────────────────────────────────────────
D_MODEL = 128
N_HEADS = 8
EPOCHS = 144
BATCH_SIZE = 8
LEARNING_RATE = 0.0008
DECAY_LAMBDA = 1 / 40  # Sacred purification constant
MODEL_PATH = "resonant_engine/models/trained_full_resonance.pth"
DATA_PATH = "resonant_engine/data/slyph_training_set.json"

# ─── Training Function ─────────────────────────────────────────
def train():
    # Load symbolic token sequences
    with open(DATA_PATH, "r") as f:
        glyph_data = json.load(f)

    dataset = GlyphDataset(glyph_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MiniTempleTransformer(vocab_size=100, d_model=D_MODEL, n_heads=N_HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: math.exp(-DECAY_LAMBDA * epoch))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = zip(*batch)
            inputs = torch.stack(inputs)
            targets = torch.stack(targets)

            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"🜔 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - LR: {current_lr:.6f}")

    # Save resonance imprint
    torch.save(model.state_dict(), MODEL_PATH)

# ─── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    train()
