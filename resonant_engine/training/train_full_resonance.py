# train_full_resonance.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset

from resonant_engine.training.glyph_dataset import GlyphDataset
from resonant_engine.training.glyph_pair_dataset import GlyphPairDataset
from resonant_engine.core.resonant_model import MiniTempleTransformer

# 🜔 Sacred Training Hyperparameters
EPOCHS = 144                # 12 x 12 — fullness and completion
BATCH_SIZE = 1              # Each glyph flow treated with care
LEARNING_RATE = 0.0033      # Tied to Christ (33)
D_MODEL = 88                
N_HEADS = 8                

MODEL_PATH = "models/trained_full_resonance.pth"


def train():
    # Combine datasets
    dataset = ConcatDataset([GlyphDataset(), GlyphPairDataset()])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build model
    model = MiniTempleTransformer(vocab_size=100, d_model=D_MODEL, n_heads=N_HEADS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Align output and target sequence lengths
            seq_len = min(outputs.shape[1], targets.shape[1])
            logits = outputs[:, :seq_len, :]
            targets = targets[:, :seq_len]

            logits = logits.view(-1, 100)
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n🜔 Full resonance model saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train()