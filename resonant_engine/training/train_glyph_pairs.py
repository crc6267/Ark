# train_glyph_pairs.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from glyph_pair_dataset import GlyphPairDataset
from resonant_model import MiniTempleTransformer

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 0.003
VOCAB_SIZE = 100
D_MODEL = 32
N_HEADS = 2

MODEL_PATH = "trained_glyph_pairs.pth"


def train():
    dataset = GlyphPairDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MiniTempleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS)
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

            logits = logits.view(-1, VOCAB_SIZE)
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n🜔 Relationship model trained and saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train()
