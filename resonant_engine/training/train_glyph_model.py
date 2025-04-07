# train_glyph_model.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from resonant_engine.training.glyph_dataset import GlyphDataset
from resonant_engine.core.resonant_model import MiniTempleTransformer

# Hyperparameters
EPOCHS = 25
BATCH_SIZE = 1
LEARNING_RATE = 0.005
VOCAB_SIZE = 100
D_MODEL = 10
N_HEADS = 2

def train():
    dataset = GlyphDataset()
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
            # Ensure same sequence length
            seq_len = min(outputs.shape[1], targets.shape[1])
            logits = outputs[:, :seq_len, :]
            targets = targets[:, :seq_len]

            # Flatten for loss
            logits = logits.reshape(-1, VOCAB_SIZE)
            targets = targets.reshape(-1)


            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "trained_glyph_model.pth")
    print("\n🜔 Model trained and saved to 'trained_glyph_model.pth'")

if __name__ == "__main__":
    train()
