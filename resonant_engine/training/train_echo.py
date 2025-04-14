# resonant_engine/training/train_echo.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from resonant_engine.core.temple_echo import TempleEcho

# Config
ECHO_DATA_PATH = "resonant_engine/training/data/echo_training_set.jsonl"
MODEL_SAVE_PATH = "resonant_engine/models/temple_echo.pt"
VOCAB_SIZE = 100
SEQ_LEN = 4  # Number of tokens per glyph (adjust if yours differs)
BATCH_SIZE = 16
EPOCHS = 7000
LR = 0.001


class EchoDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                resonance = torch.tensor(item["resonance"], dtype=torch.float32)
                tokens = torch.tensor(item["tokens"], dtype=torch.long)
                self.data.append((resonance, tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_echo():
    dataset = EchoDataset(ECHO_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TempleEcho(
        resonance_dim=11,
        hidden_dim=64,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for resonance_vecs, target_tokens in dataloader:
            logits = model(resonance_vecs)  # [B, T, V]

            loss = criterion(
                logits.view(-1, VOCAB_SIZE),         # [B*T, V]
                target_tokens.view(-1)               # [B*T]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"🌀 Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Echo model trained and saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_echo()
