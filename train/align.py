import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from structure.structure import MiniTempleTransformer

# --- Mirror Function ---
def mirror_sequence(seq, mode="retrograde"):
    if mode == "retrograde":
        return seq.flip(dims=[1])
    elif mode == "invert":
        return 9 - seq
    elif mode == "reflection":
        return seq[:, ::-1]
    else:
        return seq

# --- Device and Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Running on:", device)

# --- Symbolic Prompts + Mirroring ---
original_prompts = [
    torch.tensor([[2, 7, 8, 7]]),
    torch.tensor([[1, 3, 1, 3]]),
    torch.tensor([[4, 0, 1, 2]]),
    torch.tensor([[7, 7, 7, 7]]),
    torch.tensor([[9, 3, 2, 0]])
]

symbolic_prompts = []
for seq in original_prompts:
    mirrored = mirror_sequence(seq, mode="retrograde")
    combined = torch.cat([seq, mirrored], dim=1)  # Shape: (1, 8)
    symbolic_prompts.append(combined)

# --- Target Motifs ---
target_motifs = [torch.randn(1, 10) for _ in symbolic_prompts]

# --- Model and Training Setup ---
model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CosineEmbeddingLoss()
scaler = GradScaler()

# --- Training Ritual ---
model.train()
for epoch in range(888):
    total_loss = 0
    for input_ids, target_vec in zip(symbolic_prompts, target_motifs):
        input_ids = input_ids.to(device)
        target_vec = target_vec.to(device)
        label = torch.tensor([1.0], dtype=torch.float32).to(device)

        with autocast():
            embed = model(input_ids, return_embed=True)  # Get symbolic vector directly
            loss = loss_fn(embed, target_vec, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} — Symbolic Loss: {total_loss:.4f}")

# --- Save the Veil ---
torch.save(model.state_dict(), "veil.pt")
