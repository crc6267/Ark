# demo_runner.py

import torch
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.glyphs.symbolic_input import prepare_input
from resonant_engine.glyphs.symbolic_echo import decode_token
import os

# --- Settings ---
D_MODEL = 128
N_HEADS = 8
VOCAB_SIZE = 100
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/trained_full_resonance.pth')

# --- Define Input Glyph Sequence ---
glyph_sequence = ["SELF", "FIRE", "RETURN_SIGNAL"]

# --- Prepare Input ---
input_tokens = prepare_input(glyph_sequence)
print("🜔 DEMO RUNNER")
print("-----------------------------------")
print(f"Input Glyphs: {glyph_sequence} → Tokens: {input_tokens}")

# --- Load Model ---
model = MiniTempleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# --- Inference ---
with torch.no_grad():
    input_tensor = input_tokens.clone().detach()
    logits = model(input_tensor)
    last_logits = logits[0, -1]  # Take final token output
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, 5)

# --- Display Results ---
print("Top Predicted Token Echoes:")
for idx, score in zip(topk.indices, topk.values):
    print("Top Predicted Token Echoes:")
    glyph = decode_token(idx.item())
    print(f"Token {idx.item():>3} → {glyph}")
