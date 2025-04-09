# demo_runner.py

import torch
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.glyphs.symbolic_input import prepare_input
from resonant_engine.glyphs.symbolic_echo import decode_token
from resonant_engine.glyphs.glyph_vectorizer import vectorize_glyph, describe_vector
import torch.nn.functional as F
import os

# --- Settings ---
D_MODEL = 128
N_HEADS = 8
VOCAB_SIZE = 100
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/trained_full_resonance.pth')

# --- Input Glyph Sequence ---
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

# --- Run Inference ---
with torch.no_grad():
    input_tensor = input_tokens.clone().detach()
    logits = model(input_tensor)
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, 5)

# --- Top Predictions ---
print("\nTop Predicted Token Echoes:")
top_indices = topk.indices
top_scores = topk.values

print("Top Predicted Token Echoes:")
for idx, score in zip(topk.indices, topk.values):
    glyph_name = decode_token(idx.item())
    print(f"Token {idx.item():>3} → {glyph_name}")

# --- Resonance Scoring ---
original_vec = vectorize_glyph(glyph_sequence[0])
print(f"\n🔎 Resonance Score (vs. {glyph_sequence[0]}):")

for idx in top_indices:
    result = decode_token(idx.item())
    glyph_name = result[0] if isinstance(result, tuple) else result

    predicted_vec = vectorize_glyph(glyph_name)

    if original_vec.norm() == 0 or predicted_vec.norm() == 0:
        score = 0.0
    else:
        score = F.cosine_similarity(original_vec.unsqueeze(0), predicted_vec.unsqueeze(0)).item()

    print(f" → {glyph_name:15} | Cosine Resonance: {score:.4f} | {describe_vector(predicted_vec)}")
