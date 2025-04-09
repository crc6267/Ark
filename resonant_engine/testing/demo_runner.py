# demo_runner.py

import torch
import torch.nn.functional as F
import os
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.glyphs.symbolic_input import prepare_input
from resonant_engine.glyphs.glyph_vectorizer import vectorize_glyph, describe_vector

# --- Settings ---
D_MODEL = 128
N_HEADS = 8
VOCAB_SIZE = 100
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/trained_resonant_vector.pth')

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

# --- Run Inference in Resonance Mode ---
with torch.no_grad():
    input_tensor = input_tokens.clone().detach()
    resonance_vec = model(input_tensor, mode="resonance")

# --- Compare Against First Glyph Vector ---
original_vec = vectorize_glyph(glyph_sequence[0])
print(f"\n🔮 Resonance Comparison (vs. {glyph_sequence[0]}):")

# Calculate cosine similarity
if original_vec.norm() == 0 or resonance_vec.norm() == 0:
    score = 0.0
else:
    score = F.cosine_similarity(resonance_vec, original_vec.unsqueeze(0)).item()

print(f" → Cosine Resonance: {score:.4f} | {describe_vector(original_vec)}")
