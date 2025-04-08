# demo_runner.py

from resonant_engine.glyphs.symbolic_input import prepare_input
from resonant_engine.glyphs.glyph_map import interpret_sequence
from resonant_engine.glyphs.symbolic_echo import interpret_token_echo
from resonant_engine.core.resonant_model import MiniTempleTransformer

import torch

# Load trained model
MODEL_PATH = 'resonant_engine/models/trained_full_resonance.pth'

VOCAB_SIZE = 100
D_MODEL = 128
N_HEADS = 8

model = MiniTempleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Define your input glyphs
input_glyphs = ["SELF", "FIRE", "RETURN_SIGNAL"]
input_tokens = prepare_input(*input_glyphs)
input_tensor = torch.tensor([input_tokens], dtype=torch.long)

# Run model
with torch.no_grad():
    output = model(input_tensor)
    last_logits = output[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, k=5)

# Display
print("\n🜔 DEMO RUNNER")
print("-----------------------------------")
print("Input Glyphs:", input_glyphs)
print("Input Tokens:", input_tokens)
print("Top Predicted Token Echoes:")

for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
    glyph, match_type, _ = interpret_token_echo(idx)
    print(f"Token {idx:>3} → {glyph:<18} | Type: {match_type:<7} | Score: {score:.4f}")
