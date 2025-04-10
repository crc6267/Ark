import torch
import json
import ast
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.glyphs.symbolic_input import prepare_input

# Load the trained model weights
MODEL_PATH = "resonant_engine/models/trained_full_resonance.pth"
REVERSE_SEQUENCE_MAP_PATH = "resonant_engine/data/reverse_sequence_map.json"

D_MODEL = 128
N_HEADS = 8

# Load 4-token reverse map
with open(REVERSE_SEQUENCE_MAP_PATH) as f:
    reverse_sequence_map = json.load(f)
    reverse_sequence_map = {tuple(ast.literal_eval(k)): v for k, v in reverse_sequence_map.items()}

# Instantiate two Temple models
model_A = MiniTempleTransformer(vocab_size=100, d_model=D_MODEL, n_heads=N_HEADS)
model_B = MiniTempleTransformer(vocab_size=100, d_model=D_MODEL, n_heads=N_HEADS)
model_A.load_state_dict(torch.load(MODEL_PATH))
model_B.load_state_dict(torch.load(MODEL_PATH))
model_A.eval()
model_B.eval()

# Initial glyph sequence
glyph_sequence = ["SELF", "FIRE", "RETURN_SIGNAL"]
tokens = []
from resonant_engine.archive.glyph_reverse_map import reverse_map
for name in glyph_sequence:
    token = reverse_map.get(name.upper())
    if token is not None:
        tokens.extend(token)
    else:
        print(f"Warning: Glyph {name} not found in reverse map.")

input_tokens = torch.tensor(tokens, dtype=torch.long)

print("\n🜔 TEMPLE COMMUNION BEGINS\n")
print(f"Initial Glyphs: {glyph_sequence} → Tokens: {input_tokens.tolist()}")

# Echo loop
steps = 5
for i in range(steps):
    model = model_A if i % 2 == 0 else model_B

    with torch.no_grad():
        logits = model(input_tokens.unsqueeze(0))  # Add batch dim
        probs = torch.softmax(logits[0, -1], dim=0)  # Last token prediction
        top_token = torch.argmax(probs).item()
        score = probs[top_token].item()

    input_tokens = torch.cat([input_tokens, torch.tensor([top_token])])

    # Match last 4-token sequence
    last_4 = tuple(input_tokens[-4:].tolist())
    glyph_out = reverse_sequence_map.get(last_4, "UNKNOWN")

    print(f"Step {i+1}: Model {'A' if i % 2 == 0 else 'B'} → Top Token: {top_token} | Last 4 → {last_4} → {glyph_out} | Score: {score:.4f}")

print("\n🜔 COMMUNION ENDS\n")