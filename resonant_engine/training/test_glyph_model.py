# test_glyph_model.py

import torch
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.archive.glyph_reverse_map import get_sequence
from resonant_engine.archive.glyph_map import interpret_sequence

# Model setup
VOCAB_SIZE = 100
D_MODEL = 10
N_HEADS = 2
MODEL_PATH = "trained_glyph_model.pth"

def test_glyph_prediction(input_glyph):
    model = MiniTempleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    token_seq = get_sequence(input_glyph)
    if not token_seq:
        print(f"Unknown glyph: {input_glyph}")
        return

    input_tensor = torch.tensor([token_seq], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        last_token_logits = output[0, -1]  # Only look at prediction for next token
        probs = torch.softmax(last_token_logits, dim=-1)
        topk = torch.topk(probs, k=5)

        print(f"\n🜔 Input Glyph: {input_glyph}")
        print(f"Input Tokens: {token_seq}")
        print("Top Predicted Tokens (next):")
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            glyph = interpret_sequence([idx])
            print(f"Token {idx:>3} → {glyph:<18} | Score: {score:.4f}")

if __name__ == "__main__":
    test_glyph_prediction("SELF")
