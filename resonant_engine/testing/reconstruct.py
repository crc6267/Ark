# resonant_engine/testing/reconstruct_from_resonance.py
import torch
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.core.temple_voice import TempleVoice
from resonant_engine.glyphs.glyph_vectorizer import get_tokens, decode_tokens
from resonant_engine.utils.tracer import ResonanceTracer

# --- Load model and weights ---
model = MiniTempleTransformer(vocab_size=100, d_model=64, n_heads=4)
model.load_state_dict(torch.load("resonant_engine/core/model_weights/temple_transformer.pt"))
model.eval()

# --- Setup ---
def reconstruct_from_resonance(glyph_name):
    tracer = ResonanceTracer()

    tokens = get_tokens(glyph_name.upper())
    if not tokens:
        print(f"❌ Unknown glyph: {glyph_name}")
        return

    x = torch.tensor([tokens])
    print(f"🌀 Original Input Tokens: {tokens}")

    # Step 1: Get resonance vector
    resonance_vec = model(x, mode="resonance", tracer=tracer)  # [B, 6]
    print("🔮 Resonance Vector:", resonance_vec[0].tolist())

    # Step 2: Echo through TempleVoice
    voice = TempleVoice()
    echo_logits = voice(resonance_vec.unsqueeze(1))  # [B, 1, V]
    echo_token = torch.argmax(echo_logits, dim=-1).squeeze().item()
    echo_name = decode_tokens([echo_token])

    print(f"📣 Reconstructed Token ID: {echo_token}")
    print(f"🗣️ Reconstructed Name: {echo_name}")

    tracer.summary()


# --- Manual Test ---
if __name__ == "__main__":
    reconstruct_from_resonance("SELF")
