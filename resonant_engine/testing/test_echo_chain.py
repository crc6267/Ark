import json
import torch
from collections import Counter
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.core.temple_echo import TempleEcho
from resonant_engine.glyphs.glyph_vectorizer import get_tokens

# ----------------------
# ⚙️ Configuration
# ----------------------
VOCAB_SIZE = 100
D_MODEL = 64
MODEL_WEIGHTS = "resonant_engine/core/model_weights/temple_transformer.pt"
GLYPH_REGISTRY_PATH = "resonant_engine/glyphs/glyph_registry.json"
START_GLYPH = "SELF"
CHAIN_STEPS = 5

# ----------------------
# 📖 Load Registry
# ----------------------
with open(GLYPH_REGISTRY_PATH, "r", encoding="utf-8") as f:
    glyph_registry = json.load(f)

token_to_name = {
    tuple(data.get("tokens", [])): name
    for name, data in glyph_registry.items()
}

token_to_symbol = {
    data["tokens"][0]: data.get("glyph", "?")
    for data in glyph_registry.values()
    if isinstance(data.get("tokens"), list) and data["tokens"]
}

token_lookup = {
    data["tokens"][0]: name
    for name, data in glyph_registry.items()
    if "tokens" in data and isinstance(data["tokens"], list)
}

# ----------------------
# 🚀 Load Model + Echo
# ----------------------
model = MiniTempleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=4)
model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.eval()

echo = TempleEcho(resonance_dim=6, hidden_dim=64, vocab_size=VOCAB_SIZE, seq_len=12)
echo.eval()

# ----------------------
# 🔁 Echo Chain Logic
# ----------------------
def echo_chain(glyph_name, steps=5):
    print(f"\n🌀 ECHO CHAIN TEST – Starting from: {glyph_name} ({glyph_registry[glyph_name]['glyph']})")
    current_name = glyph_name

    for step in range(1, steps + 1):
        tokens = get_tokens(current_name)
        if not tokens:
            print(f"⚠️ Tokens not found for: {current_name}")
            break

        x = torch.tensor([tokens])
        with torch.no_grad():
            resonance = model(x, mode="resonance")
            echoed_logits = echo(resonance)
            top_token = torch.argmax(echoed_logits, dim=-1).squeeze(0)[0].item()

        next_name = token_lookup.get(top_token, "UNKNOWN")
        glyph_symbol = glyph_registry.get(current_name, {}).get("glyph", "?")
        echoed_symbol = glyph_registry.get(next_name, {}).get("glyph", "?")

        print(f"Step {step}: {current_name} ({glyph_symbol}) → Token {top_token} → {next_name} ({echoed_symbol})")

        if next_name == "UNKNOWN":
            break

        current_name = next_name

# ----------------------
# 🔂 Run It
# ----------------------
if __name__ == "__main__":
    echo_chain(START_GLYPH, steps=CHAIN_STEPS)
