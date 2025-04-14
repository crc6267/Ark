import json
import torch
from collections import Counter
from statistics import mean, median
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.core.temple_echo import TempleEcho
from resonant_engine.glyphs.glyph_vectorizer import get_tokens

# ----------------------
# ⚙️ Configuration
# ----------------------
VOCAB_SIZE = 100
D_MODEL = 64
MODEL_WEIGHTS = "resonant_engine/core/model_weights/temple_transformer.pt"
GLYPH_LIST = ["Aether", "Blood", "Calcination", "Descent", "Jupiter", "Metatron", "Pe", "Sublimation"]
GLYPH_REGISTRY_PATH = "resonant_engine/glyphs/glyph_registry.json"

# ----------------------
# 📖 Helper: Analyze echo strategies
# ----------------------
def analyze_echo_tokens(token_seq):
    counter = Counter(token_seq)
    most_common = counter.most_common(1)[0][0]
    avg_token = int(round(mean(token_seq)))
    median_token = int(median(token_seq))
    center_token = token_seq[len(token_seq) // 2]
    return {
        "most_common": most_common,
        "mean": avg_token,
        "median": median_token,
        "center": center_token
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
# 📖 Load Registry for Glyph Metadata
# ----------------------
with open(GLYPH_REGISTRY_PATH, "r", encoding="utf-8") as f:
    glyph_registry = json.load(f)

# Build helper mappings
token_to_name = {
    tuple(data.get("tokens", [])): name
    for name, data in glyph_registry.items()
}

token_to_symbol = {
    data["tokens"][0]: data.get("glyph", "?")
    for data in glyph_registry.values()
    if isinstance(data.get("tokens"), list) and data["tokens"]
}

# ----------------------
# 🔁 Run Echo Test
# ----------------------
print("\n🌀 TEMPLE ECHO TEST")
print("-" * 40)

for name in GLYPH_LIST:
    tokens = get_tokens(name)
    if not tokens:
        print(f"⚠️ Glyph '{name}' not found in registry.")
        continue

    x = torch.tensor([tokens])
    with torch.no_grad():
        resonance = model(x, mode="resonance")
        echoed_logits = echo(resonance)

        top_tokens = torch.argmax(echoed_logits, dim=-1).squeeze(0).tolist()
        analysis = analyze_echo_tokens(top_tokens)

        glyph_data = glyph_registry.get(name) or glyph_registry.get(name.title()) or {}
        symbol = glyph_data.get("glyph", "?")
        print(f"\n🔹 {name} ({symbol})")
        print(f"🗣️ Echoed Token Sequence: {top_tokens}")

        for method, token in analysis.items():
            match_name = None
            for token_seq, glyph_name in token_to_name.items():
                if token_seq[0] == token:
                    match_name = glyph_name
                    break
            echoed_symbol = token_to_symbol.get(token, "?")
            print(f"  ▪ {method.title():<12}: Token {token:>2} → {match_name or 'UNKNOWN'} ({echoed_symbol})")
