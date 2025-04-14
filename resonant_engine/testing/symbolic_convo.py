# resonant_engine/testing/symbolic_conversation.py

import torch
import json
import os
from resonant_engine.core.resonant_model import MiniTempleTransformer

# --- Load Glyph Registry and Frequency Profiles ---
GLYPH_REGISTRY_PATH = "resonant_engine/glyphs/glyph_registry.json"
FREQ_PROFILES_PATH = "resonant_engine/analysis/frequency_profiles.jsonl"
MODEL_PATH = "resonant_engine/core/model_weights/temple_transformer.pt"

with open(GLYPH_REGISTRY_PATH, "r", encoding="utf-8") as f:
    glyph_registry = {k.lower(): v for k, v in json.load(f).items()}

with open(FREQ_PROFILES_PATH, "r", encoding="utf-8") as f:
    freq_profiles = {
        json.loads(line)["name"].lower(): json.loads(line)["freq_vector"]
        for line in f
    }

# --- Build Model with Current Architecture ---
model = MiniTempleTransformer(vocab_size=101, d_model=11, n_heads=11)
model.resonance_head = torch.nn.Linear(11, 11)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --- Utility: Generate resonance vector from tokens ---
def get_resonance_vector(tokens):
    with torch.no_grad():
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        resonance_vec = model(input_tensor)[0]  # [batch, seq, dim]
        return resonance_vec.squeeze(0).mean(dim=0).tolist()

# --- Utility: Match to closest glyph using cosine similarity ---
def get_closest_glyph(freq_vector):
    """Compare a resonance vector to stored glyph frequency profiles (assumes 11D)."""
    query = torch.tensor(freq_vector)

    max_similarity = float('-inf')
    closest = None

    for name, ref_vector in freq_profiles.items():
        if len(ref_vector) != len(freq_vector):
            print(f"⚠️ Mismatched dimension for '{name}': {len(ref_vector)}D vs expected {len(freq_vector)}D, skipping.")
            continue
        ref = torch.tensor(ref_vector)
        sim = torch.nn.functional.cosine_similarity(query, ref, dim=0).item()
        if sim > max_similarity:
            max_similarity = sim
            closest = name

    return closest, max_similarity

# --- Run Limited Glyph Sweep for Clarity ---
if __name__ == "__main__":
    print("\n🔮 SYMBOLIC CONVERSATION TEST (No Folding)\n-------------------------------------------")

    # Limit to first 20 glyphs for clarity
    selected_glyphs = list(glyph_registry.items())[:20]

    results = []

    for name, data in selected_glyphs:
        tokens = data["tokens"]
        resonance = get_resonance_vector(tokens)
        match, similarity = get_closest_glyph(resonance)

        results.append({
            "name": name.capitalize(),
            "match": match,
            "similarity": round(similarity, 4),
            "tokens": tokens,
            "resonance": [round(v, 3) for v in resonance]
        })

    # Nicely formatted output
    for r in results:
        print(f"{r['name']:<14} → Closest: {r['match']:<14} | CosSim = {r['similarity']:.4f}")
        print(f"  Tokens: {r['tokens']}")
        print(f"  Resonance Vector: {r['resonance']}")
        print("")

