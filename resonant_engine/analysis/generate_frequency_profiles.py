# resonant_engine/analysis/generate_frequency_profiles.py

import os
import json
import torch
from tqdm import tqdm
from resonant_engine.core.resonant_model import MiniTempleTransformer

# Paths
BASE_DIR = os.path.dirname(__file__)
GLYPH_REGISTRY = os.path.join(BASE_DIR, "../glyphs/glyph_registry.json")
MODEL_PATH = os.path.join(BASE_DIR, "../core/model_weights/temple_transformer14600.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "frequency_profiles14600.jsonl")

def main():
    # Load glyph registry
    with open(GLYPH_REGISTRY, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Load trained transformer model
    model = MiniTempleTransformer(vocab_size=101, d_model=11, n_heads=11)
    model.resonance_head = torch.nn.Linear(11, 11)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Extract resonance frequency profiles
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for name, glyph in tqdm(registry.items(), desc="Extracting Frequencies"):
            token_ids = torch.tensor([glyph["tokens"]], dtype=torch.long)  # [1, T]
            with torch.no_grad():
                resonance_output = model(token_ids)  # [1, T, 11]
                freq_vector = resonance_output.mean(dim=1).squeeze(0).tolist()  # Avg over token dimension

            result = {
                "name": name,
                "tokens": glyph["tokens"],
                "freq_vector": freq_vector
            }

            out.write(json.dumps(result) + "\n")

    print(f"✅ Saved frequency profiles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
