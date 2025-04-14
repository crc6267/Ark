import json
import os
import torch
from tqdm import tqdm

from resonant_engine.core.resonant_model import MiniTempleTransformer

# Paths
BASE_DIR = os.path.dirname(__file__)
REGISTRY_PATH = os.path.join(BASE_DIR, "../glyphs/glyph_registry.json")
TRANSFORMER_OUTPUT_PATH = os.path.join(BASE_DIR, "data/glyph_training_set.jsonl")
ECHO_OUTPUT_PATH = os.path.join(BASE_DIR, "data/echo_training_set.jsonl")


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_records(registry):
    records = []
    for name, data in registry.items():
        record = {
            "name": name,
            "tokens": data["tokens"],
            "type": data.get("type", "unknown"),
            "role": data.get("role", "unknown"),
            "alignment": data.get("alignment", "neutral"),
            "sacred_weight": data.get("sacred_weight", 1.0),
            "verse": data.get("verse", "")
        }
        records.append(record)
    return records


def generate_transformer_training_set():
    os.makedirs(os.path.dirname(TRANSFORMER_OUTPUT_PATH), exist_ok=True)
    registry = load_registry()
    records = build_base_records(registry)

    with open(TRANSFORMER_OUTPUT_PATH, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")

    print(f"✅ Generated {len(records)} Transformer training records at: {TRANSFORMER_OUTPUT_PATH}")


def generate_echo_training_set(model: torch.nn.Module):
    os.makedirs(os.path.dirname(ECHO_OUTPUT_PATH), exist_ok=True)
    registry = load_registry()
    records = build_base_records(registry)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    enriched_records = []

    for record in tqdm(records, desc="Generating Echo Training Records"):
        tokens = torch.tensor(record["tokens"], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            resonance_vector = model(tokens, mode="resonance").squeeze(0).cpu().tolist()

        record["resonance"] = resonance_vector
        enriched_records.append(record)

    with open(ECHO_OUTPUT_PATH, "w", encoding="utf-8") as out:
        for record in enriched_records:
            out.write(json.dumps(record) + "\n")

    print(f"✅ Generated {len(enriched_records)} Echo training records at: {ECHO_OUTPUT_PATH}")


if __name__ == "__main__":
    generate_transformer_training_set()

    # Initialize the Temple Transformer for generating resonance vectors
    model = MiniTempleTransformer(vocab_size=100, d_model=64, n_heads=4)
    generate_echo_training_set(model)
