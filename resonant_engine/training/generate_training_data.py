import json
import os

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "../glyphs/glyph_registry.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data/glyph_training_set.jsonl")


def generate_training_set():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

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

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")

    print(f"✅ Generated {len(records)} training records at: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_training_set()