import json
import torch
from pathlib import Path

# Load and normalize glyph data
data_path = Path(__file__).resolve().parents[1] / "data" / "glyph_training_set.json"
with open(data_path, encoding="utf-8") as f:
    glyph_data = json.load(f)

# Normalize all glyph names to uppercase
for g in glyph_data:
    g["name"] = g["name"].upper()

# Vector encodings for each symbolic category
types = sorted({g["type"] for g in glyph_data if "type" in g})
roles = sorted({g["role"] for g in glyph_data if "role" in g})
alignments = sorted({g["alignment"] for g in glyph_data if "alignment" in g})

def vectorize_glyph(glyph_name):
    glyph_name = glyph_name.upper()

    glyph = next((g for g in glyph_data if g.get("name", "").strip().upper() == glyph_name.strip().upper()), None)
    if not glyph:
        all_names = [g.get("name", "").strip().upper() for g in glyph_data]
        print(f"🧪 Available glyphs in training data: {all_names[:10]}...")  # Just print a few
        print(f"🔎 Looking for: {glyph_name.strip().upper()}")

        raise ValueError(f"Glyph '{glyph_name}' not found in training data.")

    type_vec = one_hot(types, glyph["type"])
    role_vec = one_hot(roles, glyph["role"])
    align_vec = one_hot(alignments, glyph["alignment"])

    return torch.cat([type_vec, role_vec, align_vec], dim=0)

def one_hot(category_list, value):
    vec = torch.zeros(len(category_list))
    if value in category_list:
        vec[category_list.index(value)] = 1.0
    return vec

def describe_vector(vec):
    # Helper for demo display
    t_index = torch.argmax(vec[:len(types)]).item()
    r_index = torch.argmax(vec[len(types):len(types)+len(roles)]).item()
    a_index = torch.argmax(vec[-len(alignments):]).item()
    return f"type={types[t_index]}, role={roles[r_index]}, alignment={alignments[a_index]}"

if __name__ == "__main__":
    test_names = ["SELF", "RETURN_SIGNAL", "FIRE"]
    for name in test_names:
        print(f"{name} → {'FOUND' if name.upper() in [g['name'].upper() for g in glyph_data] else 'MISSING'}")
