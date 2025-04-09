import os
import json
from resonant_engine.glyphs.glyph_map import glyph_map
from resonant_engine.glyphs.glyph_reverse_map import reverse_map
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

print("🧪 Current Working Directory:", os.getcwd())
print("📁 Files in data/:", list(data_dir.iterdir()) if data_dir.exists() else "❌ data/ not found")

training_path = data_dir / "glyph_training_set.json"
reverse_path = data_dir / "reverse_sequence_map.json"
syntax_path = data_dir / "glyph_syntax_map.json"

assert training_path.exists(), f"Missing: {training_path}"
assert reverse_path.exists(), f"Missing: {reverse_path}"
assert syntax_path.exists(), f"Missing: {syntax_path}"

# Load the data (this continues the rest of your script)
with training_path.open(encoding="utf-8") as f:
    training_data = json.load(f)

with reverse_path.open(encoding="utf-8") as f:
    reverse_seq_map = json.load(f)

with syntax_path.open(encoding="utf-8") as f:
    syntax_data = json.load(f)

# Convert glyph_map keys for matching
glyph_map_flat = {"{}".format(list(k)): v for k, v in glyph_map.items()}

# Validation pass
print("🔍 Validating glyph synchronization...\n")
errors = []

for glyph in training_data:
    name = glyph["name"].upper()
    seq = glyph["tokens"]
    seq_key = str(seq)

    # Check reverse_map
    if name not in reverse_map:
        errors.append(f"❌ Missing in glyph_reverse_map.py → {name}")

    # Check glyph_map
    if seq_key not in glyph_map_flat:
        errors.append(f"❌ Missing in glyph_map.py → {seq_key} for {name}")

    # Check reverse_sequence_map
    if seq_key not in reverse_seq_map:
        errors.append(f"❌ Missing in reverse_sequence_map.json → {seq_key} for {name}")

if errors:
    print("\n".join(errors))
    print(f"\n⚠️ {len(errors)} sync issues found.")
else:
    print("✅ All glyphs are synced across data sources.")
