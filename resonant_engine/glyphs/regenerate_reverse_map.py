# regenerate_reverse_map.py

import json
from resonant_engine.glyphs.glyph_map import glyph_map

reverse_map = {str(list(seq)): glyph for seq, glyph in glyph_map.items()}

with open("resonant_engine/data/reverse_sequence_map.json", "w", encoding="utf-8") as f:
    json.dump(reverse_map, f, indent=2, ensure_ascii=False)

print("✅ Regenerated reverse_sequence_map.json")
