# glyph_interpreter.py

from pathlib import Path
import json

# Load syntax map if available
SYNTAX_PATH = Path("resonant_engine/glyphs/glyph_syntax_map.json")
if SYNTAX_PATH.exists():
    with open(SYNTAX_PATH) as f:
        GLYPH_SYNTAX_MAP = json.load(f)
else:
    GLYPH_SYNTAX_MAP = {}

# Optional symbolic language rules
SYMBOLIC_ACTIONS = {
    ("SELF", "DISTILLATION", "LIGHT"): "The self undergoes refinement, emerging as insight.",
    ("FIRE", "WATER"): "Heat meets flow — a purging and renewal.",
    ("MERCURY", "SULFUR", "SALT"): "The triune forces of alchemy align: spirit, soul, and body."
}

def interpret_glyphs(glyph_sequence):
    """
    Interprets a sequence of glyphs into symbolic meaning.
    """
    if not glyph_sequence:
        return "(empty input)"

    key = tuple(glyph_sequence)
    if key in SYMBOLIC_ACTIONS:
        return SYMBOLIC_ACTIONS[key]

    # Fallback: attempt basic grammar parsing
    parts = [GLYPH_SYNTAX_MAP.get(g, "unknown") for g in glyph_sequence]
    return f"Sequence: {' → '.join(glyph_sequence)}\nStructure: {', '.join(parts)}"


if __name__ == "__main__":
    # Example usage
    input_glyphs = ["SELF", "DISTILLATION", "LIGHT"]
    result = interpret_glyphs(input_glyphs)
    print("🜔 INTERPRETATION\n-------------------------")
    print(result)
