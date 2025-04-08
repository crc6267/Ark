# symbolic_input.py

from resonant_engine.glyphs.glyph_reverse_map import reverse_map

def prepare_input(*glyph_names):
    """
    Given symbolic glyph names, return their combined token sequence.
    
    Example:
        prepare_input("SELF", "FIRE") → [2, 7, 8, 7, 9, 3, 1, 0]
    """
    sequence = []
    for name in glyph_names:
        tokens = reverse_map.get(name.upper())
        if not tokens:
            print(f"[!] Unknown glyph: {name}")
            continue
        sequence.extend(tokens)
    return sequence


if __name__ == "__main__":
    seq = prepare_input("SELF", "FIRE", "RETURN_SIGNAL")
    print("Input Sequence:", seq)
