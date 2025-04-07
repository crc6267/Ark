# symbolic_echo.py

from resonant_engine.glyphs.glyph_map import glyph_map


def interpret_token_echo(token):
    """
    Given a single token, search all glyphs to find the most likely symbolic alignment.
    Returns best guess glyph, match type, and list of candidates.
    """
    token = int(token)
    candidates = []

    for seq, glyph in glyph_map.items():
        if token == seq[0]:
            candidates.append((glyph, "prefix"))
        elif token in seq:
            candidates.append((glyph, "partial"))

    if not candidates:
        return "UNKNOWN", "none", []

    # Prioritize prefix matches
    prefix_matches = [c for c in candidates if c[1] == "prefix"]
    if prefix_matches:
        return prefix_matches[0][0], "prefix", prefix_matches

    # Otherwise return first partial
    return candidates[0][0], "partial", candidates


if __name__ == "__main__":
    for t in [4, 2, 38, 7]:
        glyph, match_type, matches = interpret_token_echo(t)
        print(f"Token {t} → {glyph} ({match_type}) | Options: {[m[0] for m in matches]}")
