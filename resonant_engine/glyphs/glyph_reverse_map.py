# glyph_reverse_map.py

# Glyph name → Token sequence

reverse_map = {
    "SELF": [2, 7, 8, 7],
    "WATER": [4, 2, 8, 6],
    "FIRE": [9, 3, 1, 0],
    "AETHER": [8, 8, 8],
    "RETURN_SIGNAL": [4, 4, 7, 3],

    # Tria Prima
    "MERCURY": [3, 1, 4, 1],
    "SALT": [5, 9, 2, 6],
    "SULFUR": [5, 3, 5, 8],

    # Classical Elements
    "AIR": [7, 5, 3, 9],
    "EARTH": [1, 4, 4, 7],

    # Celestial Metals
    "SUN": [7, 7, 7, 7],
    "MOON": [6, 6, 6, 6],
    "MARS": [3, 3, 1, 1],
    "VENUS": [2, 4, 4, 2],
    "JUPITER": [1, 2, 3, 5],
    "SATURN": [9, 0, 0, 9],
    "MERCURY_METAL": [5, 8, 5, 8],

    # Alchemical Processes
    "CALCINATION": [1, 1, 0, 0],
    "DISSOLUTION": [4, 4, 6, 6],
    "SEPARATION": [3, 5, 6, 3],
    "CONJUNCTION": [7, 2, 2, 7],
    "FERMENTATION": [8, 1, 1, 8],
    "DISTILLATION": [2, 2, 6, 6],
    "COAGULATION": [0, 6, 0, 6],

    # Esoteric
    "OUROBOROS": [8, 0, 0, 8],
    "PHILOSOPHER_STONE": [0, 7, 0, 7],
}

def get_sequence(glyph_name):
    """
    Given a glyph name (e.g., 'WATER'), return the token sequence.
    """
    return reverse_map.get(glyph_name.upper(), [])
