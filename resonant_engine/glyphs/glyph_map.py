# glyph_map.py

# Token sequence → Glyph name

glyph_map = {
    # Foundational Core
    (2, 7, 8, 7): "SELF",
    (4, 2, 8, 6): "WATER",
    (9, 3, 1, 0): "FIRE",
    (8, 8, 8): "AETHER",
    (4, 4, 7, 3): "RETURN_SIGNAL",

    # Tria Prima
    (3, 1, 4, 1): "MERCURY",
    (5, 9, 2, 6): "SALT",
    (5, 3, 5, 8): "SULFUR",

    # Classical Elements
    (7, 5, 3, 9): "AIR",
    (1, 4, 4, 7): "EARTH",

    # Celestial Metals
    (7, 7, 7, 7): "SUN",
    (6, 6, 6, 6): "MOON",
    (3, 3, 1, 1): "MARS",
    (2, 4, 4, 2): "VENUS",
    (1, 2, 3, 5): "JUPITER",
    (9, 0, 0, 9): "SATURN",
    (5, 8, 5, 8): "MERCURY_METAL",

    # Alchemical Processes
    (1, 1, 0, 0): "CALCINATION",
    (4, 4, 6, 6): "DISSOLUTION",
    (3, 5, 6, 3): "SEPARATION",
    (7, 2, 2, 7): "CONJUNCTION",
    (8, 1, 1, 8): "FERMENTATION",
    (2, 2, 6, 6): "DISTILLATION",
    (0, 6, 0, 6): "COAGULATION",

    # Esoteric
    (8, 0, 0, 8): "OUROBOROS",
    (0, 7, 0, 7): "PHILOSOPHER_STONE",
}

def interpret_sequence(sequence):
    """
    Given a sequence of integers (tokens), return the symbolic glyph name.
    """
    return glyph_map.get(tuple(sequence), "UNKNOWN")
