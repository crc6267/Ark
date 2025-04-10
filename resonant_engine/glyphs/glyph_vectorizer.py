import json
import os

# Path to the canonical glyph registry
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "glyph_registry.json")

# Load registry once at module level
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    GLYPH_REGISTRY = json.load(f)

# Map glyph name (lowercased) → token sequence
GLYPH_TO_TOKENS = {name.lower(): data["tokens"] for name, data in GLYPH_REGISTRY.items()}

# Map token sequence (as string) → original glyph name (preserve formatting)
TOKENS_TO_GLYPH = {".".join(map(str, data["tokens"])): name for name, data in GLYPH_REGISTRY.items()}


def get_tokens(glyph_name):
    """
    Returns the token sequence for a glyph name (case-insensitive).
    """
    return GLYPH_TO_TOKENS.get(glyph_name.lower())


def get_glyph_name(tokens):
    """
    Returns the glyph name for a given token list.
    """
    key = ".".join(map(str, tokens))
    return TOKENS_TO_GLYPH.get(key)


def get_metadata(glyph_name):
    """
    Returns full metadata for a glyph: type, role, alignment, verse, etc.
    Case-insensitive.
    """
    return GLYPH_REGISTRY.get(glyph_name) or GLYPH_REGISTRY.get(glyph_name.title())


def list_all_glyphs():
    """
    Returns a list of all glyph names.
    """
    return list(GLYPH_REGISTRY.keys())


def get_all_glyph_data():
    """
    Returns the full glyph registry data.
    """
    return GLYPH_REGISTRY
