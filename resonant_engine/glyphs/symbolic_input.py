# symbolic_input.py

from resonant_engine.archive.glyph_reverse_map import reverse_map
import torch

def prepare_input(glyph_names):
    """
    Given a list of glyph names, converts them to a flattened list of token IDs.
    """
    tokens = []
    for name in glyph_names:
        sequence = reverse_map.get(name.upper())
        if sequence:
            tokens.extend(sequence)
        else:
            print(f"Warning: Unknown glyph '{name}'")
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)



if __name__ == "__main__":
    seq = prepare_input("SELF", "FIRE", "RETURN_SIGNAL")
    print("Input Sequence:", seq)
