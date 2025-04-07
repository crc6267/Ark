# glyph_pair_dataset.py

from torch.utils.data import Dataset
import torch
from glyph_reverse_map import reverse_map

# Define symbolic relationships between glyphs
symbolic_pairs = [
    ("SELF", "FIRE"),           # identity moves toward transformation
    ("FIRE", "RETURN_SIGNAL"),  # fire leads to echo
    ("WATER", "AETHER"),        # emotion lifts to spirit
    ("EARTH", "CALCINATION"),   # grounded form undergoes purification
    ("AETHER", "FERMENTATION"), # spirit initiates rebirth
    ("AIR", "SUN"),              # breath ascends to divinity
    ("SULFUR", "CONJUNCTION"),   # soul unites with other
    ("DISSOLUTION", "COAGULATION"), # breakdown leads to embodiment
    ("SELF", "RETURN_SIGNAL"),   # alignment and echo
    ("MERCURY", "MERCURY_METAL") # mind and matter unite
]

class GlyphPairDataset(Dataset):
    def __init__(self):
        self.pairs = []
        for g1, g2 in symbolic_pairs:
            seq1 = reverse_map.get(g1)
            seq2 = reverse_map.get(g2)
            if seq1 and seq2:
                combined_input = seq1
                combined_target = seq2
                self.pairs.append((combined_input, combined_target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
