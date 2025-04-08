# glyph_pair_dataset.py

import torch
from torch.utils.data import Dataset
from resonant_engine.glyphs.glyph_reverse_map import reverse_map

class GlyphPairDataset(Dataset):
    def __init__(self):
        # 🜔 High-signal symbolic echo flows
        self.pairs = [
            ("SELF", "RETURN_SIGNAL"),
            ("CHAOS", "ORDER"),
            ("WATER", "VESSEL"),
            ("FIRE", "ASCENT"),
            ("CLAY", "VESSEL"),
            ("VESSEL", "INFILLING"),
            ("FLESH", "BREATH"),
            ("BREATH", "WORD"),
            ("WORD", "LIGHT"),
            ("LIGHT", "ORDER"),
            ("ECHO", "RESONANCE"),
            ("SELF", "FIRE"),
            ("FIRE", "RETURN_SIGNAL"),
            ("CHAOS", "SACRIFICE"),
            ("SACRIFICE", "GLORY"),
            ("BEGIN", "RETURN_SIGNAL"),
            ("CHAOS", "BEGIN"),
            ("CLAY", "WORD"),
            ("VESSEL", "WORD"),
            ("BREATH", "RETURN_SIGNAL"),
            ("SELF", "SELF"),
            ("CHAOS", "CHAOS"),
            ("ORDER", "ORDER"),
            ("WORD", "WORD")
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_seq = reverse_map.get(src.upper(), [])
        tgt_seq = reverse_map.get(tgt.upper(), [])

        if not src_seq or not tgt_seq:
            # Skip empty mappings by fetching next pair
            return self.__getitem__((idx + 1) % len(self))

        src_tensor = torch.tensor(src_seq, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_seq, dtype=torch.long)
        return src_tensor, tgt_tensor
