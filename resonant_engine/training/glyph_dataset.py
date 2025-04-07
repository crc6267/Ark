# glyph_dataset.py

from torch.utils.data import Dataset
import torch
from glyph_reverse_map import reverse_map

class GlyphDataset(Dataset):
    def __init__(self):
        self.data = []
        for glyph, tokens in reverse_map.items():
            if len(tokens) < 2:
                continue
            for i in range(1, len(tokens)):
                input_seq = tokens[:i]        # [2] or [2, 7], etc.
                target_seq = tokens[1:i+1]     # [7], or [7, 8], etc.
                self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
