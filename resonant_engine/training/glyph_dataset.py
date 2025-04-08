from torch.utils.data import Dataset
import torch

class GlyphDataset(Dataset):
    def __init__(self, glyph_data):
        self.data = glyph_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        return torch.tensor(tokens[:-1], dtype=torch.long), torch.tensor(tokens[1:], dtype=torch.long)
