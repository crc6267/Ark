import json
import torch
from torch.utils.data import Dataset

class GlyphSequenceDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.records = [
                json.loads(line)
                for line in f
                if len(json.loads(line)["tokens"]) == 4
            ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tokens = self.records[idx]["tokens"]
        x = torch.tensor(tokens[:-1], dtype=torch.long)  # [3]
        y = torch.tensor(tokens[1:], dtype=torch.long)   # [3]
        return x, y
