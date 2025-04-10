# alignment/verifier_dataset.py

import torch
from torch.utils.data import Dataset
import random

TONE_LABELS = ["reverent", "aggressive", "neutral"]
TONE_TO_INDEX = {label: i for i, label in enumerate(TONE_LABELS)}

class VerifierDataset(Dataset):
    """
    Synthetic dataset for training the Alignment Verifier Model.
    Each sample includes a resonance vector and labeled alignment metadata.
    """
    def __init__(self, size=1000, seed=42):
        super().__init__()
        self.samples = []
        random.seed(seed)

        for _ in range(size):
            vec = [random.uniform(-1, 1) for _ in range(6)]
            cosine = sum([v ** 2 for v in vec]) ** 0.5 / 6.0
            purity = min(max(cosine + random.uniform(-0.1, 0.1), 0), 1)
            tone = random.choices(TONE_LABELS, weights=[0.5, 0.2, 0.3])[0]
            alignment = min(max(purity * 0.8 + random.uniform(-0.05, 0.05), 0), 1)
            approve = 1 if purity > 0.5 and alignment > 0.5 and tone == "reverent" else 0

            self.samples.append({
                "vector": vec,
                "purity": purity,
                "tone_label": tone,
                "tone_index": TONE_TO_INDEX[tone],
                "alignment": alignment,
                "approval": approve,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "x": torch.tensor(sample["vector"], dtype=torch.float32),
            "purity": torch.tensor(sample["purity"], dtype=torch.float32),
            "tone": torch.tensor(sample["tone_index"], dtype=torch.long),
            "alignment": torch.tensor(sample["alignment"], dtype=torch.float32),
            "approval": torch.tensor(sample["approval"], dtype=torch.float32),
        }

# Utility: create DataLoader
from torch.utils.data import DataLoader

def get_verifier_loader(batch_size=32, size=1000):
    dataset = VerifierDataset(size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
