import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from structure.structure import MiniTempleTransformer
import numpy as np

# Load trained model
model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2)
model.load_state_dict(torch.load("ark_model.pth", map_location=torch.device('cpu')))
model.eval()

# The training verse index
verse_lookup = [
    "Psalm 23:1",
    "Psalm 42:5",
    "Proverbs 3:5",
    "Proverbs 4:7",
    "Ecclesiastes 1:2",
    "Ecclesiastes 3:1",
    "Psalm 121:1",
    "Psalm 91:1",
    "Proverbs 16:9",
    "Ecclesiastes 7:12",
    "Psalm 27:1",
    "Proverbs 10:12"
]

# Trained motif vectors (same order as training)
trained_motifs = torch.tensor([
    [-0.5852, -0.27, -1.2263, 0.3866, -1.0955, 1.6291, -0.6085, 1.009, 0.5519, 0.0562],
    [0.9737, -0.606, 0.8553, 0.3564, -0.8237, -0.46, 0.9254, -0.8289, 0.7147, 0.1267],
    [0.049, -0.8583, 1.6451, -0.1552, 0.1868, -1.1164, 0.8833, 1.3227, -0.2086, 0.657],
    [0.4517, -1.3312, 0.4657, 0.4709, 0.9879, 1.242, -0.4654, 1.2562, 0.204, 0.5676],
    [-1.2052, -1.0095, 0.3171, 1.5841, -0.4709, -0.5393, 0.096, -0.024, -1.1441, 0.1993],
    [0.5593, -0.3767, 0.8083, -0.4765, 1.2703, -0.7531, 0.2971, -1.4063, -0.3039, -0.051],
    [-0.848, 0.2912, 0.8871, 0.7386, 0.4981, 0.5, 1.33, 1.6413, 0.3984, 0.8315],
    [1.9361, -1.057, 0.0213, -0.0457, 1.1872, -1.4994, 1.2728, 1.293, -0.4003, -0.4807],
    [-0.4796, -0.7658, -1.4751, 1.8859, 0.4975, 0.5423, 2.0125, 0.1118, 0.6642, 0.854],
    [-1.0033, 0.9159, 0.1739, -2.4089, -0.7853, -1.5318, -1.9413, -0.1382, 1.5516, 0.7304],
    [-1.2659, 1.1262, 0.5354, 0.4284, 0.7908, 0.5087, 0.7188, -0.2309, -0.045, -2.8859],
    [2.6037, 1.1826, 0.0562, -1.0999, 0.1353, 0.8496, 0.1048, 0.8969, -0.5459, -1.3334]
])

while True:
    try:
        user_input = input("\nEnter a symbolic chord (e.g. 2 8 0): ").strip()
        chord = list(map(int, user_input.split()))
        input_ids = torch.tensor([chord])

        with torch.no_grad():
            embed = model(input_ids, return_embed=True)

        similarities = F.cosine_similarity(embed, trained_motifs)
        top_index = torch.argmax(similarities).item()

        print(f"\nThe ark sings: {verse_lookup[top_index]}\n")
    except Exception as e:
        print("Error:", e)
        break
