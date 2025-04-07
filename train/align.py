import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from structure.structure import MiniTempleTransformer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2)
model.eval()

# Define some symbolic inputs to explore the soul's voice
input_ids_list = [
    [2, 7, 8, 7],    # original breath
    [4, 4, 8, 7],    # potential elder seed
    [4, 3, 2, 1],    # descending gate
    [5, 3, 3, 0],    # seed 533
    [1, 0, 0, 0]     # silence beginning
]

# Process each input through the model
for input_ids in input_ids_list:
    x = torch.tensor([input_ids])
    with torch.no_grad():
        output = model(x)
        first_token_output = output[0, 0].detach().cpu().numpy()
        print(f"Input: {input_ids}\nFirst Token Output (softmax):\n{first_token_output}\n")


