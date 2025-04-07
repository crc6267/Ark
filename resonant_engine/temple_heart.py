import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TempleHeartLayer(nn.Module):
    """
    Symbolic layer that simulates transformation through distortion, refinement, and restoration.

    This layer compresses the signal to a 5D space, applies chaotic transformation,
    then expands and finalizes the output through nonlinear functions (tanh, sin, cos).

    Chaos is intentionally introduced to model spiritual transformation — refining disorder
    into a meaningful structure through controlled instability.
    """
    def __init__(self, d_model):
        super().__init__()
        self.compress1 = nn.Linear(d_model, 5)
        self.middle = nn.Linear(5, 6)
        self.expand1 = nn.Linear(6, 5)
        self.out_proj = nn.Linear(5, d_model)
        self.chaos_r = 3.5699456

    def forward(self, x):
        """
        Transforms input through compression, chaos perturbation, and reconstruction.
        Simulates the passage of a signal through a metaphoric 'trial' or inner purification process.

        Args:
            x (Tensor): input sequence of shape [B, T, D]

        Returns:
            Tensor: transformed output of shape [B, T, D]
        """
        x = torch.tanh(self.compress1(x))
        x = torch.sin(x * math.pi)
        chaos = x.clone()
        for _ in range(2):
            chaos = self.chaos_r * chaos * (1 - chaos)
        x = x + 0.05 * chaos
        x = torch.tanh(self.middle(x))
        x = torch.cos(x * math.pi)
        x = F.relu(self.expand1(x))
        x = self.out_proj(x)
        return x