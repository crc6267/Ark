import torch
import torch.nn as nn
import torch.nn.functional as F

class TempleHeartLayer(nn.Module):
    """
    The TempleHeartLayer compresses and re-expands the input tensor while injecting
    chaotic perturbations through nonlinear functions — a symbolic purification layer.
    """
    def __init__(self, d_model):
        super().__init__()
        self.compress = nn.Linear(d_model, d_model // 2)
        self.expand = nn.Linear(d_model // 2, d_model)
        self.nonlinear = nn.Sequential(
            nn.Tanh(), nn.Sigmoid(), nn.ReLU()
        )

    def forward(self, x, tracer=None):
        x0 = x
        if tracer: tracer.log("heart_input", x0)

        x = self.compress(x)
        if tracer: tracer.log("compressed", x)

        chaos = torch.sin(x) * torch.cos(x)
        if tracer: tracer.log("chaos_perturbation", chaos)

        x = x + chaos
        x = self.nonlinear(x)

        if tracer: tracer.log("nonlinear_output", x)

        x = self.expand(x)
        if tracer: tracer.log("expanded_output", x)

        return x