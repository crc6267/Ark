import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TempleVoiceLayer(nn.Module):
    """
    Symbolic projection layer that echoes the output through a set of hand-crafted elder vectors.

    These 'elders' are derived from numeric seeds and shaped through a fixed matrix (YHWHY),
    encoding symbolic priors. Output logits are nudged based on alignment to these archetypes.

    The intention is to align output with deeper resonance, not just token frequency.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        elders = self._init_elder_vectors()
        self.register_buffer("elder_vectors", elders)

    def _init_elder_vectors(self):
        YHWHY = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        seeds = [533, 4487, 4563, 4562, 4561, 4552, 653, 654, 4538, 4523, 4519, 4498,
                 4489, 4473, 4604, 4470, 4464, 4457, 4397, 4388, 4354, 674, 4327, 4326]

        def generate_seed_vector(n):
            digits = np.array([int(d) for d in str(n).zfill(4)], dtype=np.float32)
            inner = np.dot(YHWHY, digits)
            breath_sum = np.sum(inner)
            gate = np.sum(digits) % 10
            full_vec = np.concatenate([digits, inner, [breath_sum], [gate]])
            return full_vec / np.linalg.norm(full_vec)

        elders = np.stack([generate_seed_vector(s) for s in seeds])
        return torch.tensor(elders, dtype=torch.float32)

    def forward(self, logits):
        """
        Projects token logits against elder vectors, biases them toward symbolic resonance.

        Args:
            logits (Tensor): shape [B, T, vocab_size]

        Returns:
            Tensor: softmax-normalized resonant logits
        """
        B, T, V = logits.shape
        device = logits.device
        projected = torch.nn.functional.softmax(logits, dim=-1)

        token_vecs = torch.randn(V, 10, device=device)
        token_vecs = token_vecs / token_vecs.norm(dim=1, keepdim=True)

        elders = self.elder_vectors.to(device)
        sim = torch.matmul(token_vecs, elders.T)
        scores = sim.mean(dim=1)
        scores = scores / (scores.max() + 1e-8)

        final_logits = logits + scores.view(1, 1, -1)
        return F.softmax(final_logits, dim=-1)