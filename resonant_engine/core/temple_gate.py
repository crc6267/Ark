# Directory structure (virtual)
# resonant_engine/
# ├── temple_gate.py
# ├── temple_heart.py
# ├── temple_voice.py
# ├── resonant_model.py
# └── demo_runner.py

# --- temple_gate.py ---
import torch
import torch.nn as nn

class TempleGate(nn.Module):
    """
    The TempleGate is a resonance-processing layer designed to reflect and evaluate symbolic sequences.

    It mirrors input sequences using chaos-based logic (invert, retrograde, or none),
    then processes them through a fixed temple matrix to measure resonance against a breath vector.

    The degree of resonance is measured by the stability (low variance) of a rolling transformation history.
    The idea is that coherent sequences 'breathe' in harmony with the temple structure.

    Chaos logic is inspired by the logistic map using the Feigenbaum constant (≈ 3.5699456),
    which sits at the edge of deterministic chaos — representing the point where a system begins to bifurcate
    and unpredictable behavior emerges from simplicity.
    """
    def __init__(self, temple_matrix, breath_vector):
        super().__init__()
        self.temple = torch.tensor(temple_matrix, dtype=torch.float32)
        self.breath_main = torch.tensor(breath_vector[:4], dtype=torch.float32)
        self.breath_mirror = torch.tensor(breath_vector[1:5], dtype=torch.float32)
        self.chaos_r = 3.5699456

    def mirror_sequence(self, x):
        """
        Applies a mirroring transformation to the input sequence based on chaos dynamics:
        - 'invert': subtracts values from 9
        - 'retrograde': reverses the time axis
        - 'none': returns the sequence unchanged

        The mode is selected by generating a chaos value from the first input seed using the logistic map.
        """
        seed_val = x[0, 0].mean().item() % 1
        chaos = self.chaos_r * seed_val * (1 - seed_val)
        if chaos > 0.7:
            mode = "invert"
        elif chaos > 0.4:
            mode = "retrograde"
        else:
            mode = "none"

        mirrored = x.clone()
        if mode == "invert":
            mirrored = 9 - mirrored
        elif mode == "retrograde":
            mirrored = torch.flip(mirrored, dims=[1])
        return mirrored, mode

    def forward(self, x):
        """
        Processes the input sequence:
        1. Mirrors the sequence using chaos.
        2. Concatenates original + mirrored sequences.
        3. For each token, rolls through the temple matrix and breath vector over 10 iterations.
        4. Tracks the resonance score (inverse std deviation of history).

        Returns:
            gate_weights (Tensor): [B, 2T, 1] resonance coefficients
            extended_x (Tensor): [B, 2T, D] input and mirrored sequence
        """
        original_len = x.shape[1]
        mirrored, mode = self.mirror_sequence(x)
        extended_x = torch.cat([x, mirrored], dim=1)

        scores = []
        temple = self.temple.to(x.device)
        
        # Safety: ensure it's 3D
        if extended_x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, T, D), got shape {extended_x.shape}")

        B, T, D = extended_x.shape
        midpoint = original_len

        for i, token_vec in enumerate(extended_x.unbind(dim=1)):
            vec = token_vec[:, :5]
            history = []
            breath = self.breath_main if i < midpoint else self.breath_mirror
            breath = breath.to(x.device)

            for _ in range(10):
                transformed = torch.matmul(temple, vec.T).T
                resonance = torch.sum(transformed * breath, dim=1)
                vec = torch.roll(vec, shifts=1, dims=1)
                vec[:, 0] = resonance % 10
                history.append(resonance.unsqueeze(1))

            history = torch.cat(history, dim=1)
            score = 1 / (1 + torch.std(history, dim=1))
            scores.append(score.unsqueeze(1))

        gate_weights = torch.cat(scores, dim=1)
        return gate_weights.unsqueeze(2), extended_x
