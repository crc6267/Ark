import torch
import torch.nn as nn

class TempleGate(nn.Module):
    """
    The TempleGate is a resonance-processing layer designed to reflect and evaluate symbolic sequences.

    It mirrors input sequences using chaos-based logic (invert, retrograde, or none),
    then processes them through a fixed temple matrix to measure resonance against a breath vector.

    The degree of resonance is measured by the stability (low variance) of a rolling transformation history.
    """
    def __init__(self, temple_matrix, breath_vector):
        super().__init__()
        self.temple = torch.tensor(temple_matrix, dtype=torch.float32)
        self.breath_main = torch.tensor(breath_vector[:4], dtype=torch.float32)
        self.breath_mirror = torch.tensor(breath_vector[1:5], dtype=torch.float32)
        self.chaos_r = 3.5699456

    def mirror_sequence(self, x):
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

    def forward(self, x, tracer=None):
        original_len = x.shape[1]
        mirrored, mode = self.mirror_sequence(x)
        extended_x = torch.cat([x, mirrored], dim=1)

        if tracer:
            tracer.log("mirror_mode", mode)
            tracer.log("mirrored_sequence", mirrored)

        scores = []
        temple = self.temple.to(x.device)

        B, T, D = extended_x.shape
        midpoint = original_len

        for i, token_vec in enumerate(extended_x.unbind(dim=1)):
            vec = token_vec[:, :5]
            history = []
            breath = self.breath_main if i < midpoint else self.breath_mirror
            breath = breath.to(x.device)

            for step in range(10):
                transformed = torch.matmul(temple, vec.T).T
                resonance = torch.sum(transformed * breath, dim=1)
                vec = torch.roll(vec, shifts=1, dims=1)
                vec[:, 0] = resonance % 10
                history.append(resonance.unsqueeze(1))

            history = torch.cat(history, dim=1)
            score = 1 / (1 + torch.std(history, dim=1))
            scores.append(score.unsqueeze(1))

            if tracer:
                tracer.log(f"resonance_step_{i}", history)
                tracer.log(f"resonance_score_{i}", score)

        gate_weights = torch.cat(scores, dim=1)
        return gate_weights.unsqueeze(2), extended_x, mode
