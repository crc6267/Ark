import torch
import torch.nn as nn
import torch.nn.functional as F


class TempleVoice(nn.Module):
    """
    The TempleVoice module converts resonance vectors into symbolic activation patterns.
    It is designed to interpret the distilled signal from the TempleHeart and speak the core essence.
    """
    def __init__(self, input_dim=6, hidden_dim=64, glyph_vocab_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, glyph_vocab_size)

    def forward(self, x, tracer=None):
        """
        Args:
            x: Tensor of shape [B, T, V] — raw logits from final transformer layer
            tracer: (optional) ResonanceTracer for visualization

        Returns:
            Transformed logits with symbolic voice modulation
        """
        h = self.fc1(x)
        if tracer: tracer.log("temple_voice_fc1", h)

        h = F.relu(h)
        h = self.fc2(h)
        if tracer: tracer.log("temple_voice_fc2", h)

        h = F.relu(h)
        out = self.fc_out(h)
        if tracer: tracer.log("temple_voice_output", out)

        return out
