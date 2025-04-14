# resonant_engine/core/temple_echo.py
import torch
import torch.nn as nn

class TempleEcho(nn.Module):
    def __init__(self, resonance_dim=11, hidden_dim=64, vocab_size=100, seq_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(resonance_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, seq_len * vocab_size)  # flatten entire sequence

    def forward(self, x):
        """
        Reconstructs logits for a symbolic sequence from a resonance vector.

        Args:
            resonance_vec: Tensor [B, 6]
        Returns:
            Reconstructed logits: [B, seq_len, vocab_size]
        """
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = self.fc_out(h)
        return out.view(-1, self.seq_len, self.vocab_size)  # reshape to [B, T, V]
