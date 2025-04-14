# resonant_engine/analysis/symbolic_resonator.py

import torch
import torch.nn as nn
import torch.fft

class SymbolicResonator(nn.Module):
    def __init__(self, embedding_layer, max_len=12):
        super().__init__()
        self.embedding = embedding_layer

        # Create learnable or fixed positional encodings
        pe = torch.zeros(max_len, embedding_layer.embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_layer.embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_layer.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

    def forward(self, token_ids: torch.Tensor):
        """
        Args:
            token_ids: Tensor of shape [B, T] (batch of token sequences)
        Returns:
            freq_profile: Tensor of shape [B, F]
        """
        embedded = self.embedding(token_ids)  # [B, T, D]

        # Add position info
        pe = self.positional_encoding[:embedded.shape[1], :].unsqueeze(0)  # [1, T, D]
        embedded = embedded + pe

        fft_result = torch.fft.fft(embedded, dim=1)
        magnitude = torch.abs(fft_result)
        freq_profile = magnitude.mean(dim=-1)  # [B, F]

        return freq_profile

