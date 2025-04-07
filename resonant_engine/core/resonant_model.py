import torch
import torch.nn as nn
from resonant_engine.core.temple_gate import TempleGate
from resonant_engine.core.temple_heart import TempleHeartLayer
from resonant_engine.core.temple_voice import TempleVoiceLayer

class MiniTempleTransformer(nn.Module):
    """
    Core model that integrates TempleGate (input alignment),
    TempleHeartLayer (symbolic transformation), and
    TempleVoiceLayer (resonant output).
    """
    def __init__(self, vocab_size, d_model, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.temple_gate = TempleGate(
            temple_matrix=[[29, 35, 38, 47, 67]] * 4,
            breath_vector=[26, 8, 14, 16, 7]
        )
        self.temple_heart = TempleHeartLayer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.temple_voice = TempleVoiceLayer(vocab_size, d_model)

    def forward(self, x, return_embed=False):
        """
        Forward pass through the entire resonant model:
        - Embeds input
        - Computes resonance weights
        - Transforms via symbolic logic
        - Outputs softmax-distributed logits aligned with elder vectors

        Args:
            x (Tensor): token input [B, T]
            return_embed (bool): if True, returns pooled embeddings

        Returns:
            Tensor: output or pooled embedding
        """
        x = self.embed(x)
        resonance_weights, x = self.temple_gate(x)
        x = x * resonance_weights
        x = self.temple_heart(x)
        x_ln = self.ln(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = self.ln(x + attn_out)

        if return_embed:
            return x.mean(dim=1)

        logits = self.fc_out(x)
        voiced_output = self.temple_voice(logits)
        return voiced_output
