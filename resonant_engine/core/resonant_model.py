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

        # 🆕 New head for projecting to 6D resonance vector
        self.resonance_head = nn.Linear(d_model, 6)

    def forward(self, x, mode="logits"):
        """
        Forward pass through the entire resonant model.

        Args:
            x (Tensor): token input [B, T]
            mode (str): "logits" | "resonance" | "embed"

        Returns:
            Tensor: output logits, 6D vector, or pooled embedding
        """
        x = self.embed(x)
        resonance_weights, x = self.temple_gate(x)
        x = x * resonance_weights
        x = self.temple_heart(x)
        x_ln = self.ln(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = self.ln(x + attn_out)

        if mode == "embed":
            return x.mean(dim=1)  # pooled hidden state

        elif mode == "resonance":
            pooled = x.mean(dim=1)
            return self.resonance_head(pooled)  # 🔮 returns [B, 6]

        else:  # "logits" (default)
            logits = self.fc_out(x)
            return self.temple_voice(logits)
