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

    def forward(self, x, mode="logits", tracer=None):
        """
        Forward pass through the entire resonant model.

        Args:
            x (Tensor): token input [B, T]
            mode (str): "logits" | "resonance" | "embed"
            tracer (ResonanceTracer): optional tracer for visualization

        Returns:
            Tensor: output logits, 6D vector, or pooled embedding
        """
        x = self.embed(x)
        if tracer: tracer.log("embedding", x)

        resonance_weights, x, mirror_mode = self.temple_gate(x, tracer=tracer)
        if tracer:
            tracer.log("mirroring_mode", mirror_mode)
            tracer.log("resonance_weights", resonance_weights)
            tracer.log("gate_output", x)

        x = x * resonance_weights
        if tracer: tracer.log("gated_input", x)

        x = self.temple_heart(x, tracer=tracer)
        if tracer: tracer.log("temple_heart_output", x)

        x_ln = self.ln(x)

        attn_out, attn_weights = self.attn(x_ln, x_ln, x_ln)
        if tracer: tracer.log("attention_weights", attn_weights)

        x = self.ln(x + attn_out)

        if mode == "embed":
            return x.mean(dim=1)  # pooled hidden state

        elif mode == "resonance":
            pooled = x.mean(dim=1)
            resonance_vec = self.resonance_head(pooled)
            if tracer: tracer.log("resonance_vector", resonance_vec)
            return resonance_vec  # 🔮 returns [B, 6]

        else:  # "logits" (default)
            logits = self.fc_out(x)
            return self.temple_voice(logits)
