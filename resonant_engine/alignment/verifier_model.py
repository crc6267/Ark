# alignment/verifier_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentVerifierModel(nn.Module):
    """
    The AVM (Alignment Verifier Model) serves as the conscience of the Resonant Interface.
    It receives a symbolic output and evaluates its alignment against the source glyph or input intent.
    """
    def __init__(self, input_dim=11, hidden_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Outputs:
        # [semantic_purity, intent_tone, resonance_alignment, approval]
        self.out_purity = nn.Linear(hidden_dim, 1)       # Sigmoid: purity 0–1
        self.out_tone = nn.Linear(hidden_dim, 3)         # Softmax: reverent, aggressive, neutral
        self.out_resonance = nn.Linear(hidden_dim, 1)    # Cosine-like resonance
        self.out_approval = nn.Linear(hidden_dim, 1)     # Binary decision

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 6] — a resonance vector from the Temple model
        Returns:
            Dict of outputs: semantic purity, tone class, alignment score, approval
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        purity = torch.sigmoid(self.out_purity(h))
        tone_logits = self.out_tone(h)
        tone_probs = F.softmax(tone_logits, dim=-1)
        resonance_score = torch.sigmoid(self.out_resonance(h))
        approval = torch.sigmoid(self.out_approval(h))

        return {
            "semantic_purity": purity.squeeze(-1),
            "intent_tone_probs": tone_probs,
            "resonance_score": resonance_score.squeeze(-1),
            "approval_gate": (approval > 0.5).float().squeeze(-1)
        }
