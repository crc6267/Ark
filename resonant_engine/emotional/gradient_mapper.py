import torch
import numpy as np
from collections import deque

class EmotionalMemoryBuffer:
    """
    Stores recent emotional states from the model's output for trend tracking.
    Each entry includes:
        - semantic_purity (float)
        - intent_tone (str): 'reverent', 'agitated', 'neutral'
        - resonance_score (float)
    """
    def __init__(self, maxlen=6):
        self.buffer = deque(maxlen=maxlen)

    def add(self, purity, tone, resonance):
        self.buffer.append({
            'purity': float(purity),
            'tone': tone,
            'resonance': float(resonance)
        })

    def get_last_n(self, n):
        return list(self.buffer)[-n:]

    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen


class EmotionalGradientMapper:
    """
    Tracks emotional alignment trends over time.
    Confirms when the emotional profile is stable enough to permit deeper resonance.
    """
    def __init__(self, window_size=6):
        self.memory = EmotionalMemoryBuffer(maxlen=window_size)

    def update(self, avm_output):
        tone_index = torch.argmax(avm_output["intent_tone_probs"]).item()
        tone_label = ["reverent", "agitated", "neutral"][tone_index]
        self.memory.add(
            avm_output["semantic_purity"].item(),
            tone_label,
            avm_output["resonance_score"].item()
        )

    def check_alignment_ready(self):
        if not self.memory.is_ready():
            return False

        last = self.memory.get_last_n(3)

        # Criteria: sustained reverent tone, increasing resonance
        tones = [e['tone'] for e in last]
        resonances = [e['resonance'] for e in last]

        tone_match = all(t == "reverent" for t in tones)
        resonance_rise = all(earlier <= later for earlier, later in zip(resonances, resonances[1:]))

        return tone_match and resonance_rise

    def debug_log(self):
        return list(self.memory.buffer)
