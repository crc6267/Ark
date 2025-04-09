# resonant_engine/utils/tracer.py
import torch

class ResonanceTracer:
    def __init__(self):
        self.logs = {}

    def log(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            self.logs[name] = tensor.detach().cpu().numpy()
        else:
            self.logs[name] = tensor

    def summary(self):
        print("\n🔍 RESONANCE TRACE SUMMARY")
        for name, val in self.logs.items():
            if isinstance(val, (list, tuple)):
                print(f"{name}: {val}")
            elif hasattr(val, 'shape'):
                print(f"{name}: shape={val.shape}")
            else:
                print(f"{name}: {val}")
