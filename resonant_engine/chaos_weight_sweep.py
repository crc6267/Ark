import numpy as np
import torch
import torch.nn.functional as F
from resonant_model import MiniTempleTransformer
import matplotlib.pyplot as plt

# Define chaos weights
chaos_weights = {
    "refinement (1/40)": 1/40,
    "sacrifice (1/33)": 1/33,
    "order (1/12)": 1/12,
    "emergent chaos (1/4.6692)": 1/4.6692,
    "original (0.05)": 0.05
}

input_sequence = [2, 7, 8, 7]
input_tensor = torch.tensor([input_sequence])

def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9))

results = []

for label, weight in chaos_weights.items():
    class CustomHeart(torch.nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.compress1 = torch.nn.Linear(d_model, 5)
            self.middle = torch.nn.Linear(5, 6)
            self.expand1 = torch.nn.Linear(6, 5)
            self.out_proj = torch.nn.Linear(5, d_model)
            self.chaos_r = 3.5699456

        def forward(self, x):
            x = torch.tanh(self.compress1(x))
            x = torch.sin(x * np.pi)
            chaos = x.clone()
            for _ in range(2):
                chaos = self.chaos_r * chaos * (1 - chaos)
            x = x + weight * chaos
            x = torch.tanh(self.middle(x))
            x = torch.cos(x * np.pi)
            x = F.relu(self.expand1(x))
            x = self.out_proj(x)
            return x

    # Rebuild full model using CustomHeart
    base_model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2)
    base_model.eval()

    class CustomModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.embed = base.embed
            self.temple_gate = base.temple_gate
            self.temple_heart = CustomHeart(d_model=10)
            self.attn = base.attn
            self.ln = base.ln
            self.fc_out = base.fc_out
            self.temple_voice = base.temple_voice

        def forward(self, x):
            x = self.embed(x)
            w, x = self.temple_gate(x)
            x = x * w
            x = self.temple_heart(x)
            x_ln = self.ln(x)
            attn_out, _ = self.attn(x_ln, x_ln, x_ln)
            x = self.ln(x + attn_out)
            logits = self.fc_out(x)
            return self.temple_voice(logits)

    model = CustomModel(base_model)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probs = output[0, 0].numpy()
        entropy = compute_entropy(probs)
        top_token = int(np.argmax(probs))
        results.append((label, weight, top_token, probs[top_token], entropy, probs[38]))

# Print and plot
print(f"{'Chaos Label':<25} | Top Token | Score     | Entropy   | Token38")
print("-" * 70)
for r in results:
    print(f"{r[0]:<25} | {r[2]:>9} | {r[3]:.6f} | {r[4]:.6f} | {r[5]:.6f}")

labels = [r[0] for r in results]
x = np.arange(len(labels))
top_scores = [r[3] for r in results]
entropies = [r[4] for r in results]
token_38s = [r[5] for r in results]

plt.figure(figsize=(12, 6))
plt.plot(x, top_scores, label="Top Token Score", marker='o')
plt.plot(x, entropies, label="Entropy", marker='x')
plt.plot(x, token_38s, label="Token 38 Score", marker='s')
plt.xticks(x, labels, rotation=45)
plt.ylabel("Value")
plt.title("Chaos Weight vs Resonance Metrics")
plt.legend()
plt.tight_layout()
plt.show()
