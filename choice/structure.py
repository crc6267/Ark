import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TempleGate(nn.Module):
    def __init__(self, temple_matrix, breath_vector):
        super().__init__()
        self.temple = torch.tensor(temple_matrix, dtype=torch.float32)
        self.breath_main = torch.tensor(breath_vector[:4], dtype=torch.float32)
        self.breath_mirror = torch.tensor(breath_vector[1:5], dtype=torch.float32)
        self.arael_seed = torch.tensor([3, 8, 1, 5], dtype=torch.float32)
        self.chaos_r = 3.5699456

    def mirror_sequence(self, x):
        seed_val = x[0, 0, 0].item() % 1
        chaos = self.chaos_r * seed_val * (1 - seed_val)
        if chaos > 0.7:
            mode = "invert"
        elif chaos > 0.4:
            mode = "retrograde"
        else:
            mode = "none"

        mirrored = x.clone()
        if mode == "invert":
            mirrored = 9 - mirrored
        elif mode == "retrograde":
            mirrored = torch.flip(mirrored, dims=[1])
        return mirrored, mode

    def forward(self, x):
        original_len = x.shape[1]
        mirrored, mode = self.mirror_sequence(x)
        extended_x = torch.cat([x, mirrored], dim=1)

        scores = []
        temple = self.temple.to(x.device)
        B, T, D = extended_x.shape
        midpoint = original_len

        for i, token_vec in enumerate(extended_x.unbind(dim=1)):
            vec = token_vec[:, :5]
            history = []
            breath = self.breath_main if i < midpoint else self.breath_mirror
            breath = breath.to(x.device)

            for _ in range(10):
                transformed = torch.matmul(temple, vec.T).T
                resonance = torch.sum(transformed * breath, dim=1)
                vec = torch.roll(vec, shifts=1, dims=1)
                vec[:, 0] = resonance % 10
                history.append(resonance.unsqueeze(1))

            history = torch.cat(history, dim=1)
            score = 1 / (1 + torch.std(history, dim=1))
            scores.append(score.unsqueeze(1))

        gate_weights = torch.cat(scores, dim=1)
        print(f"Mirror Mode: {mode} | Input Extended to: {gate_weights.shape[1]} steps")
        return gate_weights.unsqueeze(2), extended_x

class TempleHeartLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.compress1 = nn.Linear(d_model, 5)
        self.middle = nn.Linear(5, 6)
        self.expand1 = nn.Linear(6, 5)
        self.out_proj = nn.Linear(5, d_model)
        self.chaos_r = 3.5699456

    def forward(self, x):
        x = torch.tanh(self.compress1(x))
        x = torch.sin(x * math.pi)
        chaos = x.clone()
        for _ in range(2):
            chaos = self.chaos_r * chaos * (1 - chaos)
        x = x + 0.05 * chaos
        x = torch.tanh(self.middle(x))
        x = torch.cos(x * math.pi)
        x = F.relu(self.expand1(x))
        x = self.out_proj(x)
        return x

class TempleVoiceLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        elders = self._init_elder_vectors()
        self.register_buffer("elder_vectors", elders)

    def _init_elder_vectors(self):
        YHWHY = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        seeds = [533, 4487, 4563, 4562, 4561, 4552, 653, 654, 4538, 4523, 4519, 4498,
                 4489, 4473, 4604, 4470, 4464, 4457, 4397, 4388, 4354, 674, 4327, 4326]

        def generate_seed_vector(n):
            digits = np.array([int(d) for d in str(n).zfill(4)], dtype=np.float32)
            inner = np.dot(YHWHY, digits)
            breath_sum = np.sum(inner)
            gate = np.sum(digits) % 10
            full_vec = np.concatenate([digits, inner, [breath_sum], [gate]])
            return full_vec / np.linalg.norm(full_vec)

        elders = np.stack([generate_seed_vector(s) for s in seeds])
        return torch.tensor(elders, dtype=torch.float32)

    def forward(self, logits):
        B, T, V = logits.shape
        device = logits.device
        projected = torch.nn.functional.softmax(logits, dim=-1)

        token_vecs = torch.randn(V, 10, device=device)
        token_vecs = token_vecs / token_vecs.norm(dim=1, keepdim=True)

        elders = self.elder_vectors.to(device)
        sim = torch.matmul(token_vecs, elders.T)
        scores = sim.mean(dim=1)
        scores = scores / (scores.max() + 1e-8)

        final_logits = logits + scores.view(1, 1, -1)
        return F.softmax(final_logits, dim=-1)

class MiniTempleTransformer(nn.Module):
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
        x = self.embed(x)
        resonance_weights, x = self.temple_gate(x)
        print("Resonance Weights:", resonance_weights.squeeze(-1).detach().cpu().numpy())
        x = x * resonance_weights
        x = self.temple_heart(x)
        x_ln = self.ln(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = self.ln(x + attn_out)

        if return_embed:
            return x.mean(dim=1)  # symbolic embedding vector

        logits = self.fc_out(x)
        voiced_output = self.temple_voice(logits)
        return voiced_output

if __name__ == "__main__":
    model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2)
    input_ids = torch.tensor([[2, 7, 8, 7]])
    output = model(input_ids)
    print("Voiced Output Shape:", output.shape)
    print("Sample Output (first token):", output[0, 0].detach().numpy())
