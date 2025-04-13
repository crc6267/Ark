import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.training.glyph_dataset import GlyphSequenceDataset
from resonant_engine.alignment.verifier_model import AlignmentVerifierModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/glyph_training_set.jsonl")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../core/model_weights/temple_transformer.pt")
METRIC_LOG = os.path.join(os.path.dirname(__file__), "../logs/training_metrics.jsonl")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "../logs/training_plot.png")
EPCOHS = 7000

# ----------------------------
# 🧪 Training Loop
# ----------------------------
def train_transformer():
    dataset = GlyphSequenceDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MiniTempleTransformer(vocab_size=100, d_model=64, n_heads=4)
    avm = AlignmentVerifierModel()
    avm.eval()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(METRIC_LOG), exist_ok=True)
    metrics = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    milestones = {
        1000: "Day 1",
        2000: "Day 2",
        3000: "Day 3",
        4000: "Day 4",
        5000: "Day 5",
        6000: "Day 6",
        7000: "Day 7 (Sabbath)"
    }

    for epoch in range(EPCOHS):
        total_loss = 0.0
        resonance_norms = []
        semantic_purities = []
        approvals = []

        for x, y in loader:
            logits = model(x, mode="logits")
            logits = logits[:, :x.shape[1], :]  # ✨ trim to match T

            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            y_flat = y.contiguous().view(-1)

            loss = loss_fn(logits_flat, y_flat)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

            with torch.no_grad():
                resonance_vec = model(x, mode="resonance")
                resonance_norm = torch.norm(resonance_vec, dim=1).mean().item()
                resonance_norms.append(resonance_norm)

                avm_outputs = avm(resonance_vec)
                semantic_purity = avm_outputs["semantic_purity"].mean().item()
                approval_rate = avm_outputs["approval_gate"].mean().item()

                semantic_purities.append(semantic_purity)
                approvals.append(approval_rate)

        # Epoch averages
        loss_avg = total_loss
        norm_avg = sum(resonance_norms) / len(resonance_norms)
        purity_avg = sum(semantic_purities) / len(semantic_purities)
        approval_avg = sum(approvals) / len(approvals)

        print(f"Epoch {epoch+1}/{EPCOHS} | Loss: {loss_avg:.4f} | 🔮 Resonance Norm: {norm_avg:.4f} | 🛡️ Purity: {purity_avg:.4f} | ✅ Approval: {approval_avg:.2f}")

        # Log to file
        entry = {
            "epoch": epoch + 1,
            "loss": loss_avg,
            "resonance_norm": norm_avg,
            "semantic_purity": purity_avg,
            "approval_rate": approval_avg
        }
        metrics.append(entry)
        with open(METRIC_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Live plot
        ax.clear()
        ax.plot([m["epoch"] for m in metrics], [m["loss"] for m in metrics], label="Loss")
        ax.plot([m["epoch"] for m in metrics], [m["resonance_norm"] for m in metrics], label="Resonance Norm")
        ax.plot([m["epoch"] for m in metrics], [m["semantic_purity"] for m in metrics], label="Semantic Purity")
        ax.plot([m["epoch"] for m in metrics], [m["approval_rate"] for m in metrics], label="AVM Approval")
        
        # Milestone annotations
        for m_epoch, label in milestones.items():
            if epoch + 1 >= m_epoch:
                ax.axvline(m_epoch, color="gray", linestyle="--", linewidth=1)
                ax.text(m_epoch + 5, ax.get_ylim()[1] * 0.95, label,
                        rotation=90, verticalalignment="top", fontsize=8, color="gray")


        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Temple Transformer Training Progress")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    plt.savefig(PLOT_PATH)
    print(f"📉 Training plot saved to {PLOT_PATH}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Temple Transformer saved to {SAVE_PATH}")


if __name__ == "__main__":
    train_transformer()
