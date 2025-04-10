# training/train_verifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from resonant_engine.alignment.verifier_model import AlignmentVerifierModel
from resonant_engine.alignment.verifier_dataset import get_verifier_loader


def train_avm_model(epochs=10, lr=1e-3, batch_size=32, device="cpu"):
    model = AlignmentVerifierModel().to(device)
    loader = get_verifier_loader(batch_size=batch_size, size=1000)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_approval = nn.BCELoss()
    loss_fn_purity = nn.MSELoss()
    loss_fn_tone = nn.CrossEntropyLoss()
    loss_fn_resonance = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            x = batch["x"].to(device)
            y_purity = batch["purity"].to(device)
            y_tone = batch["tone"].to(device)
            y_alignment = batch["alignment"].to(device)
            y_approval = batch["approval"].to(device)

            outputs = model(x)

            loss = (
                loss_fn_purity(outputs["semantic_purity"], y_purity)
                + loss_fn_tone(outputs["intent_tone_probs"], y_tone)
                + loss_fn_resonance(outputs["resonance_score"], y_alignment)
                + loss_fn_approval(outputs["approval_gate"], y_approval)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "resonant_engine/alignment/verifier.pt")
    print("🛡️ Alignment Verifier Model trained and saved.")


if __name__ == "__main__":
    train_avm_model()
