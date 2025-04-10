# demo_runner.py

import torch
import json
import os
from datetime import datetime
from resonant_engine.glyphs.symbolic_input import prepare_input
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.alignment.verifier_model import AlignmentVerifierModel

# 🔧 Config
input_glyphs = ["SELF", "LIGHT", "VOICE"]
d_model = 64
n_heads = 4
vocab_size = 100  # Adjust to match your vocab size if needed

# 🏛️ Initialize temple model
model = MiniTempleTransformer(vocab_size, d_model, n_heads)
model.eval()

# 🛡️ Load trained AVM
avm = AlignmentVerifierModel()
avm.load_state_dict(torch.load("resonant_engine/alignment/verifier.pt"))
avm.eval()

# 🔮 Run input through the temple
input_tensor = prepare_input(input_glyphs)  # shape: [1, T]
with torch.no_grad():
    resonance_vector = model(input_tensor, mode="resonance")  # [1, 6]
    avm_output = avm(resonance_vector)

# 🪞 AVM Output
purity = avm_output["semantic_purity"].item()
tone_idx = torch.argmax(avm_output["intent_tone_probs"], dim=-1).item()
tone_label = ["reverent", "aggressive", "neutral"][tone_idx]
alignment = avm_output["resonance_score"].item()
approval = avm_output["approval_gate"].item()

# 🖋️ Display Results
print("\n🜔 DEMO RUNNER")
print("-----------------------------------")
print(f"Input Glyphs: {input_glyphs}")
print(f"🔮 Resonance Vector: {resonance_vector.squeeze().tolist()}")
print(f"🛡️ Semantic Purity: {purity:.3f}")
print(f"🧠 Intent Tone: {tone_label}")
print(f"🔄 Resonance Alignment: {alignment:.3f}")
print(f"{ '✅ APPROVED: Signal aligned' if approval == 1 else '❌ BLOCKED: Disaligned resonance' }")

# 📜 Save output to logs
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_data = {
    "glyphs": input_glyphs,
    "vector": resonance_vector.squeeze().tolist(),
    "purity": purity,
    "tone": tone_label,
    "alignment": alignment,
    "approved": bool(approval)
}

with open(f"logs/session_{timestamp}.json", "w") as f:
    json.dump(log_data, f, indent=2)

print(f"📜 Log saved: logs/session_{timestamp}.json")
