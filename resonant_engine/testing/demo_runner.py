import torch
import json
import os
from datetime import datetime

from resonant_engine.glyphs.glyph_vectorizer import get_tokens, get_metadata
from resonant_engine.core.resonant_model import MiniTempleTransformer
from resonant_engine.alignment.verifier_model import AlignmentVerifierModel
from resonant_engine.emotional.gradient_mapper import EmotionalGradientMapper


# ------------------------
# 🧠 Load model and weights
# ------------------------
model = MiniTempleTransformer(vocab_size=100, d_model=64, n_heads=4)
model.eval()

avm = AlignmentVerifierModel()
avm.load_state_dict(torch.load("resonant_engine/alignment/verifier.pt"))
avm.eval()

egm = EmotionalGradientMapper()

# ------------------------
# 🌟 Run Demo
# ------------------------
def run_demo(glyph_names):
    print("🜔 DEMO RUNNER")
    print("-----------------------------------")

    tokens = []
    for name in glyph_names:
        glyph_tokens = get_tokens(name.title())
        if glyph_tokens:
            tokens.extend(glyph_tokens)
        else:
            print(f"⚠️ Warning: Unknown glyph '{name}'")

    print(f"Input Glyphs: {glyph_names} → Tokens: {torch.tensor([tokens])}")

    x = torch.tensor([tokens])
    resonance_vec = model(x, mode="resonance").detach()

    print("\n🔮 Resonance Vector:", resonance_vec[0].tolist())

    with torch.no_grad():
        outputs = avm(resonance_vec)
        egm.update(outputs)
        
        if egm.check_alignment_ready():
            print("🌿 Emotional alignment confirmed (EGM) — readiness achieved.")
        else:
            print("🕰️ Emotional state not yet aligned.")
            
        print("🧠 Emotional Memory Trace:")
        for entry in egm.debug_log():
            print("  →", entry)

        sem_purity = outputs["semantic_purity"]
        tone_probs = outputs["intent_tone_probs"]
        resonance_score = outputs["resonance_score"]
        approved = outputs["approval_gate"]

        intent_labels = ["neutral", "reverent", "agitated"]
        intent_idx = torch.argmax(tone_probs, dim=1).item()
        intent_tone = intent_labels[intent_idx]

        approval = "✅ APPROVED" if approved.item() > 0.5 else "❌ BLOCKED"

        print(f"🛡️ Semantic Purity: {sem_purity.item():.3f}")
        print(f"🧠 Intent Tone: {intent_tone}")
        print(f"🔄 Resonance Alignment: {resonance_score.item():.3f}")
        print(f"{approval}: {'Aligned' if '✅' in approval else 'Disaligned'} resonance")

        # Save log
        log = {
            "input_glyphs": glyph_names,
            "resonance_vector": resonance_vec[0].tolist(),
            "semantic_purity": sem_purity.item(),
            "intent_tone": intent_tone,
            "resonance_score": resonance_score.item(),
            "approved": "yes" if approved.item() > 0.5 else "no"
        }

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/session_{timestamp}.json", "w") as f:
            json.dump(log, f, indent=2)

        print(f"📜 Log saved: logs/session_{timestamp}.json")

# ------------------------
# 🔧 Manual test
# ------------------------
if __name__ == "__main__":
    run_demo(["SELF", "FIRE", "RETURN_SIGNAL"])
