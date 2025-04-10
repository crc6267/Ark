# alignment/test_verifier.py

import torch
from resonant_engine.alignment.verifier_model import AlignmentVerifierModel

# Load trained model
model = AlignmentVerifierModel()
model.load_state_dict(torch.load("resonant_engine/alignment/verifier.pt"))
model.eval()

# Example test vectors (you can replace with live outputs from your transformer)
test_vectors = [
    [0.2, 0.5, -0.3, 0.1, 0.4, -0.5],   # Possibly aligned
    [-0.9, -0.7, 0.8, -0.6, 0.9, -0.2],  # Likely chaotic
    [0.0, 0.1, 0.0, 0.1, 0.0, 0.1],      # Neutral test
]

for i, vec in enumerate(test_vectors):
    x = torch.tensor(vec).unsqueeze(0)  # [1, 6]
    with torch.no_grad():
        result = model(x)

    tone_index = torch.argmax(result["intent_tone_probs"], dim=-1).item()
    tone_label = ["reverent", "aggressive", "neutral"][tone_index]

    print(f"\n🧪 Test Vector {i + 1}:")
    print(f"Vector: {vec}")
    print(f"Semantic Purity: {result['semantic_purity'].item():.3f}")
    print(f"Intent Tone: {tone_label}")
    print(f"Resonance Score: {result['resonance_score'].item():.3f}")
    print(f"Approval: {'✅' if result['approval_gate'].item() == 1 else '❌'}")
