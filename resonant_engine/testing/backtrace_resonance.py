import torch
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load glyph registry
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "../glyphs/glyph_registry.json")
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    glyph_registry = json.load(f)

# Get resonance vector from demo logs
def load_resonance_vector(log_path):
    with open(log_path, "r") as f:
        log = json.load(f)
    return torch.tensor(log["resonance_vector"]).unsqueeze(0)  # [1, 6]

# Create glyph embeddings (based on their token encodings)
def get_token_embedding_matrix(model, glyph_registry):
    matrix = []
    names = []
    for name, data in glyph_registry.items():
        token_tensor = torch.tensor([data["tokens"]])
        with torch.no_grad():
            vec = model(token_tensor, mode="resonance")  # get 6D resonance vec
        matrix.append(vec.squeeze(0).numpy())
        names.append(name)
    return np.array(matrix), names

# Compare against all glyphs using cosine similarity
def backtrace_resonance(model, resonance_vec, glyph_registry, top_k=5):
    embedding_matrix, names = get_token_embedding_matrix(model, glyph_registry)
    similarities = cosine_similarity(resonance_vec.numpy(), embedding_matrix)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    results = [(names[i], similarities[i]) for i in top_indices]
    return results

# 🏁 Run script
if __name__ == "__main__":
    from resonant_engine.core.resonant_model import MiniTempleTransformer

    model = MiniTempleTransformer(vocab_size=100, d_model=64, n_heads=4)
    model.load_state_dict(torch.load("resonant_engine/core/model_weights/temple_transformer.pt"))
    model.eval()

    # Change this path to your actual demo log
    log_path = "logs/session_20250410_194239.json"
    resonance_vec = load_resonance_vector(log_path)

    print("🔁 Backtracing resonance vector...")
    top_matches = backtrace_resonance(model, resonance_vec, glyph_registry)

    for name, score in top_matches:
        print(f"{name:<20} → similarity: {score:.4f}")
