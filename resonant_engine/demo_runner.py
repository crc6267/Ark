import torch
from resonant_model import MiniTempleTransformer

if __name__ == "__main__":
    model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2)
    model.eval()

    input_ids = torch.tensor([[2, 7, 8, 7]])  # Example input sequence
    with torch.no_grad():
        output = model(input_ids)

    print("\nInput:", input_ids.tolist())
    print("Output Probabilities (first token):")
    print(output[0, 0].numpy())
