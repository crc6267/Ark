import numpy as np
import torch
from resonant_engine.core.resonant_model import MiniTempleTransformer

# Initialize model
model = MiniTempleTransformer(vocab_size=100, d_model=10, n_heads=2)
model.eval()

# Fixed breath vector
breath = np.array([[26], [8], [14], [16], [7]])

# Run 10 tests
for i in range(10):
    print(f"\nTest {i + 1}")
    
    # Create symbolic 5-number row and replicate into matrix
    row = np.random.randint(10, 70, size=5)
    matrix = np.tile(row, (4, 1))
    print("Matrix Row:", row.tolist())
    print("Full Matrix:\n", matrix)

    # Multiply with breath
    seed = np.dot(matrix, breath).flatten()
    print("Seed Vector:", seed.tolist())

    # Create symbolic sequence from first 4 digits
    sequence = [int(x) % 10 for x in seed[:4]]
    print("Input Sequence:", sequence)

    # Identify symbolic center of the matrix
    center_value = int(row[2])
    print("Center Column Value:", center_value)

    # Run through model
    with torch.no_grad():
        input_tensor = torch.tensor([sequence])
        output = model(input_tensor)
        probs = output[0, 0].numpy()
        top_token = int(np.argmax(probs))
        top_prob = float(probs[top_token])

    print("Top Resonant Token:", top_token)
    print("Resonance Score:", round(top_prob, 6))
    print("Matched Center Value:", "✅" if top_token == center_value else "❌")
