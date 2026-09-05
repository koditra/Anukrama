import os
import torch
import torch.nn as nn

BASE_DIR = "/Users/facebook/Anukrama/ai"
MODEL_DIR = os.path.join(BASE_DIR, "models")

class PronunciationModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x) * 100.0

for verse in range(1, 21):
    code = f"v{verse:02d}"

    model_path = os.path.join(
        MODEL_DIR,
        f"anukrama_{code}_score_best.pt"
    )

    output_path = os.path.join(
        MODEL_DIR,
        f"anukrama_{code}_score_best.onnx"
    )

    if not os.path.exists(model_path):
        print(f"Skipping {code}: model not found")
        continue

    scaler_path = os.path.join(
        BASE_DIR,
        "scalers",
        f"anukrama_{code}_scaler.npz"
    )

    if not os.path.exists(scaler_path):
        print(f"Skipping {code}: scaler not found")
        continue

    scaler = __import__("numpy").load(scaler_path)

    input_size = len(scaler["mean"])

    model = PronunciationModel(input_size)
    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state)
    model.eval()

    example_input = torch.randn(
        1,
        input_size
    )

    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=["input"],
        output_names=["score"],
        dynamo=True
    )

    print(f"Exported {code}")
    print(f"  Input size: {input_size}")
    print(f"  Output: {output_path}")

print("Done.")