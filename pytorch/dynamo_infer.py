import torch
import torch.nn as nn
import time

# Simple MLP model for inference
class SimpleMLP(nn.Module):
    def __init__(self, in_features=1024, hidden_features=2048, out_features=10):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.seq(x)

# Set matmul precision for better performance on L4
torch.set_float32_matmul_precision('high')

# Check for L4 GPU (compute capability 8.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate random input data
n_times = 5
batch_size = 1024*n_times
in_features = 128*n_times
hidden_features = 256*n_times
out_features = 10*n_times
x = torch.randn(batch_size, in_features, device=device)

# Instantiate model and move to device
model = SimpleMLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features).to(device)
model.eval()

# Warmup
with torch.no_grad():
    for _ in range(10):
        _ = model(x)

# Measure uncompiled inference
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(50):
        _ = model(x)
torch.cuda.synchronize()
end = time.time()
print(f"Uncompiled inference time: {end - start:.4f} seconds")

# Compile model with torch.compile (PyTorch 2.0+)
model.compile()
model.eval()

# Warmup compiled model
with torch.no_grad():
    for _ in range(10):
        _ = model(x)

# Measure compiled inference
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(50):
        _ = model(x)
torch.cuda.synchronize()
end = time.time()
print(f"Compiled (torch.compile) inference time: {end - start:.4f} seconds")
