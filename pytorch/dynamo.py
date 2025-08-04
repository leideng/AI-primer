import torch
import torch.nn as nn
import torch.optim as optim
import time



# Training step
def train_step(optimizer,model,loss_fn,x,y):
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()


# Simple MLP model
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

torch.set_float32_matmul_precision('high')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate random data
batch_size = 4096
in_features = 10240
hidden_features = 20480
out_features = 10
x = torch.randn(batch_size, in_features, device=device)
y = torch.randint(0, 10, (batch_size,), device=device)


# Instantiate model and move to device
model1 = SimpleMLP(in_features=in_features,hidden_features=hidden_features,out_features=out_features).to(device)
optimizer1 = optim.Adam(model1.parameters())
loss_fn1 = nn.CrossEntropyLoss()

# Warmup
warmup_steps = 10
for _ in range(warmup_steps):
    train_step(optimizer1,model1,loss_fn1,x,y)



# Measure uncompiled
torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    train_step(optimizer1,model1,loss_fn1,x,y)
torch.cuda.synchronize()
end = time.time()
print(f"Uncompiled time: {end - start:.4f} seconds")

# Instantiate model and move to device
model2 = SimpleMLP(in_features=in_features,hidden_features=hidden_features,out_features=out_features).to(device)
optimizer2 = optim.Adam(model2.parameters())
loss_fn2 = nn.CrossEntropyLoss()
# warmup
for _ in range(warmup_steps):
    train_step(optimizer2,model2,loss_fn2,x,y)


# Measure compiled
torch.cuda.synchronize()

# Compile with torch.compile (PyTorch 2.0+)
compiled_model = torch.compile(model2)
start = time.time()
for _ in range(50):
    train_step(optimizer2,compiled_model,loss_fn2,x,y)
torch.cuda.synchronize()
end = time.time()
print(f"Compiled (torch.compile) time: {end - start:.4f} seconds")
