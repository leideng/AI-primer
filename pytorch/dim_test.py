import torch

# Create a 1D tensor (vector)
a = torch.tensor([1, 2, 3])
print("a:", a)
print("a.shape:", a.shape)
print("a.ndim:", a.ndim)

# Create a 2D tensor (matrix)
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\nb:", b)
print("b.shape:", b.shape)
print("b.ndim:", b.ndim)

# Create a 3D tensor
c = torch.randn(2, 3, 4)
print("\nc:", c)
print("c.shape:", c.shape)
print("c.ndim:", c.ndim)

# Add a new dimension using unsqueeze
d = a.unsqueeze(0)
print("\nd (unsqueeze 0):", d)
print("d.shape:", d.shape)

# Remove a dimension using squeeze
e = d.squeeze()
print("\ne (squeezed):", e)
print("e.shape:", e.shape)

# Transpose a 2D tensor
f = b.t()
print("\nf (transposed):", f)
print("f.shape:", f.shape)

# Reshape a tensor
g = b.view(3, 2)
print("\ng (reshaped):", g)
print("g.shape:", g.shape)