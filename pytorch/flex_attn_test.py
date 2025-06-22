# a test program for flex attention in PyTorch 
# This script benchmarks the flex attention implementation in PyTorch.
# See https://zhuanlan.zhihu.com/p/21829504838
# See https://pytorch.org/blog/flexattention/

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
)


def noop(score, b, h, q_idx, kv_idx):
    return score

flex_attention = torch.compile(flex_attention, dynamic=False)

torch._dynamo.config.cache_size_limit = 192
torch._dynamo.config.accumulated_cache_size_limit = 192

# Set the device to CUDA if available, otherwise use CPU   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create random tensors for query, key, and value
batch_size = 1
num_heads = 1
seq_len = 10
head_dim = 32  # Fixed key and value dimension

# Create tensors with proper dimensions
query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")
print(f"Head dimension: {head_dim}")

# Test flex attention
output = flex_attention(query, key, value, score_mod=noop)
loss = output.sum()
loss.backward()

print(f"Output shape: {output.shape}")
print(f"Test completed successfully!")