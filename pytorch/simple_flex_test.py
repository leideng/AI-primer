#!/usr/bin/env python3
"""Simple FlexAttention Test - Less than 50 lines"""

import torch
from torch.nn.attention.flex_attention import flex_attention

def noop(score, b, h, q_idx, kv_idx):
    return score

# Configure PyTorch
torch._dynamo.config.cache_size_limit = 192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create test tensors with 64-dimension key/value
batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

print(f"Testing FlexAttention on {device}")
print(f"Input shapes: Q{query.shape}, K{key.shape}, V{value.shape}")

try:
    # Compile and test FlexAttention
    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    output = flex_attention_compiled(query, key, value, score_mod=noop)
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Success! Output shape: {output.shape}")
    print(f"✓ Loss: {loss.item():.6f}")
    print(f"✓ Gradients computed: {query.grad is not None}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    if device.type == 'cpu':
        print("💡 Try running on CUDA for better FlexAttention support") 