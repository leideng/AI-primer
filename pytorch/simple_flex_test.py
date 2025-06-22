#!/usr/bin/env python3
"""Simple FlexAttention Test - Inference Only (< 50 lines)"""

import torch
from torch.nn.attention.flex_attention import flex_attention

def noop(score, b, h, q_idx, kv_idx):
    return score

# Configure PyTorch
torch._dynamo.config.cache_size_limit = 192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create test tensors with 64-dimension key/value (no gradients needed for inference)
batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

print(f"Testing FlexAttention Inference on {device}")
print(f"Input shapes: Q{query.shape}, K{key.shape}, V{value.shape}")

try:
    # Compile and test FlexAttention for inference
    if device.type == 'cuda':
        flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    else:
        flex_attention_compiled = flex_attention
    
    with torch.no_grad():  # Inference mode
        output = flex_attention_compiled(query, key, value, score_mod=noop)
    
    print(f"✓ Success! Output shape: {output.shape}")
    print(f"✓ Output mean: {output.mean().item():.6f}")
    print(f"✓ Output std: {output.std().item():.6f}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    if device.type == 'cpu':
        print("💡 Try running on CUDA for better FlexAttention support") 