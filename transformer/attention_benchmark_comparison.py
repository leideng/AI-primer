#!/usr/bin/env python3
"""
Attention Implementation Comparison: Simple vs Flash vs Flex
Benchmarks three attention implementations with consistent 64-dimension key/value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

def noop(score, b, h, q_idx, kv_idx):
    return score

class SimpleAttention(nn.Module):
    """Simple attention implementation for baseline comparison"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = 64  # Fixed to 64
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)

class FlashAttention(nn.Module):
    """FlashAttention implementation using PyTorch's SDPA"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = 64  # Fixed to 64
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch's scaled_dot_product_attention with FlashAttention backend
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)

class FlexAttentionWrapper(nn.Module):
    """Wrapper for FlexAttention to match other implementations"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = 64  # Fixed to 64
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
        # Compile flex_attention for better performance
        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply flex attention
        attn_output = self.flex_attention(q, k, v, score_mod=noop)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)

def benchmark_attention(attention_module, x, num_runs=100, warmup_runs=10):
    """Benchmark attention module performance"""
    device = x.device
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = attention_module(x)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = attention_module(x)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time, output

def compare_attention_implementations():
    """Compare all three attention implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Attention Implementation Comparison on {device}")
    print("=" * 70)
    
    # Test parameters
    batch_size = 4
    seq_len = 512
    embed_dim = 512  # Must be divisible by num_heads * head_dim (8 * 64 = 512)
    num_heads = 8
    num_runs = 100
    
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}")
    print(f"Head dimension: 64 (fixed)")
    print()
    
    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Create attention modules
    simple_attn = SimpleAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
    flash_attn = FlashAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
    flex_attn = FlexAttentionWrapper(embed_dim=embed_dim, num_heads=num_heads).to(device)
    
    implementations = [
        ("Simple Attention", simple_attn),
        ("FlashAttention", flash_attn),
        ("FlexAttention", flex_attn)
    ]
    
    results = {}
    
    # Benchmark each implementation
    for name, module in implementations:
        print(f"Benchmarking {name}...")
        try:
            time_taken, output = benchmark_attention(module, x, num_runs)
            results[name] = {
                'time': time_taken,
                'output': output,
                'success': True
            }
            print(f"  ✓ {name}: {time_taken*1000:.3f} ms")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
            results[name] = {
                'time': None,
                'output': None,
                'success': False,
                'error': str(e)
            }
    
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    
    # Find the fastest successful implementation
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) > 1:
        fastest_time = min(v['time'] for v in successful_results.values())
        
        print(f"{'Implementation':<20} {'Time (ms)':<12} {'Speedup':<12} {'Memory (MB)':<12}")
        print("-" * 60)
        
        for name, result in results.items():
            if result['success']:
                time_ms = result['time'] * 1000
                speedup = result['time'] / fastest_time
                memory_mb = 0
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    _ = result['output']
                    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                
                print(f"{name:<20} {time_ms:<12.3f} {speedup:<12.2f}x {memory_mb:<12.1f}")
            else:
                print(f"{name:<20} {'Failed':<12} {'N/A':<12} {'N/A':<12}")
    
    # Output comparison
    print("\n" + "=" * 70)
    print("Output Comparison")
    print("=" * 70)
    
    successful_outputs = [v['output'] for v in results.values() if v['success']]
    
    if len(successful_outputs) > 1:
        # Compare outputs
        for i, (name1, result1) in enumerate(results.items()):
            if not result1['success']:
                continue
            for name2, result2 in list(results.items())[i+1:]:
                if not result2['success']:
                    continue
                
                with torch.no_grad():
                    diff = torch.abs(result1['output'] - result2['output']).mean().item()
                    max_diff = torch.abs(result1['output'] - result2['output']).max().item()
                    
                    print(f"{name1} vs {name2}:")
                    print(f"  Mean absolute difference: {diff:.6f}")
                    print(f"  Max absolute difference: {max_diff:.6f}")
                    
                    # Check if outputs are close
                    tolerance = 1e-3
                    is_close = torch.allclose(result1['output'], result2['output'], atol=tolerance)
                    print(f"  Outputs are {'close' if is_close else 'different'} (tolerance: {tolerance})")
                    print()
    
    # Test with different sequence lengths
    print("=" * 70)
    print("Performance across sequence lengths")
    print("=" * 70)
    
    seq_lengths = [128, 256, 512, 1024]
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        x_test = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        for name, module in implementations:
            try:
                time_taken, _ = benchmark_attention(module, x_test, num_runs=50)
                print(f"  {name}: {time_taken*1000:.3f} ms")
            except:
                print(f"  {name}: Failed")

if __name__ == "__main__":
    compare_attention_implementations() 