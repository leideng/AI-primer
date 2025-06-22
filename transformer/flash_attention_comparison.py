# FlashAttention vs FlexAttention comparison
# See https://arxiv.org/abs/2205.14135 for FlashAttention paper
# See https://pytorch.org/blog/flexattention/ for FlexAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

class FlashAttention(nn.Module):
    """Standard FlashAttention implementation using PyTorch's SDPA"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = 64  # Fixed head dimension to 64
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
    """Wrapper for FlexAttention to match FlashAttention interface"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = 64  # Fixed head dimension to 64
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
        
        # Define score modification function (no modification for standard attention)
        def noop(score, b, h, q_idx, kv_idx):
            return score
        
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
    """Compare FlashAttention vs FlexAttention performance and outputs"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 4
    seq_len = 512
    embed_dim = 512  # Must be divisible by num_heads * head_dim (8 * 64 = 512)
    num_heads = 8
    num_runs = 100
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {embed_dim // num_heads} (fixed to 64)")
    print(f"  Number of benchmark runs: {num_runs}")
    print()
    
    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Create attention modules
    flash_attn = FlashAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
    flex_attn = FlexAttentionWrapper(embed_dim=embed_dim, num_heads=num_heads).to(device)
    
    # Test FlashAttention
    print("Benchmarking FlashAttention...")
    try:
        flash_time, flash_output = benchmark_attention(flash_attn, x, num_runs)
        print(f"FlashAttention average time: {flash_time*1000:.3f} ms")
        print(f"FlashAttention output shape: {flash_output.shape}")
        flash_success = True
    except Exception as e:
        print(f"FlashAttention failed: {e}")
        flash_success = False
        flash_time = None
        flash_output = None
    
    print()
    
    # Test FlexAttention
    print("Benchmarking FlexAttention...")
    try:
        flex_time, flex_output = benchmark_attention(flex_attn, x, num_runs)
        print(f"FlexAttention average time: {flex_time*1000:.3f} ms")
        print(f"FlexAttention output shape: {flex_output.shape}")
        flex_success = True
    except Exception as e:
        print(f"FlexAttention failed: {e}")
        flex_success = False
        flex_time = None
        flex_output = None
    
    print()
    
    # Compare outputs if both succeeded
    if flash_success and flex_success:
        print("Output comparison:")
        with torch.no_grad():
            diff = torch.abs(flash_output - flex_output).mean().item()
            max_diff = torch.abs(flash_output - flex_output).max().item()
            print(f"  Mean absolute difference: {diff:.6f}")
            print(f"  Max absolute difference: {max_diff:.6f}")
            
            # Check if outputs are close (within tolerance)
            tolerance = 1e-3
            is_close = torch.allclose(flash_output, flex_output, atol=tolerance)
            print(f"  Outputs are {'close' if is_close else 'different'} (tolerance: {tolerance})")
    
    print()
    
    # Performance comparison
    if flash_success and flex_success:
        print("Performance comparison:")
        if flash_time < flex_time:
            speedup = flex_time / flash_time
            print(f"  FlashAttention is {speedup:.2f}x faster than FlexAttention")
        else:
            speedup = flash_time / flex_time
            print(f"  FlexAttention is {speedup:.2f}x faster than FlashAttention")
    
    # Memory usage comparison
    print("\nMemory usage comparison:")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
        # FlashAttention memory
        torch.cuda.reset_peak_memory_stats()
        flash_output = flash_attn(x)
        flash_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"  FlashAttention peak memory: {flash_memory:.2f} MB")
        
        # FlexAttention memory
        torch.cuda.reset_peak_memory_stats()
        flex_output = flex_attn(x)
        flex_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"  FlexAttention peak memory: {flex_memory:.2f} MB")
        print(f"  Memory difference: {abs(flash_memory - flex_memory):.2f} MB")
    
    # Test with different sequence lengths
    print("\n" + "="*60)
    print("Performance comparison with different sequence lengths:")
    print("="*60)
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        x_test = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # FlashAttention
        try:
            flash_time, _ = benchmark_attention(flash_attn, x_test, num_runs=50)
            print(f"  FlashAttention: {flash_time*1000:.3f} ms")
        except:
            print(f"  FlashAttention: Failed")
            flash_time = None
        
        # FlexAttention
        try:
            flex_time, _ = benchmark_attention(flex_attn, x_test, num_runs=50)
            print(f"  FlexAttention: {flex_time*1000:.3f} ms")
        except:
            print(f"  FlexAttention: Failed")
            flex_time = None
        
        # Speedup comparison
        if flash_time is not None and flex_time is not None:
            if flash_time < flex_time:
                speedup = flex_time / flash_time
                print(f"  FlashAttention is {speedup:.2f}x faster")
            else:
                speedup = flash_time / flex_time
                print(f"  FlexAttention is {speedup:.2f}x faster")

def test_attention_correctness():
    """Test the correctness of both attention implementations"""
    print("Testing attention correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small test case for easier verification
    batch_size = 2
    seq_len = 8
    embed_dim = 64  # Must be divisible by num_heads * head_dim (4 * 64 = 256, but we'll use 64 for simplicity)
    num_heads = 1  # Use 1 head to make embed_dim compatible with head_dim=64
    
    # Create simple test input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Create attention modules
    flash_attn = FlashAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
    flex_attn = FlexAttentionWrapper(embed_dim=embed_dim, num_heads=num_heads).to(device)
    
    # Test outputs
    with torch.no_grad():
        flash_output = flash_attn(x)
        flex_output = flex_attn(x)
        
        print(f"FlashAttention output shape: {flash_output.shape}")
        print(f"FlexAttention output shape: {flex_output.shape}")
        
        # Check if shapes match
        if flash_output.shape == flex_output.shape:
            print("✓ Output shapes match")
        else:
            print("✗ Output shapes don't match")
        
        # Check if outputs are reasonable (not NaN or inf)
        if torch.isfinite(flash_output).all():
            print("✓ FlashAttention outputs are finite")
        else:
            print("✗ FlashAttention outputs contain NaN or inf")
            
        if torch.isfinite(flex_output).all():
            print("✓ FlexAttention outputs are finite")
        else:
            print("✗ FlexAttention outputs contain NaN or inf")

if __name__ == "__main__":
    # Test correctness first
    test_attention_correctness()
    print("\n" + "="*60)
    
    # Run performance comparison
    compare_attention_implementations() 