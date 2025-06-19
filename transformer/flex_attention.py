# Performance comparison between different attention implementations
# See https://pytorch.org/blog/flexattention/ for reference
import torch
import torch.nn as nn
import time
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel

class CustomMultiheadAttention(nn.Module):
    """Custom Multihead Attention implementation for comparison"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)

def benchmark_attention(attention_module, x, num_runs=100, warmup_runs=10, is_torch_attn=False):
    """Benchmark attention module performance"""
    device = x.device
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            if is_torch_attn:
                _ = attention_module(x, x, x)[0]  # PyTorch MultiheadAttention expects (query, key, value)
            else:
                _ = attention_module(x)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            if is_torch_attn:
                output = attention_module(x, x, x)[0]  # PyTorch MultiheadAttention expects (query, key, value)
            else:
                output = attention_module(x)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time, output

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 4
    seq_len = 512
    embed_dim = 512
    num_heads = 8
    num_runs = 100
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Number of benchmark runs: {num_runs}")
    print()
    
    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Create attention modules
    custom_attn = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)
    torch_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).to(device)
    
    # Test with different SDP backends for PyTorch MultiheadAttention
    backends = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    backend_names = ["MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"]
    
    print("Benchmarking Custom Multihead Attention...")
    custom_time, custom_output = benchmark_attention(custom_attn, x, num_runs)
    print(f"Custom Multihead Attention average time: {custom_time*1000:.3f} ms")
    print(f"Custom Multihead Attention output shape: {custom_output.shape}")
    print()
    
    # Benchmark PyTorch MultiheadAttention with different backends
    for backend, backend_name in zip(backends, backend_names):
        print(f"Benchmarking PyTorch MultiheadAttention with {backend_name} backend...")
        try:
            with sdpa_kernel(backends=[backend]):
                torch_time, torch_output = benchmark_attention(torch_attn, x, num_runs, is_torch_attn=True)
                print(f"PyTorch MultiheadAttention ({backend_name}) average time: {torch_time*1000:.3f} ms")
                print(f"PyTorch MultiheadAttention output shape: {torch_output.shape}")
                
                # Check if outputs are similar (they should be close but not identical due to different implementations)
                with torch.no_grad():
                    diff = torch.abs(custom_output - torch_output).mean().item()
                    print(f"Output difference (mean abs): {diff:.6f}")
                
                # Calculate speedup
                speedup = torch_time / custom_time
                print(f"Custom Attention is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than {backend_name}")
                print()
                
        except Exception as e:
            print(f"Failed to benchmark {backend_name}: {e}")
            print()
    
    # Memory usage comparison
    print("Memory usage comparison:")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Custom attention memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    custom_output = custom_attn(x)
    if device.type == 'cuda':
        custom_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Custom Multihead Attention peak memory: {custom_memory:.2f} MB")
    
    # PyTorch attention memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch_output = torch_attn(x, x, x)[0]  # PyTorch MultiheadAttention expects (query, key, value)
    if device.type == 'cuda':
        torch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"PyTorch MultiheadAttention peak memory: {torch_memory:.2f} MB")
        print(f"Memory difference: {abs(custom_memory - torch_memory):.2f} MB")
    
    # Additional comparison: Test with different sequence lengths
    print("\n" + "="*50)
    print("Performance comparison with different sequence lengths:")
    print("="*50)
    
    seq_lengths = [128, 256, 512, 1024]
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        x_test = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Custom attention
        custom_time, _ = benchmark_attention(custom_attn, x_test, num_runs=50)
        print(f"  Custom Attention: {custom_time*1000:.3f} ms")
        
        # PyTorch attention with best available backend
        try:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                torch_time, _ = benchmark_attention(torch_attn, x_test, num_runs=50, is_torch_attn=True)
                print(f"  PyTorch Attention: {torch_time*1000:.3f} ms")
                speedup = torch_time / custom_time
                print(f"  Speedup: {speedup:.2f}x")
        except:
            try:
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    torch_time, _ = benchmark_attention(torch_attn, x_test, num_runs=50, is_torch_attn=True)
                    print(f"  PyTorch Attention: {torch_time*1000:.3f} ms")
                    speedup = torch_time / custom_time
                    print(f"  Speedup: {speedup:.2f}x")
            except:
                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    torch_time, _ = benchmark_attention(torch_attn, x_test, num_runs=50, is_torch_attn=True)
                    print(f"  PyTorch Attention: {torch_time*1000:.3f} ms")
                    speedup = torch_time / custom_time
                    print(f"  Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()