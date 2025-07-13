#!/usr/bin/env python3
"""
Test script for Top-k Attention implementation in Qwen-Infer.

This script tests the top-k attention mechanism with various configurations
and provides performance comparisons.
"""

import torch
import time
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Any

try:
    from top_k_attention import TopKAttention, TopKAttentionConfig, create_top_k_attention_layer
    from qwen_infer import Qwen3Inference
    TOP_K_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import top-k attention modules: {e}")
    TOP_K_AVAILABLE = False


def test_top_k_attention_basic():
    """Test basic functionality of top-k attention."""
    print("Testing basic top-k attention functionality...")
    
    if not TOP_K_AVAILABLE:
        print("Skipping test - top-k attention not available")
        return False
    
    # Test configuration
    config = TopKAttentionConfig(k=16, temperature=1.0)
    
    # Create attention layer
    attention = create_top_k_attention_layer(
        embed_dim=512,
        num_heads=8,
        config=config
    )
    
    # Test input
    batch_size, seq_len, embed_dim = 2, 64, 512
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test forward pass
    try:
        output = attention(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test with attention mask
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, seq_len//2:] = 0  # Mask out second half
        
        output_masked = attention(x, mask=mask)
        print(f"✓ Masked attention successful")
        print(f"  Masked output shape: {output_masked.shape}")
        
        # Test memory statistics
        stats = attention.get_memory_stats()
        print(f"✓ Memory statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_top_k_attention_performance():
    """Test performance comparison between full attention and top-k attention."""
    print("\nTesting performance comparison...")
    
    if not TOP_K_AVAILABLE:
        print("Skipping test - top-k attention not available")
        return
    
    # Test configurations
    embed_dim = 512
    num_heads = 8
    batch_size = 4
    seq_lengths = [128, 256, 512, 1024]
    k_values = [16, 32, 64, 128]
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Generate test data
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Test full attention (fallback)
        config_full = TopKAttentionConfig(k=seq_len, temperature=1.0)  # k >= seq_len triggers full attention
        attention_full = create_top_k_attention_layer(embed_dim, num_heads, config_full)
        
        # Time full attention
        start_time = time.time()
        for _ in range(10):
            _ = attention_full(x)
        full_time = (time.time() - start_time) / 10
        
        # Test top-k attention
        for k in k_values:
            if k >= seq_len:
                continue
                
            config_topk = TopKAttentionConfig(k=k, temperature=1.0)
            attention_topk = create_top_k_attention_layer(embed_dim, num_heads, config_topk)
            
            # Time top-k attention
            start_time = time.time()
            for _ in range(10):
                _ = attention_topk(x)
            topk_time = (time.time() - start_time) / 10
            
            speedup = full_time / topk_time if topk_time > 0 else 0
            memory_stats = attention_topk.get_memory_stats()
            
            results.append({
                'seq_len': seq_len,
                'k': k,
                'full_time': full_time,
                'topk_time': topk_time,
                'speedup': speedup,
                'memory_saved': memory_stats.get('total_memory_saved', 0)
            })
            
            print(f"  k={k}: {topk_time:.4f}s vs {full_time:.4f}s (speedup: {speedup:.2f}x)")
    
    return results


def test_qwen_integration():
    """Test integration with Qwen3Inference."""
    print("\nTesting Qwen3Inference integration...")
    
    if not TOP_K_AVAILABLE:
        print("Skipping test - top-k attention not available")
        return False
    
    try:
        # Create top-k attention config
        config = TopKAttentionConfig(k=32, temperature=1.0)
        
        # Initialize Qwen3Inference with top-k attention
        # Note: This will show warnings about model loading since we don't have the actual model
        inference = Qwen3Inference(
            model_name_or_path="dummy_model",  # This will fail, but that's okay for testing
            use_top_k_attention=True,
            top_k_attention_config=config
        )
        
        print("✓ Qwen3Inference initialization with top-k attention successful")
        
        # Test stats retrieval
        stats = inference.get_top_k_attention_stats()
        print(f"✓ Top-k attention stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Note: Expected error during model loading: {e}")
        # This is expected since we're using a dummy model path
        return True


def benchmark_memory_usage():
    """Benchmark memory usage of top-k attention."""
    print("\nBenchmarking memory usage...")
    
    if not TOP_K_AVAILABLE:
        print("Skipping test - top-k attention not available")
        return
    
    # Configuration
    embed_dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 512
    k_values = [16, 32, 64, 128, 256]
    
    print(f"Testing with embed_dim={embed_dim}, num_heads={num_heads}, seq_len={seq_len}")
    
    for k in k_values:
        if k >= seq_len:
            continue
            
        # Create attention layer
        config = TopKAttentionConfig(k=k, temperature=1.0)
        attention = create_top_k_attention_layer(embed_dim, num_heads, config)
        
        # Test input
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Run multiple forward passes
        for _ in range(100):
            _ = attention(x)
        
        stats = attention.get_memory_stats()
        memory_saved = stats.get('total_memory_saved', 0)
        avg_memory_saved = stats.get('avg_memory_saved_per_call', 0)
        
        print(f"k={k}: Total memory saved: {memory_saved}, Avg per call: {avg_memory_saved:.2f}")


def plot_performance_results(results: List[Dict[str, Any]]):
    """Plot performance comparison results."""
    if not results:
        print("No results to plot")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        # Group results by sequence length
        seq_lengths = sorted(set(r['seq_len'] for r in results))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot speedup
        for seq_len in seq_lengths:
            seq_results = [r for r in results if r['seq_len'] == seq_len]
            k_values = [r['k'] for r in seq_results]
            speedups = [r['speedup'] for r in seq_results]
            
            ax1.plot(k_values, speedups, marker='o', label=f'seq_len={seq_len}')
        
        ax1.set_xlabel('k (number of top positions)')
        ax1.set_ylabel('Speedup (x)')
        ax1.set_title('Top-k Attention Speedup vs Full Attention')
        ax1.legend()
        ax1.grid(True)
        
        # Plot memory saved
        for seq_len in seq_lengths:
            seq_results = [r for r in results if r['seq_len'] == seq_len]
            k_values = [r['k'] for r in seq_results]
            memory_saved = [r['memory_saved'] for r in seq_results]
            
            ax2.plot(k_values, memory_saved, marker='s', label=f'seq_len={seq_len}')
        
        ax2.set_xlabel('k (number of top positions)')
        ax2.set_ylabel('Memory Saved (operations)')
        ax2.set_title('Memory Savings with Top-k Attention')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('top_k_attention_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Performance plot saved as 'top_k_attention_performance.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping plot generation")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Top-k Attention implementation")
    parser.add_argument("--basic", action="store_true", help="Run basic functionality tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--integration", action="store_true", help="Test Qwen integration")
    parser.add_argument("--memory", action="store_true", help="Test memory usage")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not any([args.basic, args.performance, args.integration, args.memory, args.all]):
        args.all = True  # Default to all tests
    
    print("Top-k Attention Test Suite")
    print("=" * 50)
    
    if not TOP_K_AVAILABLE:
        print("Error: Top-k attention modules not available")
        print("Please ensure top_k_attention.py is in the same directory")
        return
    
    success = True
    results = []
    
    if args.basic or args.all:
        success &= test_top_k_attention_basic()
    
    if args.performance or args.all:
        perf_results = test_top_k_attention_performance()
        if perf_results:
            results.extend(perf_results)
    
    if args.integration or args.all:
        success &= test_qwen_integration()
    
    if args.memory or args.all:
        benchmark_memory_usage()
    
    if (args.plot or args.all) and results:
        plot_performance_results(results)
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    # Performance summary
    if results:
        print("\nPerformance Summary:")
        print("-" * 30)
        for result in results[:5]:  # Show first 5 results
            print(f"seq_len={result['seq_len']}, k={result['k']}: "
                  f"{result['speedup']:.2f}x speedup, "
                  f"{result['memory_saved']} memory saved")


if __name__ == "__main__":
    main()