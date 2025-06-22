#!/usr/bin/env python3
"""
Simple test script to compare FlashAttention vs FlexAttention
Run this script to see the performance and correctness comparison.
"""

import sys
import os

# Add the current directory to the path so we can import the comparison module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flash_attention_comparison import test_attention_correctness, compare_attention_implementations

def main():
    print("=" * 80)
    print("FlashAttention vs FlexAttention Comparison")
    print("=" * 80)
    print()
    
    # Test correctness first
    test_attention_correctness()
    print("\n" + "=" * 80)
    
    # Run performance comparison
    compare_attention_implementations()
    
    print("\n" + "=" * 80)
    print("Comparison completed!")
    print("=" * 80)

if __name__ == "__main__":
    main() 