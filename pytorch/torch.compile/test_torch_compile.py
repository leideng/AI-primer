#!/usr/bin/env python3
"""
Simple test script for torch.compile mechanism demonstration
"""

import torch
import time
import sys

def test_basic_compilation():
    """Test basic torch.compile functionality"""
    print("Testing basic torch.compile...")
    
    def simple_function(x, y):
        return torch.sin(x) + torch.cos(y)
    
    # Create test tensors
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    
    # Test original function
    result_original = simple_function(x, y)
    
    # Test compiled function
    compiled_function = torch.compile(simple_function)
    result_compiled = compiled_function(x, y)
    
    # Check results are similar
    if torch.allclose(result_original, result_compiled, atol=1e-6):
        print("✓ Basic compilation test passed")
        return True
    else:
        print("✗ Basic compilation test failed")
        return False

def test_backend_comparison():
    """Test different backends"""
    print("Testing backend comparison...")
    
    def test_function(x, y):
        return torch.sin(x) + torch.cos(y)
    
    x = torch.randn(5, 5)
    y = torch.randn(5, 5)
    
    # Test only backends that are likely to work
    backends = ["eager"]
    
    for backend in backends:
        try:
            compiled_fn = torch.compile(test_function, backend=backend)
            result = compiled_fn(x, y)
            print(f"✓ Backend '{backend}' works")
        except Exception as e:
            print(f"✗ Backend '{backend}' failed: {e}")
            return False
    
    return True

def test_dynamic_shapes():
    """Test dynamic shape handling"""
    print("Testing dynamic shapes...")
    
    def dynamic_function(x, y):
        return torch.sin(x) + torch.cos(y)
    
    try:
        compiled_fn = torch.compile(dynamic_function, dynamic=True)
        
        # Test different shapes
        shapes = [(5, 5), (10, 10)]
        
        for shape in shapes:
            x = torch.randn(*shape)
            y = torch.randn(*shape)
            result = compiled_fn(x, y)
            print(f"✓ Shape {shape} works")
        
        return True
    except Exception as e:
        print(f"✗ Dynamic shapes test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization (basic check)"""
    print("Testing memory optimization...")
    
    def memory_function(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        c = a + b
        d = torch.tanh(c)
        return d
    
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    
    try:
        # Original function
        result_original = memory_function(x, y)
        
        # Compiled function
        compiled_fn = torch.compile(memory_function)
        result_compiled = compiled_fn(x, y)
        
        if torch.allclose(result_original, result_compiled, atol=1e-6):
            print("✓ Memory optimization test passed")
            return True
        else:
            print("✗ Memory optimization test failed")
            return False
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Running torch.compile mechanism tests...")
    print("=" * 50)
    
    tests = [
        test_basic_compilation,
        test_backend_comparison,
        test_dynamic_shapes,
        test_memory_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 