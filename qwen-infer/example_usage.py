#!/usr/bin/env python3
"""
Example usage of Qwen3 inference program.

This script demonstrates various ways to use the Qwen3Inference class.
"""

from qwen_infer import Qwen3Inference
import time

# This line sets the file path to the Qwen3 model directory.
# r means that this is a raw string, meaning that
# Backslashes (\) are treated as literal characters, not escape sequences
__lei_model_path__ = r"D:\models\qwen3-0.6B"

def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Initialize inference with a smaller model for faster loading
    inference = Qwen3Inference(
        model_name_or_path=__lei_model_path__,  # Small model for demo
        device="auto",
        torch_dtype="auto"
    )
    
    # Simple generation
    prompt = "Explain quantum computing in simple terms."
    print(f"Prompt: {prompt}")
    
    result = inference.generate(
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.7
    )
    
    print(f"Generated: {result}")
    print()


def example_streaming():
    """Streaming generation example."""
    print("=== Streaming Generation Example ===")
    
    inference = Qwen3Inference(
        model_name_or_path=__lei_model_path__,
        device="auto",
        torch_dtype="auto"
    )
    
    prompt = "Write a short story about a robot learning to paint."
    print(f"Prompt: {prompt}")
    print("Generating (streaming):")
    
    result = inference.generate(
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.8,
        stream=True
    )
    
    print(f"\nFinal result: {result}")
    print()


def example_batch_generation():
    """Batch generation example."""
    print("=== Batch Generation Example ===")
    
    inference = Qwen3Inference(
        model_name_or_path=__lei_model_path__,
        device="auto",
        torch_dtype="auto"
    )
    
    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "What is 2 + 2?",
        "Name three programming languages."
    ]
    
    print("Batch prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\nGenerating...")
    start_time = time.time()
    
    results = inference.batch_generate(
        prompts=prompts,
        max_new_tokens=100,
        temperature=0.5
    )
    
    end_time = time.time()
    
    print(f"\nBatch generation completed in {end_time - start_time:.2f} seconds")
    print("\nResults:")
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n{i}. Prompt: {prompt}")
        print(f"   Result: {result}")
    print()


def example_different_parameters():
    """Example with different generation parameters."""
    print("=== Different Parameters Example ===")
    
    inference = Qwen3Inference(
        model_name_or_path=__lei_model_path__,
        device="auto",
        torch_dtype="auto"
    )
    
    prompt = "Write a creative story about a magical forest."
    
    # High temperature (more creative)
    print("High temperature (0.9):")
    result1 = inference.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.9,
        top_p=0.95
    )
    print(f"Result: {result1}")
    print()
    
    # Low temperature (more focused)
    print("Low temperature (0.3):")
    result2 = inference.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.8
    )
    print(f"Result: {result2}")
    print()


def example_memory_optimization():
    """Example with memory optimization."""
    print("=== Memory Optimization Example ===")
    
    # This example shows how to use memory optimization for larger models
    # Note: This requires more GPU memory and may not work on all systems
    
    try:
        print("Attempting to load model with 8-bit quantization...")
        inference = Qwen3Inference(
            model_name_or_path=__lei_model_path__,
            device="auto",
            torch_dtype="auto",
            load_in_8bit=True,  # Use 8-bit quantization
            use_flash_attention=True
        )
        
        prompt = "Explain the benefits of renewable energy."
        print(f"Prompt: {prompt}")
        
        result = inference.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7
        )
        
        print(f"Generated: {result}")
        
    except ImportError as e:
        print(f"Quantization failed - bitsandbytes not available: {e}")
        print("To use quantization, install bitsandbytes: pip install -U bitsandbytes")
        print("Continuing without quantization...")
        
        # Try without quantization
        try:
            inference = Qwen3Inference(
                model_name_or_path=__lei_model_path__,
                device="auto",
                torch_dtype="auto",
                load_in_8bit=False,
                use_flash_attention=True
            )
            
            prompt = "Explain the benefits of renewable energy."
            print(f"Prompt: {prompt}")
            
            result = inference.generate(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7
            )
            
            print(f"Generated (without quantization): {result}")
            
        except Exception as e2:
            print(f"Model loading failed: {e2}")
            
    except Exception as e:
        print(f"Memory optimization example failed: {e}")
        print("This may be due to insufficient GPU memory or other system constraints.")
    print()


def example_custom_model_path():
    """Example with custom model path."""
    print("=== Custom Model Path Example ===")
    
    # This example shows how to use a local model
    # You would need to have downloaded the model locally first
    
    # Example path (you would need to adjust this to your actual model path)
    local_model_path = __lei_model_path__
    
    try:
        inference = Qwen3Inference(
            model_name_or_path=local_model_path,
            device="auto",
            torch_dtype="auto"
        )
        
        prompt = "What is machine learning?"
        print(f"Prompt: {prompt}")
        
        result = inference.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        print(f"Generated: {result}")
        
    except Exception as e:
        print(f"Custom model path example failed: {e}")
        print("This is expected if the local model path doesn't exist.")
    print()


def example_torch_compile():
    """Example with torch.compile for faster inference."""
    print("=== Torch Compile Example ===")
    
    # Test without compilation
    print("Testing without torch.compile...")
    inference_no_compile = Qwen3Inference(
        model_name_or_path=__lei_model_path__,
        device="auto",
        torch_dtype="auto",
        use_compile=False
    )
    
    prompt = "Explain the benefits of renewable energy in one paragraph."
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    result_no_compile = inference_no_compile.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7
    )
    end_time = time.time()
    time_without_compile = end_time - start_time
    
    print(f"Without compilation: {time_without_compile:.2f} seconds")
    print(f"Result: {result_no_compile}")
    print()
    
    # Test with compilation
    print("Testing with torch.compile...")
    inference_with_compile = Qwen3Inference(
        model_name_or_path=__lei_model_path__,
        device="auto",
        torch_dtype="auto",
        use_compile=True,
        compile_mode="default"
    )
    
    start_time = time.time()
    result_with_compile = inference_with_compile.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7
    )
    end_time = time.time()
    time_with_compile = end_time - start_time
    
    print(f"With compilation: {time_with_compile:.2f} seconds")
    print(f"Result: {result_with_compile}")
    print()
    
    # Calculate speedup
    speedup = time_without_compile / time_with_compile if time_with_compile > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    print()


def example_compile_modes():
    """Example showing different torch.compile modes."""
    print("=== Torch Compile Modes Example ===")
    
    modes = ["default", "reduce-overhead", "max-autotune"]
    
    for mode in modes:
        print(f"Testing mode: {mode}")
        try:
            inference = Qwen3Inference(
                model_name_or_path=__lei_model_path__,
                device="auto",
                torch_dtype="auto",
                use_compile=True,
                compile_mode=mode
            )
            
            prompt = "Write a short poem about technology."
            print(f"Prompt: {prompt}")
            
            start_time = time.time()
            result = inference.generate(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8
            )
            end_time = time.time()
            
            print(f"Mode '{mode}' completed in {end_time - start_time:.2f} seconds")
            print(f"Result: {result}")
            print()
            
        except Exception as e:
            print(f"Mode '{mode}' failed: {e}")
            print()


def main():
    """Run all examples."""
    print("Qwen3 Inference Examples")
    print("=" * 50)
    
    # Run examples
    #example_basic_usage()
    #example_streaming()
    example_batch_generation()
    #example_different_parameters()
    #example_memory_optimization()
    #example_custom_model_path()
    #example_torch_compile()
    #example_compile_modes()
    
    print("All examples completed!")

if __name__ == "__main__":
    main() 