# Qwen3 Inference Program

A comprehensive Python program for running inference with Qwen3 models, supporting both local and Hugging Face models with various optimization options.

## Features

- **Multiple Model Sources**: Support for Hugging Face models and local model paths
- **Memory Optimization**: 8-bit and 4-bit quantization support
- **Device Flexibility**: Automatic device detection (CPU, CUDA, MPS)
- **Streaming Generation**: Real-time text generation with streaming output
- **Batch Processing**: Efficient batch inference for multiple prompts
- **Flash Attention**: Optional flash attention for improved performance
- **Torch Compile**: Optional torch.compile for faster inference
- **Flexible Parameters**: Comprehensive generation parameter control
- **Command Line Interface**: Easy-to-use CLI for quick inference

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install torch transformers accelerate
```

For quantization support:
```bash
pip install -U bitsandbytes
```

For additional optimizations:
```bash
pip install flash-attn  # For flash attention
```

## Quick Start

### Basic Usage

```python
from qwen_infer import Qwen3Inference

# Initialize inference
inference = Qwen3Inference(
    model_name_or_path="Qwen/Qwen3-8B",
    device="auto"
)

# Generate text
result = inference.generate(
    prompt="Explain quantum computing in simple terms.",
    max_new_tokens=200,
    temperature=0.7
)
print(result)

# With torch.compile for faster inference
inference_fast = Qwen3Inference(
    model_name_or_path="Qwen/Qwen3-8B",
    device="auto",
    use_compile=True,
    compile_mode="default"
)
result = inference_fast.generate("Write a story about AI.")
print(result)
```

### Command Line Usage

```bash
# Basic inference
python qwen_infer.py --prompt "Hello, how are you?"

# With custom parameters
python qwen_infer.py \
    --model "Qwen/Qwen3-8B" \
    --prompt "Write a short story about a robot." \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --stream

# Batch mode
python qwen_infer.py --batch

# With torch.compile for faster inference
python qwen_infer.py --use_compile --compile_mode default --prompt "Hello, how are you?"
```

## Usage Examples

### 1. Basic Inference

```python
from qwen_infer import Qwen3Inference

inference = Qwen3Inference("Qwen/Qwen3-8B")
result = inference.generate("What is machine learning?")
print(result)
```

### 2. Streaming Generation

```python
inference = Qwen3Inference("Qwen/Qwen3-8B")
result = inference.generate(
    prompt="Write a story about space exploration.",
    stream=True,
    max_new_tokens=200
)
```

### 3. Batch Processing

```python
prompts = [
    "What is the capital of France?",
    "Explain photosynthesis.",
    "What is 2 + 2?"
]

results = inference.batch_generate(prompts)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

### 4. Memory Optimization

```python
# For large models with limited GPU memory
inference = Qwen3Inference(
    model_name_or_path="Qwen/Qwen3-235B-A22B",
    load_in_8bit=True,  # 8-bit quantization (requires bitsandbytes)
    device="auto"
)

# 4-bit quantization for even more memory savings
inference = Qwen3Inference(
    model_name_or_path="Qwen/Qwen3-235B-A22B",
    load_in_4bit=True,  # 4-bit quantization (requires bitsandbytes)
    device="auto"
)
```

### 5. Custom Model Path

```python
# Use a locally downloaded model
inference = Qwen3Inference("./models/Qwen3-8B")
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name or path | `Qwen/Qwen3-8B` |
| `--prompt` | Input prompt | `"Hello, how are you?"` |
| `--max_new_tokens` | Maximum tokens to generate | `512` |
| `--temperature` | Sampling temperature | `0.7` |
| `--top_p` | Top-p sampling parameter | `0.9` |
| `--top_k` | Top-k sampling parameter | `50` |
| `--repetition_penalty` | Repetition penalty | `1.1` |
| `--device` | Device to use | `auto` |
| `--torch_dtype` | Torch dtype | `auto` |
| `--stream` | Enable streaming | `False` |
| `--batch` | Enable batch mode | `False` |
| `--load_in_8bit` | Load model in 8-bit | `False` |
| `--load_in_4bit` | Load model in 4-bit | `False` |
| `--no_flash_attention` | Disable flash attention | `False` |
| `--use_compile` | Enable torch.compile for faster inference | `False` |
| `--compile_mode` | torch.compile mode (default, reduce-overhead, max-autotune) | `default` |

## Model Options

### Available Models

- `Qwen/Qwen3-0.6B` - Smallest model (0.6B parameters)
- `Qwen/Qwen3-1.7B` - Small model (1.7B parameters)
- `Qwen/Qwen3-8B` - Medium model (8B parameters)
- `Qwen/Qwen3-14B` - Large model (14B parameters)
- `Qwen/Qwen3-32B` - Huge model (32B parameters)
- `Qwen/Qwen3-30B-A3B` - Advanced model (30B parameters, A3B architecture)
- `Qwen/Qwen3-235B-A22B` - Massive model (235B parameters, A22B architecture)

### Model Selection Guide

- **Development/Testing**: Use 0.6B or 1.7B models
- **Production (Limited GPU)**: Use 8B or 14B models with quantization
- **Production (High-end GPU)**: Use 32B or 30B-A3B models
- **Research/Enterprise**: Use 235B-A22B model with multiple GPUs

## Performance Optimization

### GPU Memory Requirements

| Model Size | FP16 | 8-bit | 4-bit |
|------------|------|-------|-------|
| 0.6B | 1.2GB | 0.6GB | 0.3GB |
| 1.7B | 3.4GB | 1.7GB | 0.85GB |
| 8B | 16GB | 8GB | 4GB |
| 14B | 28GB | 14GB | 7GB |
| 32B | 64GB | 32GB | 16GB |
| 30B-A3B | 60GB | 30GB | 15GB |
| 235B-A22B | 470GB | 235GB | 117.5GB |

### Optimization Tips

1. **Use quantization** for large models (requires bitsandbytes):
   ```python
   load_in_8bit=True  # or load_in_4bit=True
   ```
   
   Note: Quantization requires the `bitsandbytes` library. Install with:
   ```bash
   pip install -U bitsandbytes
   ```

2. **Enable flash attention** for better performance:
   ```python
   use_flash_attention=True
   ```

3. **Use appropriate dtype**:
   ```python
   torch_dtype="float16"  # for GPU
   torch_dtype="float32"  # for CPU
   ```

4. **Set memory limits** for multi-GPU setups:
   ```python
   max_memory={"0": "40GB", "1": "40GB"}
   ```

5. **Use torch.compile** for faster inference:
   ```python
   use_compile=True,
   compile_mode="default"  # or "reduce-overhead", "max-autotune"
   ```

## Examples

### Run the Example Script

```bash
python example_usage.py
```

This will demonstrate:
- Basic usage
- Streaming generation
- Batch processing
- Different generation parameters
- Memory optimization
- Custom model paths

### Interactive Mode

```bash
# Start interactive session
python qwen_infer.py --batch

# Enter prompts one by one
What is artificial intelligence?
Explain quantum computing.
Write a poem about nature.
# Press Ctrl+D when done
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Use smaller models
   - Enable quantization (`--load_in_8bit` or `--load_in_4bit`)
   - Reduce `max_new_tokens`

2. **Quantization Errors**
   - Install bitsandbytes: `pip install -U bitsandbytes`
   - Check CUDA compatibility: `python -c "import bitsandbytes"`
   - Use without quantization if issues persist

3. **Slow Generation**
   - Use GPU if available
   - Enable flash attention
   - Use appropriate dtype

4. **Model Loading Errors**
   - Check internet connection for Hugging Face models
   - Verify local model path exists
   - Ensure sufficient disk space

5. **CUDA Errors**
   - Update CUDA drivers
   - Check PyTorch CUDA compatibility
   - Use CPU fallback: `--device cpu`

### Performance Monitoring

```python
import time

start_time = time.time()
result = inference.generate(prompt)
end_time = time.time()

print(f"Generation time: {end_time - start_time:.2f} seconds")
print(f"Tokens per second: {len(result.split()) / (end_time - start_time):.2f}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 