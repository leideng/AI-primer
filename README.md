# AI-primer
This repository records my own journey to learn AI. It includdes the following catergories.

## Transformer
### Self-Attention
- `transformer/self_attention_example.py`: Basic self-attention implementation with fixed 64-dimension key/value
- `transformer/self_attn.py`: BERT attention visualization
- `transformer/self_attn.ipynb`: Jupyter notebook for attention analysis

### Attention Implementations
- `transformer/flex_attention.py`: Custom multihead attention with performance benchmarking
- `transformer/flash_attention_comparison.py`: FlashAttention vs FlexAttention comparison
- `transformer/attention_benchmark_comparison.py`: **NEW** - Comprehensive comparison of Simple vs Flash vs Flex Attention
- `transformer/test_flash_vs_flex.py`: Test script for attention comparison
- `pytorch/flex_attn_test.py`: PyTorch flex attention test with 64-dimension key/value
- `pytorch/simple_flex_test.py`: **NEW** - Simple FlexAttention test (< 50 lines)

### Encoder
### Decoder
### KV Cache

## Usage Examples

### Quick FlexAttention Test (Simple)
```bash
cd pytorch
python simple_flex_test.py
```

### Comprehensive Attention Comparison
```bash
cd transformer
python attention_benchmark_comparison.py
```

### Compare FlashAttention vs FlexAttention
```bash
cd transformer
python test_flash_vs_flex.py
```

### Run Flex Attention Test
```bash
cd pytorch
python flex_attn_test.py
```

### Run Attention Benchmarking
```bash
cd transformer
python flex_attention.py
```

## Key Features

### Fixed 64-Dimension Key/Value
All attention implementations use a fixed 64-dimension for key and value tensors, providing:
- Consistent performance characteristics
- Easier comparison between implementations
- Optimized memory usage patterns

### Three Attention Implementations
1. **Simple Attention**: Baseline implementation for comparison
2. **FlashAttention**: Optimized attention using PyTorch's SDPA with FlashAttention backend
3. **FlexAttention**: PyTorch's flexible attention implementation with score modification

### Comprehensive Benchmarking
- Performance comparison across different sequence lengths
- Memory usage analysis
- Output correctness validation
- Speedup calculations relative to fastest implementation

