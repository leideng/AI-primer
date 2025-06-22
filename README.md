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
- `transformer/test_flash_vs_flex.py`: Test script for attention comparison
- `pytorch/flex_attn_test.py`: PyTorch flex attention test with 64-dimension key/value

### Encoder
### Decoder
### KV Cache

## Usage Examples

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

