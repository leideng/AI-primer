# a test program for flex attention in PyTorch 
# This script benchmarks the flex attention implementation in PyTorch.
# See https://zhuanlan.zhihu.com/p/21829504838
# See https://pytorch.org/blog/flexattention/

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
)

flex_attention = torch.compile(flex_attention, dynamic=False)

torch._dynamo.config.cache_size_limit = 192
torch._dynamo.config.accumulated_cache_size_limit = 192