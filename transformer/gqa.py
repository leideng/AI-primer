import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads # Dimension of each attention head
        self.group_size = self.num_heads // self.num_kv_heads # How many query heads share one KV head

        # Linear projections for Query, Key, and Value
        # Q projection: maps embed_dim to num_heads * head_dim, we will later 
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        # K projection: maps embed_dim to num_kv_heads * head_dim
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        # V projection: maps embed_dim to num_kv_heads * head_dim
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        # Output projection: maps embed_dim back to embed_dim
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        # query, key, value typically have shape (batch_size, sequence_length, embed_dim)
        batch_size, seq_len, _ = query.size()

        # 1. Project Q, K, V
        # Apply linear projections and reshape to separate heads
        # Q: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # K: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_kv_heads, head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # V: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_kv_heads, head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 2. Transpose for attention calculation
        # Change shape to (batch_size, num_heads/num_kv_heads, seq_len, head_dim)
        # This puts the head dimension before the sequence length for batch matrix multiplication
        q = q.transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2) # (batch_size, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2) # (batch_size, num_kv_heads, seq_len, head_dim)

        # 3. Repeat K and V heads to match the number of query heads
        # This is the core of GQA: each KV head is shared by `group_size` query heads.
        # We effectively duplicate the KV heads so that each query head has a corresponding KV head to attend to.
        # k: (batch_size, num_kv_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        k = k.repeat_interleave(self.group_size, dim=1)
        # v: (batch_size, num_kv_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 4. Scaled Dot-Product Attention
        # Calculate attention scores: Q @ K_T / sqrt(head_dim)
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask if provided (e.g., for causal masking in LLMs)
        if attn_mask is not None:
            # Mask out positions that should not be attended to (set to -inf)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # 5. Compute weighted sum of values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)

        # 6. Concatenate heads and apply final linear projection
        # Transpose back to (batch_size, seq_len, num_heads, head_dim)
        # Then flatten the last two dimensions to (batch_size, seq_len, num_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Apply the final output projection layer
        # (batch_size, seq_len, num_heads * head_dim) -> (batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output

# --- End-to-End Usage Example ---

# Define parameters for the attention module
embed_dim = 768  # Dimension of the input hidden state
num_heads = 12   # Total number of query heads
num_kv_heads = 3 # Number of Key/Value head groups (must divide num_heads)
                 # Here, 12 / 3 = 4. So, 4 query heads share one KV head.
                 # If num_kv_heads = 12, it's MHA.
                 # If num_kv_heads = 1, it's MQA.

# Instantiate the GQA module
gqa_layer = GroupedQueryAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    dropout=0.1
)

# Create a dummy input hidden state (e.g., from a previous layer or embedding)
batch_size = 2
sequence_length = 50
# x represents the hidden state for each token in the sequence for each item in the batch
x = torch.randn(batch_size, sequence_length, embed_dim)

print(f"Input hidden state shape (x): {x.shape}")

# Perform self-attention using the GQA layer
# In self-attention, query, key, and value are all the same input 'x'
y = gqa_layer(query=x, key=x, value=x)

print(f"Output hidden state shape (y) after GQA self-attention: {y.shape}")

# You can also test with an attention mask (e.g., a causal mask for language models)
# A causal mask ensures that a token can only attend to previous tokens.
# For a sequence of length L, the mask is an L x L lower triangular matrix of ones.
attn_mask = torch.tril(torch.ones(sequence_length, sequence_length)).bool()
# Expand mask to match attention score dimensions (batch_size, num_heads, seq_len, seq_len)
# The mask needs to be broadcastable to the attention scores.
# (1, 1, seq_len, seq_len) is a common way to make it broadcastable.
attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

print(f"\nTesting with a causal attention mask of shape: {attn_mask.shape}")
y_masked = gqa_layer(query=x, key=x, value=x, attn_mask=attn_mask)
print(f"Output hidden state shape (y_masked) with mask: {y_masked.shape}")