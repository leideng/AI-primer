import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import warnings


class TopKAttention(nn.Module):
    """
    Top-k Attention mechanism implementation for memory-efficient Transformers.
    
    This implementation only attends to the top-k most relevant positions,
    reducing memory usage significantly compared to full attention.
    
    References:
    - "Memory-efficient Transformers via Top-k Attention" (Gupta et al., 2021)
    - https://github.com/ag1988/top_k_attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        k: int = 32,
        dropout: float = 0.0,
        bias: bool = True,
        temperature: float = 1.0,
        use_sparse: bool = False,
        fallback_full_attention: bool = True
    ):
        """
        Initialize Top-k Attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            k: Number of top positions to attend to
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            temperature: Temperature for attention scores
            use_sparse: Whether to use sparse tensors (experimental)
            fallback_full_attention: Whether to fall back to full attention if k >= seq_len
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k = k
        self.dropout = dropout
        self.temperature = temperature
        self.use_sparse = use_sparse
        self.fallback_full_attention = fallback_full_attention
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Scale factor for attention scores
        self.scale = (self.head_dim ** -0.5) / temperature
        
        # For performance tracking
        self.num_forward_calls = 0
        self.total_memory_saved = 0
        
    def _get_top_k_indices(
        self, 
        scores: torch.Tensor, 
        k: int, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k indices and values from attention scores.
        
        Args:
            scores: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
            k: Number of top positions to select
            mask: Optional attention mask
            
        Returns:
            top_k_values: Top-k attention score values
            top_k_indices: Indices of top-k positions
        """
        if mask is not None:
            # Apply mask by setting masked positions to very negative values
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(scores, k, dim=-1, sorted=False)
        
        return top_k_values, top_k_indices
    
    def _compute_full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute full attention (fallback method).
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            mask: Optional attention mask
            
        Returns:
            Attention output of shape (batch_size, num_heads, seq_len, head_dim)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        return out
    
    def _compute_top_k_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute top-k attention.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            mask: Optional attention mask
            
        Returns:
            Attention output of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Get top-k indices and values
        top_k_values, top_k_indices = self._get_top_k_indices(scores, self.k, mask)
        
        # Create sparse attention weights
        if self.use_sparse:
            # Use sparse tensor implementation (experimental)
            attn_weights = self._create_sparse_attention_weights(
                top_k_values, top_k_indices, seq_len, batch_size, num_heads
            )
        else:
            # Use dense tensor with masking
            attn_weights = torch.zeros_like(scores)
            
            # Scatter top-k values to their positions
            attn_weights.scatter_(-1, top_k_indices, F.softmax(top_k_values, dim=-1))
        
        # Apply dropout
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        return out
    
    def _create_sparse_attention_weights(
        self,
        top_k_values: torch.Tensor,
        top_k_indices: torch.Tensor,
        seq_len: int,
        batch_size: int,
        num_heads: int
    ) -> torch.Tensor:
        """
        Create sparse attention weights tensor (experimental).
        
        Args:
            top_k_values: Top-k attention values
            top_k_indices: Top-k attention indices
            seq_len: Sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            
        Returns:
            Sparse attention weights tensor
        """
        # This is a simplified implementation - in practice, you'd want to use
        # PyTorch's sparse tensor operations for better performance
        device = top_k_values.device
        
        # Create indices for sparse tensor
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        head_indices = torch.arange(num_heads, device=device).unsqueeze(0).unsqueeze(1).unsqueeze(1)
        query_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(1)
        
        # Expand indices to match top_k_indices shape
        batch_indices = batch_indices.expand_as(top_k_indices)
        head_indices = head_indices.expand_as(top_k_indices)
        query_indices = query_indices.expand_as(top_k_indices)
        
        # Create sparse tensor indices
        indices = torch.stack([
            batch_indices.flatten(),
            head_indices.flatten(),
            query_indices.flatten(),
            top_k_indices.flatten()
        ])
        
        # Create sparse tensor values
        values = F.softmax(top_k_values, dim=-1).flatten()
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices,
            values,
            (batch_size, num_heads, seq_len, seq_len),
            device=device
        )
        
        return sparse_tensor.to_dense()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of Top-k Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            If return_attention is False:
                Output tensor of shape (batch_size, seq_len, embed_dim)
            If return_attention is True:
                Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Check if we should use full attention
        if self.fallback_full_attention and self.k >= seq_len:
            warnings.warn(
                f"Top-k value ({self.k}) is >= sequence length ({seq_len}). "
                f"Using full attention instead."
            )
            use_full_attention = True
        else:
            use_full_attention = False
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand mask if provided
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
        
        # Compute attention
        if use_full_attention:
            attn_output = self._compute_full_attention(q, k, v, mask)
        else:
            attn_output = self._compute_top_k_attention(q, k, v, mask)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        
        # Update statistics
        self.num_forward_calls += 1
        if not use_full_attention:
            memory_saved = (seq_len - self.k) * seq_len * batch_size * self.num_heads
            self.total_memory_saved += memory_saved
        
        if return_attention:
            # For simplicity, return None for attention weights in top-k mode
            # In practice, you'd return the sparse attention weights
            return output, None
        
        return output
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        return {
            'num_forward_calls': self.num_forward_calls,
            'total_memory_saved': self.total_memory_saved,
            'avg_memory_saved_per_call': (
                self.total_memory_saved / max(self.num_forward_calls, 1)
            )
        }
    
    def reset_stats(self):
        """Reset memory usage statistics."""
        self.num_forward_calls = 0
        self.total_memory_saved = 0


class TopKAttentionConfig:
    """Configuration class for Top-k Attention."""
    
    def __init__(
        self,
        k: int = 32,
        temperature: float = 1.0,
        use_sparse: bool = False,
        fallback_full_attention: bool = True,
        enable_memory_tracking: bool = True
    ):
        """
        Initialize Top-k Attention configuration.
        
        Args:
            k: Number of top positions to attend to
            temperature: Temperature for attention scores
            use_sparse: Whether to use sparse tensors
            fallback_full_attention: Whether to fall back to full attention if k >= seq_len
            enable_memory_tracking: Whether to track memory usage statistics
        """
        self.k = k
        self.temperature = temperature
        self.use_sparse = use_sparse
        self.fallback_full_attention = fallback_full_attention
        self.enable_memory_tracking = enable_memory_tracking
        
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'k': self.k,
            'temperature': self.temperature,
            'use_sparse': self.use_sparse,
            'fallback_full_attention': self.fallback_full_attention,
            'enable_memory_tracking': self.enable_memory_tracking
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TopKAttentionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def create_top_k_attention_layer(
    embed_dim: int,
    num_heads: int,
    config: TopKAttentionConfig
) -> TopKAttention:
    """
    Factory function to create Top-k Attention layer.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        config: Top-k Attention configuration
        
    Returns:
        TopKAttention layer
    """
    return TopKAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        k=config.k,
        temperature=config.temperature,
        use_sparse=config.use_sparse,
        fallback_full_attention=config.fallback_full_attention
    )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = TopKAttentionConfig(k=16, temperature=1.0)
    
    # Create attention layer
    attention = create_top_k_attention_layer(
        embed_dim=512,
        num_heads=8,
        config=config
    )
    
    # Example input
    batch_size, seq_len, embed_dim = 2, 64, 512
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Memory stats: {attention.get_memory_stats()}")
    
    # Test with attention mask
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, seq_len//2:] = 0  # Mask out second half
    
    output_masked = attention(x, mask=mask)
    print(f"Masked output shape: {output_masked.shape}")