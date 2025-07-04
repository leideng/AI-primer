import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# This code implements the scaled dot product attention mechanism, which is a key component of the Transformer architecture.
# It computes attention scores based on the dot product of query and key vectors, scales them,
# applies a softmax function to obtain attention weights, and then uses these weights to compute a
# weighted sum of the value vectors. This implementation is designed to be compatible with PyTorch's
# built-in `scaled_dot_product_attention` function, allowing for easy comparison and validation.    
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0):
    """
    Compute 'Scaled Dot Product Attention'

    :param query: (batch_size, num_heads, seq_len_q, embed_size_per_head)
    :param key: (batch_size, num_heads, seq_len_k, embed_size_per_head)
    :param value: (batch_size, num_heads, seq_len_v, embed_size_per_head)
    :param attn_mask: (seq_len_q, seq_len_k) or (batch_size, num_heads, seq_len_q, seq_len_k)
                      Optional mask applied to the attention mechanism.
    :param dropout_p: Dropout probability for the attention weights.
    :return: output: (batch_size, num_heads, seq_len_q, embed_size_per_head)
             attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    # Calculate dot product between query and key
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply attention mask if provided
    if attn_mask is not None:
        scores += attn_mask

    # Apply softmax to get attention weights
    # Subtract max scores for numerical stability
    # This is the log-sum-exp trick to prevent overflow in softmax
    maxes = torch.max(scores, dim=-1, keepdim=True)[0]
    attn_weights = F.softmax(scores-maxes, dim=-1)

    # Apply dropout to attention weights
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=torch.is_grad_enabled())

    # Multiply attention weights by value
    output = torch.matmul(attn_weights, value)

    return output, scores, attn_weights

# Example usage
if __name__ == "__main__":
    batch_size = 20
    num_heads = 4
    seq_len_q = 40
    seq_len_k = 500
    seq_len_v = 500
    embed_size_per_head = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.manual_seed(0)  # For reproducibility
    q = torch.randn(batch_size, num_heads, seq_len_q, embed_size_per_head, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len_k, embed_size_per_head,  device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len_v, embed_size_per_head,  device=device, dtype=torch.float32)

    start_time = time.time()
    my_output, my_scores, my_attn_weights = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
    end_time = time.time()  
    print("Custom implementation time (secs):", end_time - start_time) 
    
    start_time = time.time()
    # Using PyTorch's built-in scaled dot product attention for comparison 
    pytorch_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    end_time = time.time()
    print("PyTorch built-in implementation time (secs):", end_time - start_time) 
    
    # Print the shapes of the outputs
    print("Custom implementation shapes:")
    print("Output shape:", my_output.shape)
    print("Attention scores shape:", my_scores.shape)
    print("Attention Weights shape:", my_attn_weights.shape)

    # Print the first few values of the outputs for comparison
    items = 5  
    print("First", items, "values of the outputs:") 
    print("Custom implementation output (first", items, "values):", my_output.view(-1)[:items])
    print("PyTorch built-in output (first", items, "values):", pytorch_output.view(-1)[:items])

    # Print the shapes of the PyTorch built-in outputs
    # Note: PyTorch's built-in function returns only the output, not the scores or attention weights
    print("PyTorch built-in implementation shapes:")
    print("Output shape:", pytorch_output.shape)

    # Verify the outputs    
    print("Outputs are close or not:", torch.allclose(my_output, pytorch_output, rtol=1e-3, atol=1e-5))
    print("Sum Difference in outputs:", torch.sum(torch.abs(my_output - pytorch_output)))
    print("Ave Difference in outputs:", torch.mean(torch.abs(my_output - pytorch_output)))
    
    if not torch.allclose(my_output, pytorch_output):
        print("Outputs are not close enough!")
    else:
        print("Outputs are close enough!")  
        print("All checks passed!")
        print("Scaled Dot Product Attention implementation is consistent with PyTorch's built-in function.")
        print("You can now use this implementation in your models.")


