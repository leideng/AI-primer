import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SelfAttention:
    def __init__(self, d_model=512, h=8):
        self.d_model = d_model  # Model dimension
        self.h = h  # Number of attention heads
        self.d_k = d_model // h  # Dimension of each head
        
        # Initialize random weights (in practice these would be learned)
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        
    def split_heads(self, x):
        """Split the last dimension into (h, d_k)"""
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.h, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, X):
        """
        Compute self-attention
        X: input of shape (batch_size, seq_len, d_model)
        """
        # Linear transformations
        Q = np.dot(X, self.W_q)  # Query
        K = np.dot(X, self.W_k)  # Key
        V = np.dot(X, self.W_v)  # Value
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention_weights = softmax(scores)
        attention_output = np.matmul(attention_weights, V)
        
        return attention_output, attention_weights

def demonstrate_basic_self_attention():
    """Demonstrate self-attention with a simple example similar to the Excel file"""
    # Example sequence
    sequence = np.array([
        [1, 0, 1],  # Input vector 1
        [0, 1, 0],  # Input vector 2
        [0, 1, 1],  # Input vector 3
    ])
    
    # Expand dimensions to match expected shape (batch_size, seq_len, d_model)
    X = sequence.reshape(1, 3, 3)
    
    # Create self-attention instance
    attention = SelfAttention(d_model=3, h=1)
    
    # Compute attention
    output, weights = attention.forward(X)
    
    print("\nBasic Self-Attention Example:")
    print("Input sequence:")
    print(sequence)
    print("\nAttention weights:")
    print(weights[0, 0])  # First head, first batch
    print("\nOutput:")
    print(output[0, 0])  # First head, first batch

def analyze_bert_attention():
    """Analyze attention patterns in BERT"""
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Example sentence
    text = "The cat sits on the mat."
    
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention weights from the first layer
    attention = outputs.attentions[0]  # Shape: (batch_size, num_heads, seq_len, seq_len)
    
    # Convert tokens to readable format
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Plot attention weights for the first head
    plt.figure(figsize=(10, 8))
    plt.imshow(attention[0, 0].numpy(), cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title('BERT Self-Attention Weights (First Head)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    plt.close()
    
    print("\nBERT Attention Analysis:")
    print("Tokens:", tokens)
    print("\nAttention matrix shape:", attention.shape)
    print("\nAttention weights for first token attending to all other tokens:")
    for i, (token, weight) in enumerate(zip(tokens, attention[0, 0, 0].numpy())):
        print(f"{token}: {weight:.4f}")

if __name__ == "__main__":
    # Demonstrate basic self-attention
    demonstrate_basic_self_attention()
    
    # Analyze BERT attention patterns
    analyze_bert_attention() 