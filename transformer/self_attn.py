from transformers import pipeline
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

# Input sentence
sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="pt")

# Forward pass with attention outputs
outputs = model(**inputs)
attentions = outputs.attentions  # Tuple: (num_layers, batch, num_heads, seq_len, seq_len)

# Visualize attention from the last layer, first head
attention = attentions[-1][0][0].detach().numpy()  # (seq_len, seq_len)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Print attention values as a table
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(attention, cmap="viridis")
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
fig.colorbar(cax)

# Annotate each cell with the attention value
for (i, j), val in np.ndenumerate(attention):
    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='w' if val < 0.5 else 'black', fontsize=8)

plt.title("Self-Attention (Last Layer, Head 0)")
plt.tight_layout()
plt.show()

# Also print the table for reference
df = pd.DataFrame(attention, columns=tokens, index=tokens)
print("Self-Attention (Last Layer, Head 0):")
print(df.round(3))
