from transformers import pipeline
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

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

plt.figure(figsize=(8, 8))
plt.imshow(attention, cmap="viridis")
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.colorbar()
plt.title("Self-Attention (Last Layer, Head 0)")
plt.show()
