"""
Google Colab Example for Simple Text RAG System

Run this in a Google Colab notebook cell by cell.
"""

# Cell 1: Install dependencies
"""
!pip install sentence-transformers requests numpy
"""

# Cell 2: Import and setup
"""
import os
from simple_rag import SimpleTextRAG
from download_texts import download_sample_texts, create_sample_texts

# Download or create sample text files
print("Setting up text files...")
try:
    text_files = download_sample_texts("texts")
except:
    print("Download failed, creating sample files...")
    text_files = create_sample_texts("texts")
"""

# Cell 3: Initialize RAG system
"""
# Initialize the RAG system
rag = SimpleTextRAG(embedding_model_name="all-MiniLM-L6-v2")

# Load documents
rag.load_documents(text_files, chunk_size=500, chunk_overlap=50)

# Build the embedding index
rag.build_index()
"""

# Cell 4: Query the system
"""
# Query the RAG system
query = "What is Python?"
result = rag.query(query, top_k=3)

print(f"Query: {result['query']}\n")
print("Retrieved Context:")
print(result['context'])
print("\n" + "="*50)
print("Retrieved Chunks:")
for i, chunk in enumerate(result['retrieved_chunks']):
    print(f"\nChunk {i+1} (similarity: {chunk['similarity']:.3f}):")
    print(f"From: {chunk['metadata']['file']}")
    print(f"Text: {chunk['text'][:200]}...")
"""

# Cell 5: Multiple queries
"""
queries = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "What are the features of Python?",
    "What is web development?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    result = rag.query(query, top_k=2)
    print(f"\nTop result (similarity: {result['retrieved_chunks'][0]['similarity']:.3f}):")
    print(result['retrieved_chunks'][0]['text'][:300] + "...")
"""

# Cell 6: Save and load
"""
# Save the RAG system
rag.save("rag_system.pkl")

# Load it back
rag2 = SimpleTextRAG()
rag2.load("rag_system.pkl")

# Test loaded system
result = rag2.query("What is Python?", top_k=2)
print(result['context'][:500])
"""

