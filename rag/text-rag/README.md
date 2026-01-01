# Simple Text RAG System

A minimal, easy-to-use Retrieval-Augmented Generation (RAG) system designed to run in Google Colab.

## Features

- **Simple API**: Easy to use with just a few lines of code
- **Colab Compatible**: Works seamlessly in Google Colab
- **Lightweight**: Uses efficient sentence transformers for embeddings
- **Flexible**: Works with any text files
- **No External APIs**: Runs entirely locally (no API keys needed)

## Quick Start

### In Google Colab

1. **Install dependencies:**
```python
!pip install sentence-transformers requests numpy
```

2. **Upload files or clone repository:**
   - Upload `simple_rag.py` and `download_texts.py` to Colab
   - Or clone from your repository

3. **Run the example:**
```python
from simple_rag import SimpleTextRAG
from download_texts import create_sample_texts

# Create sample text files
text_files = create_sample_texts("texts")

# Initialize RAG
rag = SimpleTextRAG()
rag.load_documents(text_files)
rag.build_index()

# Query
result = rag.query("What is Python?", top_k=3)
print(result['context'])
```

### Local Usage

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run example:**
```python
from simple_rag import SimpleTextRAG
from download_texts import download_sample_texts

# Download sample texts
text_files = download_sample_texts("texts")

# Initialize and use RAG
rag = SimpleTextRAG()
rag.load_documents(text_files, chunk_size=500, chunk_overlap=50)
rag.build_index()

# Query
result = rag.query("What is artificial intelligence?", top_k=3)
print(result['context'])
```

## Usage Examples

### Basic Usage

```python
from simple_rag import SimpleTextRAG

# Initialize
rag = SimpleTextRAG(embedding_model_name="all-MiniLM-L6-v2")

# Load your text files
rag.load_documents(["file1.txt", "file2.txt"], chunk_size=500)

# Build index
rag.build_index()

# Query
result = rag.query("Your question here", top_k=3)
print(result['context'])
```

### Advanced Usage

```python
# Retrieve specific chunks
chunks = rag.retrieve("machine learning", top_k=5)
for chunk in chunks:
    print(f"Similarity: {chunk['similarity']:.3f}")
    print(f"Text: {chunk['text']}")
    print(f"From: {chunk['metadata']['file']}")

# Save and load
rag.save("my_rag.pkl")
rag2 = SimpleTextRAG()
rag2.load("my_rag.pkl")
```

## Components

- **simple_rag.py**: Main RAG system implementation
- **download_texts.py**: Script to download/create sample text files
- **colab_notebook.ipynb**: Complete Colab notebook example
- **requirements.txt**: Python dependencies

## How It Works

1. **Document Loading**: Text files are loaded and split into chunks
2. **Embedding**: Each chunk is converted to a vector using sentence transformers
3. **Indexing**: Embeddings are stored for fast retrieval
4. **Retrieval**: Query is embedded and compared with document embeddings using cosine similarity
5. **Generation**: Retrieved chunks provide context for answering questions

## Customization

- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` in `load_documents()`
- **Embedding Model**: Change `embedding_model_name` (e.g., "all-mpnet-base-v2" for better quality)
- **Top-K**: Adjust number of retrieved chunks with `top_k` parameter

## Requirements

- Python 3.7+
- sentence-transformers
- requests
- numpy

## License

MIT License

