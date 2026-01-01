"""
Simple Text RAG (Retrieval-Augmented Generation) System
Designed to run in Google Colab with minimal dependencies.
"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path


class SimpleTextRAG:
    """
    A simple RAG system that uses sentence embeddings and cosine similarity
    for retrieval, and can work with any text generation model.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use.
                                  Default is a lightweight model suitable for Colab.
        """
        self.embedding_model_name = embedding_model_name
        self.embeddings_model = None
        self.documents = []
        self.embeddings = None
        self.chunk_metadata = []  # Store metadata for each chunk
        
    def _load_embedding_model(self):
        """Lazy load the embedding model."""
        if self.embeddings_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.embedding_model_name}")
                self.embeddings_model = SentenceTransformer(self.embedding_model_name)
                print("Embedding model loaded successfully!")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
    
    def load_documents(self, text_files: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Load and chunk documents from text files.
        
        Args:
            text_files: List of paths to text files
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.documents = []
        self.chunk_metadata = []
        
        for file_path in text_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking by character count
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.chunk_metadata.append({
                    'file': os.path.basename(file_path),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
        
        print(f"Loaded {len(self.documents)} chunks from {len(text_files)} files")
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            
            if end >= len(text):
                break
            start = end - chunk_overlap
        
        return chunks
    
    def build_index(self):
        """Build the embedding index for all documents."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        self._load_embedding_model()
        
        print("Building embedding index...")
        self.embeddings = self.embeddings_model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Index built with {len(self.embeddings)} embeddings")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunk text, metadata, and similarity score
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        self._load_embedding_model()
        
        # Encode query
        query_embedding = self.embeddings_model.encode(query, convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'metadata': self.chunk_metadata[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def query(self, question: str, top_k: int = 3, use_context: bool = True) -> Dict:
        """
        Query the RAG system and return context for generation.
        
        Args:
            question: The question to answer
            top_k: Number of relevant chunks to retrieve
            use_context: Whether to include retrieved context
            
        Returns:
            Dictionary with query, retrieved context, and metadata
        """
        retrieved = self.retrieve(question, top_k=top_k)
        
        if use_context:
            context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved)])
        else:
            context = ""
        
        return {
            'query': question,
            'context': context,
            'retrieved_chunks': retrieved,
            'num_chunks': len(retrieved)
        }
    
    def save(self, path: str):
        """Save the RAG system to disk."""
        save_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'chunk_metadata': self.chunk_metadata,
            'embedding_model_name': self.embedding_model_name
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"RAG system saved to {path}")
    
    def load(self, path: str):
        """Load the RAG system from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.documents = save_data['documents']
        self.embeddings = save_data['embeddings']
        self.chunk_metadata = save_data['chunk_metadata']
        self.embedding_model_name = save_data['embedding_model_name']
        
        print(f"RAG system loaded from {path}")
        print(f"Loaded {len(self.documents)} document chunks")


def generate_answer(query_result: Dict, model=None, max_length: int = 200) -> str:
    """
    Generate an answer using the retrieved context.
    
    Args:
        query_result: Result from rag.query()
        model: Optional model for generation (if None, returns formatted context)
        max_length: Maximum length of generated answer
        
    Returns:
        Generated answer or formatted context
    """
    query = query_result['query']
    context = query_result['context']
    
    if model is None:
        # Simple template-based response
        return f"""Based on the retrieved context:

{context}

Question: {query}

Answer: [Use the context above to answer the question. In a real implementation, you would use an LLM here.]"""
    
    # If a model is provided, use it for generation
    prompt = f"""Context:
{context}

Question: {query}

Answer:"""
    
    # This is a placeholder - replace with actual model generation
    if hasattr(model, 'generate'):
        return model.generate(prompt, max_length=max_length)
    else:
        return prompt


# Simple test function
def test_rag():
    """Test the RAG system with sample data."""
    # Create sample documents
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing allows computers to understand and process human language.",
    ]
    
    # Save to temporary files
    import tempfile
    temp_files = []
    for i, doc in enumerate(sample_docs):
        temp_file = f"/tmp/test_doc_{i}.txt"
        with open(temp_file, 'w') as f:
            f.write(doc)
        temp_files.append(temp_file)
    
    # Initialize and test RAG
    rag = SimpleTextRAG()
    rag.load_documents(temp_files)
    rag.build_index()
    
    # Test query
    result = rag.query("What is Python?", top_k=2)
    print("\nQuery Result:")
    print(f"Query: {result['query']}")
    print(f"\nRetrieved Context:\n{result['context']}")
    
    # Cleanup
    for f in temp_files:
        os.remove(f)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_rag()

