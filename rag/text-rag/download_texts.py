"""
Download sample text files from the web for RAG testing.
"""

import os
import requests
from pathlib import Path


def download_text_file(url: str, output_path: str) -> bool:
    """
    Download a text file from a URL.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✓ Saved to {output_path} ({len(response.text)} characters)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False


def download_sample_texts(output_dir: str = "texts"):
    """
    Download sample text files from public sources.
    
    Args:
        output_dir: Directory to save the text files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample text files from public sources
    # Using Project Gutenberg and other public domain sources
    sources = [
        {
            "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "filename": "pride_and_prejudice.txt",
            "description": "Pride and Prejudice by Jane Austen"
        },
        {
            "url": "https://www.gutenberg.org/files/84/84-0.txt",
            "filename": "frankenstein.txt",
            "description": "Frankenstein by Mary Shelley"
        },
        {
            "url": "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
            "filename": "vscode_readme.txt",
            "description": "VS Code README"
        },
        {
            "url": "https://raw.githubusercontent.com/python/cpython/main/README.rst",
            "filename": "python_readme.txt",
            "description": "Python README"
        },
    ]
    
    downloaded_files = []
    
    for source in sources:
        output_path = os.path.join(output_dir, source["filename"])
        if download_text_file(source["url"], output_path):
            downloaded_files.append(output_path)
    
    print(f"\n✓ Downloaded {len(downloaded_files)} files to {output_dir}/")
    return downloaded_files


def create_sample_text_files(output_dir: str = "texts"):
    """
    Create sample text files locally (useful if downloads fail).
    
    Args:
        output_dir: Directory to save the text files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_texts = {
        "ai_intro.txt": """
Artificial Intelligence (AI) is a branch of computer science that aims to create 
intelligent machines capable of performing tasks that typically require human intelligence. 
These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that enables systems to learn and improve from 
experience without being explicitly programmed. It uses algorithms to analyze data, 
identify patterns, and make decisions or predictions.

Deep Learning is a subset of machine learning that uses neural networks with multiple 
layers to model and understand complex patterns in data. It has been particularly 
successful in areas like image recognition, natural language processing, and speech recognition.

Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
between computers and human language. It enables machines to read, understand, and 
generate human language in a valuable way.
""",
        "python_basics.txt": """
Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum and first released in 1991.

Key features of Python include:
- Easy to learn syntax that emphasizes readability
- Dynamic typing and automatic memory management
- Extensive standard library
- Support for multiple programming paradigms (object-oriented, functional, procedural)
- Large ecosystem of third-party packages

Python is widely used in:
- Web development (Django, Flask)
- Data science and machine learning (NumPy, Pandas, TensorFlow, PyTorch)
- Automation and scripting
- Scientific computing
- Artificial intelligence and natural language processing

The Python community is large and active, contributing to a rich ecosystem of 
libraries and frameworks that make Python suitable for almost any programming task.
""",
        "web_development.txt": """
Web development involves creating websites and web applications. It can be divided 
into frontend (client-side) and backend (server-side) development.

Frontend development focuses on what users see and interact with. Technologies include:
- HTML for structure
- CSS for styling
- JavaScript for interactivity
- Modern frameworks like React, Vue, and Angular

Backend development handles server-side logic, databases, and APIs. Common technologies include:
- Server-side languages: Python, JavaScript (Node.js), Java, PHP, Ruby
- Databases: PostgreSQL, MySQL, MongoDB
- Web frameworks: Django, Express.js, Spring Boot

Full-stack developers work on both frontend and backend. Modern web development 
also involves DevOps practices, cloud deployment, and containerization technologies 
like Docker and Kubernetes.
"""
    }
    
    downloaded_files = []
    
    for filename, content in sample_texts.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        downloaded_files.append(output_path)
        print(f"✓ Created {output_path}")
    
    print(f"\n✓ Created {len(downloaded_files)} sample text files in {output_dir}/")
    return downloaded_files


if __name__ == "__main__":
    import sys
    
    # Try downloading from web first
    print("Attempting to download text files from the web...")
    files = download_sample_texts()
    
    # If downloads failed, create sample files
    if len(files) < 2:
        print("\nSome downloads failed. Creating sample text files instead...")
        files = create_sample_text_files()
    
    print(f"\nReady! Found {len(files)} text files:")
    for f in files:
        print(f"  - {f}")

