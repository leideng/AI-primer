"""
Simple Multimodal RAG (Retrieval-Augmented Generation) System
Designed to run in Google Colab with minimal dependencies.
Uses CLIP for image-text embeddings to retrieve relevant images for image generation.
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch


class SimpleMultimodalRAG:
    """
    A simple multimodal RAG system that uses CLIP embeddings to retrieve
    relevant images based on text queries, useful for enhancing image generation prompts.
    """
    
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the multimodal RAG system.
        
        Args:
            clip_model_name: Name of the CLIP model to use.
                           Default is a lightweight model suitable for Colab.
        """
        self.clip_model_name = clip_model_name
        self.clip_model = None
        self.clip_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.images = []  # List of image paths
        self.image_embeddings = None  # Image embeddings
        self.image_metadata = []  # Metadata for each image
        
    def _load_clip_model(self):
        """Lazy load the CLIP model."""
        if self.clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                print(f"Loading CLIP model: {self.clip_model_name}")
                print(f"Using device: {self.device}")
                
                self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
                self.clip_model.eval()
                
                print("CLIP model loaded successfully!")
            except ImportError:
                raise ImportError(
                    "transformers not installed or CLIP not available. "
                    "Install with: pip install transformers torch pillow"
                )
    
    def load_images(self, image_paths: List[str], metadata: Optional[List[Dict]] = None):
        """
        Load images into the RAG system.
        
        Args:
            image_paths: List of paths to image files
            metadata: Optional list of metadata dictionaries for each image
        """
        self.images = []
        self.image_metadata = []
        
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping...")
                continue
            
            self.images.append(img_path)
            
            # Store metadata
            if metadata and i < len(metadata):
                self.image_metadata.append(metadata[i])
            else:
                self.image_metadata.append({
                    'path': img_path,
                    'filename': os.path.basename(img_path),
                    'index': len(self.images) - 1
                })
        
        print(f"Loaded {len(self.images)} images")
    
    def build_index(self):
        """Build the embedding index for all images."""
        if not self.images:
            raise ValueError("No images loaded. Call load_images() first.")
        
        self._load_clip_model()
        
        print("Building image embedding index...")
        image_embeddings_list = []
        
        for i, img_path in enumerate(self.images):
            try:
                # Load and process image
                image = Image.open(img_path).convert('RGB')
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                
                # Get image embedding
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                    image_embeddings_list.append(image_features.cpu().numpy()[0])
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(self.images)} images...")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add zero vector as placeholder
                image_embeddings_list.append(np.zeros(512))  # Default CLIP embedding size
        
        self.image_embeddings = np.array(image_embeddings_list)
        print(f"Index built with {len(self.image_embeddings)} image embeddings")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query into embedding space.
        
        Args:
            text: Text query string
            
        Returns:
            Normalized text embedding vector
        """
        self._load_clip_model()
        
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
        
        return text_features.cpu().numpy()[0]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant images for a text query.
        
        Args:
            query: The search query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing image path, metadata, and similarity score
        """
        if self.image_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(self.image_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.images[idx],
                'metadata': self.image_metadata[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def query(self, text_query: str, top_k: int = 3) -> Dict:
        """
        Query the multimodal RAG system and return relevant images.
        
        Args:
            text_query: The text query describing desired images
            top_k: Number of relevant images to retrieve
            
        Returns:
            Dictionary with query, retrieved images, and metadata
        """
        retrieved = self.retrieve(text_query, top_k=top_k)
        
        return {
            'query': text_query,
            'retrieved_images': retrieved,
            'num_images': len(retrieved)
        }
    
    def enhance_prompt(self, text_query: str, top_k: int = 3, 
                      style: str = "descriptive") -> str:
        """
        Enhance an image generation prompt using retrieved images.
        
        Args:
            text_query: Original text query
            top_k: Number of images to use for enhancement
            style: How to enhance ('descriptive', 'concise', 'detailed')
            
        Returns:
            Enhanced prompt string
        """
        retrieved = self.retrieve(text_query, top_k=top_k)
        
        if not retrieved:
            return text_query
        
        # Extract visual concepts from retrieved images
        image_descriptions = []
        for img_info in retrieved:
            filename = img_info['metadata'].get('filename', 'image')
            # Could add image captioning here, but keeping it simple
            image_descriptions.append(f"similar to {filename}")
        
        if style == "descriptive":
            enhanced = f"{text_query}, inspired by visual styles from: {', '.join(image_descriptions)}"
        elif style == "concise":
            enhanced = f"{text_query} (style reference: {image_descriptions[0]})"
        else:  # detailed
            enhanced = f"{text_query}\nVisual style references:\n" + "\n".join([f"- {desc}" for desc in image_descriptions])
        
        return enhanced
    
    def save(self, path: str):
        """Save the RAG system to disk."""
        save_data = {
            'images': self.images,
            'image_embeddings': self.image_embeddings,
            'image_metadata': self.image_metadata,
            'clip_model_name': self.clip_model_name
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Multimodal RAG system saved to {path}")
    
    def load(self, path: str):
        """Load the RAG system from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.images = save_data['images']
        self.image_embeddings = save_data['image_embeddings']
        self.image_metadata = save_data['image_metadata']
        self.clip_model_name = save_data['clip_model_name']
        
        print(f"Multimodal RAG system loaded from {path}")
        print(f"Loaded {len(self.images)} images")
    
    def add_image_with_caption(self, image_path: str, caption: str, metadata: Optional[Dict] = None):
        """
        Add a single image with a caption for better retrieval.
        
        Args:
            image_path: Path to the image file
            caption: Text caption describing the image
            metadata: Optional additional metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.images.append(image_path)
        
        meta = {
            'path': image_path,
            'filename': os.path.basename(image_path),
            'caption': caption,
            'index': len(self.images) - 1
        }
        if metadata:
            meta.update(metadata)
        
        self.image_metadata.append(meta)
        
        # Rebuild index if it exists
        if self.image_embeddings is not None:
            self.build_index()


def visualize_retrieved_images(retrieved_images: List[Dict], max_display: int = 5):
    """
    Display retrieved images (for use in notebooks).
    
    Args:
        retrieved_images: List of retrieved image dictionaries
        max_display: Maximum number of images to display
    """
    try:
        from IPython.display import display, Image as IPImage
        import matplotlib.pyplot as plt
        
        num_images = min(len(retrieved_images), max_display)
        
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        if num_images == 1:
            axes = [axes]
        
        for i, img_info in enumerate(retrieved_images[:num_images]):
            img_path = img_info['image_path']
            similarity = img_info['similarity']
            
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Similarity: {similarity:.3f}", fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib/IPython not available. Install with: pip install matplotlib ipython")
        print(f"Retrieved {len(retrieved_images)} images:")
        for img_info in retrieved_images[:max_display]:
            print(f"  - {img_info['image_path']} (similarity: {img_info['similarity']:.3f})")


# Simple test function
def test_multimodal_rag():
    """Test the multimodal RAG system with sample data."""
    # This would require actual images, so we'll just test the structure
    print("Multimodal RAG test requires actual images.")
    print("Please use the Colab notebook or provide image paths.")


if __name__ == "__main__":
    test_multimodal_rag()

