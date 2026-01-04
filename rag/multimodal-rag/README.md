# Simple Multimodal RAG System

A minimal, easy-to-use Multimodal Retrieval-Augmented Generation (RAG) system designed to run in Google Colab. Uses CLIP to retrieve relevant images based on text queries, enhancing image generation prompts.

## Features

- **Simple API**: Easy to use with just a few lines of code
- **Colab Compatible**: Works seamlessly in Google Colab
- **CLIP-based**: Uses CLIP for image-text embeddings
- **Image Retrieval**: Retrieve relevant images based on text queries
- **Prompt Enhancement**: Enhance image generation prompts using retrieved images
- **No External APIs**: Runs entirely locally (no API keys needed)

## Quick Start

### In Google Colab

1. **Install dependencies:**
```python
%pip install transformers torch pillow requests numpy matplotlib
```

2. **Upload files or clone repository:**
   - Upload `simple_multimodal_rag.py` and `download_images.py` to Colab
   - Or clone from your repository

3. **Run the example:**
```python
from simple_multimodal_rag import SimpleMultimodalRAG
from download_images import create_sample_images_from_urls

# Create sample images
image_files, metadata = create_sample_images_from_urls("images")

# Initialize RAG
rag = SimpleMultimodalRAG()
rag.load_images(image_files, metadata)
rag.build_index()

# Query
result = rag.query("mountain landscape", top_k=3)
print(f"Found {result['num_images']} relevant images")
```

### Local Usage

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run example:**
```python
from simple_multimodal_rag import SimpleMultimodalRAG
from download_images import create_sample_images_from_urls

# Download sample images
image_files, metadata = create_sample_images_from_urls("images")

# Initialize and use RAG
rag = SimpleMultimodalRAG()
rag.load_images(image_files, metadata)
rag.build_index()

# Query
result = rag.query("nature scenery", top_k=3)
for img_info in result['retrieved_images']:
    print(f"{img_info['image_path']} - similarity: {img_info['similarity']:.3f}")
```

## Usage Examples

### Basic Usage

```python
from simple_multimodal_rag import SimpleMultimodalRAG

# Initialize
rag = SimpleMultimodalRAG(clip_model_name="openai/clip-vit-base-patch32")

# Load your images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
metadata = [
    {"caption": "A beautiful sunset"},
    {"caption": "Mountain landscape"},
    {"caption": "City skyline"}
]
rag.load_images(image_paths, metadata)

# Build index
rag.build_index()

# Query
result = rag.query("sunset over mountains", top_k=2)
```

### Enhance Image Generation Prompts

```python
# Original prompt
original_prompt = "a futuristic city"

# Enhance using retrieved images
enhanced_prompt = rag.enhance_prompt(original_prompt, top_k=2, style="descriptive")
print(enhanced_prompt)
# Output: "a futuristic city, inspired by visual styles from: similar to city_skyline.jpg, similar to modern_building.jpg"

# Use enhanced prompt with your image generation model
# generated_image = image_generation_model.generate(enhanced_prompt)
```

### Visualize Retrieved Images

```python
from simple_multimodal_rag import visualize_retrieved_images

result = rag.query("nature landscape", top_k=5)
visualize_retrieved_images(result['retrieved_images'], max_display=5)
```

### Add Images with Captions

```python
# Add a single image with caption
rag.add_image_with_caption(
    "new_image.jpg",
    "A serene lake surrounded by mountains",
    metadata={"source": "user_upload"}
)
rag.build_index()  # Rebuild index
```

## Components

- **simple_multimodal_rag.py**: Main multimodal RAG system implementation
- **download_images.py**: Script to download/create sample images
- **colab_notebook.ipynb**: Complete Colab notebook example
- **requirements.txt**: Python dependencies

## How It Works

1. **Image Loading**: Images are loaded and stored with optional metadata/captions
2. **Embedding**: Each image is converted to a vector using CLIP's image encoder
3. **Indexing**: Image embeddings are stored for fast retrieval
4. **Retrieval**: Text query is embedded using CLIP's text encoder and compared with image embeddings using cosine similarity
5. **Enhancement**: Retrieved images provide visual context to enhance image generation prompts

## Use Cases

- **Image Generation Enhancement**: Retrieve similar images to improve prompts for image generation models
- **Visual Search**: Find images based on text descriptions
- **Style Transfer**: Find images with similar styles to reference
- **Content-Based Image Retrieval**: Search your image collection using natural language

## Customization

- **CLIP Model**: Change `clip_model_name` (e.g., "openai/clip-vit-large-patch14" for better quality but slower)
- **Top-K**: Adjust number of retrieved images with `top_k` parameter
- **Prompt Style**: Choose enhancement style: 'descriptive', 'concise', or 'detailed'

## Requirements

- Python 3.7+
- transformers (for CLIP)
- torch
- pillow (for image processing)
- requests (for downloading images)
- numpy
- matplotlib (for visualization, optional)

## License

MIT License

