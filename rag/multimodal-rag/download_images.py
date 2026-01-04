"""
Download sample images from the web for multimodal RAG testing.
"""

import os
import requests
from pathlib import Path
from PIL import Image
import io
import numpy as np


def download_image(url: str, output_path: str) -> bool:
    """
    Download an image from a URL.
    
    Args:
        url: URL to download from
        output_path: Local path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Verify it's an image
        img = Image.open(io.BytesIO(response.content))
        img.verify()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save the image
        img = Image.open(io.BytesIO(response.content))
        img.save(output_path)
        
        print(f"✓ Saved to {output_path} ({img.size[0]}x{img.size[1]})")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False


def download_sample_images(output_dir: str = "images"):
    """
    Download sample images from public sources.
    
    Args:
        output_dir: Directory to save the images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample images from public sources (using Unsplash and other free image APIs)
    # Note: These URLs may need to be updated or use a different source
    sources = [
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
            "filename": "mountain_landscape.jpg",
            "description": "Mountain landscape",
            "caption": "A beautiful mountain landscape with snow-capped peaks"
        },
        {
            "url": "https://images.unsplash.com/photo-1518791841217-8f162f1e1131",
            "filename": "cat.jpg",
            "description": "Cat",
            "caption": "A cute domestic cat"
        },
        {
            "url": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7",
            "filename": "car.jpg",
            "description": "Car",
            "caption": "A modern car on the road"
        },
    ]
    
    downloaded_files = []
    metadata = []
    
    for source in sources:
        output_path = os.path.join(output_dir, source["filename"])
        if download_image(source["url"], output_path):
            downloaded_files.append(output_path)
            metadata.append({
                'filename': source["filename"],
                'description': source["description"],
                'caption': source["caption"]
            })
    
    print(f"\n✓ Downloaded {len(downloaded_files)} images to {output_dir}/")
    return downloaded_files, metadata


def create_sample_images_from_urls(output_dir: str = "images"):
    """
    Create sample images using direct image URLs (more reliable).
    
    Args:
        output_dir: Directory to save the images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Using direct image URLs from reliable sources
    image_sources = [
        {
            "url": "https://picsum.photos/400/300?random=1",
            "filename": "landscape_1.jpg",
            "caption": "A scenic landscape with mountains and trees"
        },
        {
            "url": "https://picsum.photos/400/300?random=2",
            "filename": "nature_1.jpg",
            "caption": "Natural scenery with vegetation"
        },
        {
            "url": "https://picsum.photos/400/300?random=3",
            "filename": "scenery_1.jpg",
            "caption": "Beautiful outdoor scenery"
        },
        {
            "url": "https://picsum.photos/400/300?random=4",
            "filename": "outdoor_1.jpg",
            "caption": "Outdoor landscape view"
        },
        {
            "url": "https://picsum.photos/400/300?random=5",
            "filename": "view_1.jpg",
            "caption": "Panoramic view of nature"
        },
    ]
    
    downloaded_files = []
    metadata = []
    
    for source in image_sources:
        output_path = os.path.join(output_dir, source["filename"])
        if download_image(source["url"], output_path):
            downloaded_files.append(output_path)
            metadata.append({
                'filename': source["filename"],
                'caption': source["caption"]
            })
    
    print(f"\n✓ Created {len(downloaded_files)} sample images in {output_dir}/")
    return downloaded_files, metadata


def create_placeholder_images(output_dir: str = "images"):
    """
    Create simple placeholder images programmatically.
    Useful when internet access is limited.
    
    Args:
        output_dir: Directory to save the images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        images_data = [
            {
                "filename": "red_circle.jpg",
                "color": (255, 0, 0),
                "shape": "circle",
                "caption": "A red circle on white background"
            },
            {
                "filename": "blue_square.jpg",
                "color": (0, 0, 255),
                "shape": "square",
                "caption": "A blue square on white background"
            },
            {
                "filename": "green_triangle.jpg",
                "color": (0, 255, 0),
                "shape": "triangle",
                "caption": "A green triangle on white background"
            },
            {
                "filename": "yellow_star.jpg",
                "color": (255, 255, 0),
                "shape": "star",
                "caption": "A yellow star on white background"
            },
        ]
        
        downloaded_files = []
        metadata = []
        
        for img_data in images_data:
            # Create a simple image
            img = Image.new('RGB', (400, 400), color='white')
            draw = ImageDraw.Draw(img)
            
            center = (200, 200)
            size = 150
            
            if img_data["shape"] == "circle":
                bbox = [center[0] - size, center[1] - size, center[0] + size, center[1] + size]
                draw.ellipse(bbox, fill=img_data["color"])
            elif img_data["shape"] == "square":
                bbox = [center[0] - size, center[1] - size, center[0] + size, center[1] + size]
                draw.rectangle(bbox, fill=img_data["color"])
            elif img_data["shape"] == "triangle":
                points = [
                    (center[0], center[1] - size),
                    (center[0] - size, center[1] + size),
                    (center[0] + size, center[1] + size)
                ]
                draw.polygon(points, fill=img_data["color"])
            elif img_data["shape"] == "star":
                # Simple star shape
                outer_points = []
                inner_points = []
                for i in range(5):
                    angle = i * 2 * 3.14159 / 5 - 3.14159 / 2
                    outer_points.append((
                        center[0] + size * 0.8 * np.cos(angle),
                        center[1] + size * 0.8 * np.sin(angle)
                    ))
                    angle += 3.14159 / 5
                    inner_points.append((
                        center[0] + size * 0.4 * np.cos(angle),
                        center[1] + size * 0.4 * np.sin(angle)
                    ))
                star_points = []
                for i in range(5):
                    star_points.append(outer_points[i])
                    star_points.append(inner_points[i])
                draw.polygon(star_points, fill=img_data["color"])
            
            output_path = os.path.join(output_dir, img_data["filename"])
            img.save(output_path)
            downloaded_files.append(output_path)
            metadata.append({
                'filename': img_data["filename"],
                'caption': img_data["caption"]
            })
            print(f"✓ Created {output_path}")
        
        print(f"\n✓ Created {len(downloaded_files)} placeholder images in {output_dir}/")
        return downloaded_files, metadata
        
    except ImportError:
        print("PIL/Pillow not available. Cannot create placeholder images.")
        return [], []


if __name__ == "__main__":
    import sys
    
    # Try downloading from web first
    print("Attempting to download images from the web...")
    files, metadata = create_sample_images_from_urls()
    
    # If downloads failed, create placeholder images
    if len(files) < 2:
        print("\nSome downloads failed. Creating placeholder images instead...")
        files, metadata = create_placeholder_images()
    
    print(f"\nReady! Found {len(files)} images:")
    for f, m in zip(files, metadata):
        print(f"  - {f} ({m.get('caption', 'no caption')})")

