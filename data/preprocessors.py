"""
Image and text preprocessing utilities
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing following MedSigLIP specification"""
    
    def __init__(self, target_size: int = 448):
        self.target_size = target_size
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image using TensorFlow bilinear interpolation
        This matches MedSigLIP's preprocessing
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize using TensorFlow
        resized = tf.image.resize(
            images=img_array,
            size=[self.target_size, self.target_size],
            method='bilinear',
            antialias=False
        )
        
        # Convert back to PIL Image
        resized_array = resized.numpy().astype(np.uint8)
        return Image.fromarray(resized_array)
    
    def load_and_preprocess(self, image_path: str) -> Image.Image:
        """Load image from path and preprocess"""
        try:
            image = Image.open(image_path)
            return self.resize_image(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Strip whitespace
        text = text.strip()
        
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 200) -> str:
        """Truncate text to max character length"""
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
        return text
    
    def preprocess(self, text: str, max_length: int = 200) -> str:
        """Full text preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.truncate_text(text, max_length)
        return text