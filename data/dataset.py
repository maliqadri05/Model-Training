"""
PyTorch Dataset for medical image-text pairs
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import Dict, Any
import logging
from .preprocessors import ImagePreprocessor, TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageTextDataset(Dataset):
    """
    Unified medical image-text dataset
    Loads from preprocessed parquet file
    """
    
    def __init__(
        self,
        data_path: str,
        processor: Any,
        image_size: int = 448,
        text_max_length: int = 64,
        split: str = 'train'
    ):
        """
        Args:
            data_path: Path to parquet file with image_path, text, source columns
            processor: HuggingFace processor for SigLIP
            image_size: Target image size
            text_max_length: Maximum text token length
            split: 'train' or 'val'
        """
        self.processor = processor
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.split = split
        
        # Load data
        logger.info(f"Loading {split} data from {data_path}...")
        self.data = pd.read_parquet(data_path)
        
        # Filter by split if split column exists
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(target_size=image_size)
        self.text_preprocessor = TextPreprocessor()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        try:
            image = self.image_preprocessor.load_and_preprocess(row['image_path'])
        except Exception as e:
            logger.warning(f"Failed to load image at index {idx}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), color='white')
        
        # Preprocess text
        text = self.text_preprocessor.preprocess(row['text'])
        
        # Process with SigLIP processor
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'source': row['source']
        }