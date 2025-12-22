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
from transformers import AutoTokenizer
from config import Config

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
        # Ensure we have a tokenizer for text. Some processors don't include a tokenizer
        # (or may not return attention_mask when used as a combined processor). We
        # prefer to use the processor for images only and a tokenizer for text so
        # the dataset always returns input_ids and attention_mask tensors.
        try:
            # If the provided processor wraps a tokenizer, try to grab it
            self.tokenizer = getattr(processor, "tokenizer", None)
            if self.tokenizer is None:
                cfg = Config()
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        except Exception:
            cfg = Config()
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    
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
        
        # Process image with the processor / feature extractor to get pixel values
        image_inputs = self.processor(images=image, return_tensors="pt")
        # Fix ambiguous tensor evaluation
        pixel_values = image_inputs.get('pixel_values')
        if pixel_values is None:
            raise ValueError("Processor did not return 'pixel_values'. Check the processor configuration.")
        pixel_values = pixel_values.squeeze(0)

        # Tokenize text with tokenizer to guarantee attention_mask exists
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.text_max_length,
            return_tensors='pt'
        )

        # Debugging tokenizer output
        print(f"Tokenizer output for text '{text}': {text_inputs}")

        # Ensure attention_mask matches input_ids size
        if 'attention_mask' not in text_inputs:
            attention_mask = torch.ones_like(text_inputs['input_ids'])
        else:
            attention_mask = text_inputs['attention_mask']

        # Ensure both tensors have consistent shapes
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = attention_mask.squeeze(0)

        # Debugging: Log the shapes of tensors
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"input_ids shape: {input_ids.shape}")

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'source': row['source']
        }