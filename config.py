"""
Configuration for SigLIP Medical Fine-tuning
Optimized for 4 datasets: Knee X-Ray, PAD-UFES-20, SCIN, SLAKE
"""
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Raw data directories (your downloaded datasets)
    raw_data_root: str = "./data/Datasets"
    knee_xray_path: str = "./data/Datasets/Mendeley Digital Knee X-Ray"
    pad_ufes_path: str = "./data/Datasets/PAD-UFES-20 (Skin lesion dataset)"
    scin_path: str = "./data/Datasets/SCIN Dermatology"
    slake_path: str = "./data/Datasets/SLAKE VQA"
    
    # Processed data
    processed_data_path: str = "./data/processed/all_med_pairs.parquet"
    train_split: float = 0.95
    val_split: float = 0.05
    
    # Dataset mixing weights (adjust based on your dataset sizes)
    dataset_weights: dict = field(default_factory=lambda: {
        "knee_xray": 0.5,      # Adjust based on actual size
        "pad_ufes_20": 1.0,
        "scin": 1.0,
        "slake": 1.0
    })


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = "google/siglip-base-patch16-224"
    image_size: int = 448  # Following MedSigLIP
    text_max_length: int = 64  # Following MedSigLIP
    
    # Loss parameters
    init_temperature: float = np.log(10)  # t' = log(10)
    init_bias: float = -10.0


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Batch and accumulation
    per_gpu_batch_size: int = 8  # Adjust for your GPU
    gradient_accumulation_steps: int = 8  # Effective batch = 64
    num_epochs: int = 2  # Testing with 2 epochs
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95  # Reduced for stability
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_ratio: float = 0.1  # 10% warmup
    lr_scheduler_type: str = "cosine"
    
    # Training settings
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware
    device: str = "cuda"  # Will be auto-detected in __post_init__
    local_rank: int = -1  # For distributed training
    seed: int = 42


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Auto-detect device FIRST (before creating directories)
        import torch
        if self.training.device == "cuda" and not torch.cuda.is_available():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("CUDA not available, falling back to CPU. Training will be slower.")
            logger.warning("To use GPU, install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            self.training.device = "cpu"
            # Disable mixed precision on CPU (not supported)
            self.training.mixed_precision = False
            # Reduce batch size for CPU
            if self.training.per_gpu_batch_size > 4:
                logger.warning(f"Reducing batch size from {self.training.per_gpu_batch_size} to 4 for CPU training")
                self.training.per_gpu_batch_size = 4
        
        # Create directories
        os.makedirs(os.path.dirname(self.data.processed_data_path), exist_ok=True)
        os.makedirs(self.training.output_dir, exist_ok=True)