"""
Main training script for SigLIP medical fine-tuning
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
import logging
from pathlib import Path
import random
import numpy as np

from config import Config
from data.dataset import MedicalImageTextDataset
from model.siglip_model import SigLIPMedical
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train SigLIP on medical data")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-path', type=str, help='Path to processed parquet file')
    parser.add_argument('--output-dir', type=str, help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override with command line args
    if args.data_path:
        config.data.processed_data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size:
        config.training.per_gpu_batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    else:
        # Set default number of epochs to 25
        config.training.num_epochs = 25
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Set seed
    set_seed(config.training.seed)
    
    # Check data file exists
    if not Path(config.data.processed_data_path).exists():
        logger.error(f"Data file not found: {config.data.processed_data_path}")
        logger.error("Please run: python prepare_datasets.py")
        return
    
    logger.info("="*60)
    logger.info("SigLIP Medical Fine-tuning")
    logger.info("="*60)
    logger.info(f"Data: {config.data.processed_data_path}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Output: {config.training.output_dir}")
    logger.info(f"Image size: {config.model.image_size}")
    logger.info(f"Text max length: {config.model.text_max_length}")
    logger.info(f"Batch size per GPU: {config.training.per_gpu_batch_size}")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.training.per_gpu_batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Mixed precision: {config.training.mixed_precision}")
    logger.info("="*60)
    
    # Load processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(config.model.model_name)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MedicalImageTextDataset(
        data_path=config.data.processed_data_path,
        processor=processor,
        image_size=config.model.image_size,
        text_max_length=config.model.text_max_length,
        split='train'
    )
    
    val_dataset = MedicalImageTextDataset(
        data_path=config.data.processed_data_path,
        processor=processor,
        image_size=config.model.image_size,
        text_max_length=config.model.text_max_length,
        split='val'
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = SigLIPMedical(config.model).to(config.training.device)
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler
    num_training_steps = len(train_loader) * config.training.num_epochs // config.training.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)
    
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.training.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint(is_best=False)
        logger.info("Checkpoint saved")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()