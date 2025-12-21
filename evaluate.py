"""
Evaluation script for trained SigLIP medical model
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import logging
from pathlib import Path
import json

from config import Config
from data.dataset import MedicalImageTextDataset
from model.siglip_model import SigLIPMedical
from evaluation.evaluator import SigLIPEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SigLIP medical model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, help='Path to evaluation data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Data split')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    if args.data_path:
        config.data.processed_data_path = args.data_path
    
    logger.info("="*60)
    logger.info("SigLIP Medical Model Evaluation")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data: {config.data.processed_data_path}")
    logger.info(f"Split: {args.split}")
    logger.info("="*60)
    
    # Load processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(config.model.model_name)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = MedicalImageTextDataset(
        data_path=config.data.processed_data_path,
        processor=processor,
        image_size=config.model.image_size,
        text_max_length=config.model.text_max_length,
        split=args.split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Load model
    logger.info("Loading model...")
    model = SigLIPMedical(config.model).to(config.training.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=config.training.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create evaluator
    evaluator = SigLIPEvaluator(
        model=model,
        processor=processor,
        device=config.training.device
    )
    
    # Run evaluation
    logger.info("\n" + "="*60)
    logger.info("Running evaluation...")
    logger.info("="*60 + "\n")
    
    results = evaluator.evaluate(dataloader)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_path = checkpoint_dir / f"eval_results_{args.split}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for saving (remove embeddings)
    save_results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(dataset),
        'retrieval_metrics': results['retrieval']
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()