"""
Training utility functions
"""
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def setup_distributed():
    """
    Setup for distributed training
    
    Returns:
        Dict with rank, world_size, and local_rank
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        logger.info("Not using distributed mode")
        return {
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'distributed': False
        }
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        'distributed': True
    }


def get_parameter_groups(model, config):
    """
    Create parameter groups with different learning rates or weight decay
    Useful for fine-tuning pretrained models
    
    Args:
        model: PyTorch model
        config: Training configuration
    
    Returns:
        List of parameter groups for optimizer
    """
    # Separate parameters that should not have weight decay
    no_decay_params = []
    decay_params = []
    
    # Parameters from loss function (temperature, bias)
    loss_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Loss function parameters
        if 'loss_fn' in name:
            loss_params.append(param)
        # Bias and LayerNorm parameters should not have weight decay
        elif 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': config.training.weight_decay,
            'lr': config.training.learning_rate
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'lr': config.training.learning_rate
        },
        {
            'params': loss_params,
            'weight_decay': 0.0,
            'lr': config.training.learning_rate
        }
    ]
    
    logger.info(f"Parameter groups created:")
    logger.info(f"  - Decay params: {len(decay_params)}")
    logger.info(f"  - No decay params: {len(no_decay_params)}")
    logger.info(f"  - Loss params: {len(loss_params)}")
    
    return param_groups


def save_training_args(args: Dict[str, Any], output_dir: str):
    """
    Save training arguments to JSON
    
    Args:
        args: Dictionary of training arguments
        output_dir: Directory to save to
    """
    output_path = Path(output_dir) / "training_args.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert non-serializable objects to strings
    serializable_args = {}
    for key, value in args.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_args[key] = value
        else:
            serializable_args[key] = str(value)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_args, f, indent=2)
    
    logger.info(f"Training arguments saved to {output_path}")


def count_parameters(model, trainable_only=False):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_gpu_memory_info():
    """
    Get GPU memory usage information
    
    Returns:
        Dict with memory info for each GPU
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        
        memory_info[f'gpu_{i}'] = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated
        }
    
    return memory_info


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricLogger:
    """Logger for tracking multiple metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def add_metric(self, name: str):
        """Add a new metric to track"""
        self.metrics[name] = AverageMeter(name)
        self.history[name] = []
    
    def update(self, name: str, value: float, n: int = 1):
        """Update metric value"""
        if name not in self.metrics:
            self.add_metric(name)
        self.metrics[name].update(value, n)
    
    def log_epoch(self):
        """Log current epoch metrics and save to history"""
        for name, meter in self.metrics.items():
            self.history[name].append(meter.avg)
            logger.info(f"  {name}: {meter.avg:.4f}")
    
    def reset_epoch(self):
        """Reset all metrics for new epoch"""
        for meter in self.metrics.values():
            meter.reset()
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full history of all metrics"""
        return self.history
    
    def save_history(self, filepath: str):
        """Save metric history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Metric history saved to {filepath}")


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None):
    """
    Load checkpoint and restore model, optimizer, scheduler states
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        Dict with checkpoint information
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
    }
    
    logger.info(f"Checkpoint loaded: epoch {info['epoch']}, step {info['global_step']}")
    
    return info


def freeze_parameters(model, pattern: str):
    """
    Freeze parameters matching a pattern
    
    Args:
        model: PyTorch model
        pattern: String pattern to match parameter names
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if pattern in name:
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Froze {frozen_count} parameters matching pattern '{pattern}'")


def unfreeze_parameters(model, pattern: str):
    """
    Unfreeze parameters matching a pattern
    
    Args:
        model: PyTorch model
        pattern: String pattern to match parameter names
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if pattern in name:
            param.requires_grad = True
            unfrozen_count += 1
    
    logger.info(f"Unfroze {unfrozen_count} parameters matching pattern '{pattern}'")