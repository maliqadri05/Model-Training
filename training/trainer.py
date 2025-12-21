"""
Training loop and utilities
"""
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for SigLIP medical fine-tuning"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        config
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            dynamic_ncols=True
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.config.training.device)
            input_ids = batch['input_ids'].to(self.config.training.device)
            attention_mask = batch['attention_mask'].to(self.config.training.device)
            
            # Forward pass with mixed precision
            if self.config.training.mixed_precision:
                with autocast():
                    loss, _, _ = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = loss / self.config.training.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                loss, _, _ = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.mixed_precision:
                    # Unscale gradients
                    self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    loss_info = self.model.get_loss_info()
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {loss.item() * self.config.training.gradient_accumulation_steps:.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                        f"Temp: {loss_info['temperature']:.2f} | "
                        f"Bias: {loss_info['bias']:.2f}"
                    )
                
                # Evaluation
                if self.global_step % self.config.training.eval_steps == 0:
                    val_loss = self.evaluate()
                    self.model.train()
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(is_best=False)
            
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.training.gradient_accumulation_steps:.4f}",
                'step': self.global_step
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        logger.info("Running evaluation...")
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(self.config.training.device)
            input_ids = batch['input_ids'].to(self.config.training.device)
            attention_mask = batch['attention_mask'].to(self.config.training.device)
            
            if self.config.training.mixed_precision:
                with autocast():
                    loss, _, _ = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                loss, _, _ = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'model': vars(self.config.model),
                'training': vars(self.config.training)
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = Path(self.config.training.output_dir) / "best_model.pt"
            logger.info(f"Saving best model to {path}")
        else:
            path = Path(self.config.training.output_dir) / f"checkpoint_step_{self.global_step}.pt"
            logger.info(f"Saving checkpoint to {path}")
        
        torch.save(checkpoint, path)
        
        # Save model in HuggingFace format
        if is_best:
            hf_path = Path(self.config.training.output_dir) / "best_model_hf"
            self.model.model.save_pretrained(hf_path)
            logger.info(f"Saved HuggingFace model to {hf_path}")
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.training.num_epochs}")
        logger.info(f"Steps per epoch: {len(self.train_dataloader)}")
        logger.info(f"Total steps: {len(self.train_dataloader) * self.config.training.num_epochs}")
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} completed | Train Loss: {train_loss:.4f}")
            
            # Evaluate at end of epoch
            val_loss = self.evaluate()
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")