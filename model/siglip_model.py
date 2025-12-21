"""
SigLIP model wrapper for medical fine-tuning
"""
import torch
import torch.nn as nn
from transformers import AutoModel
try:
    from .loss import SigmoidContrastiveLoss
except ImportError:
    from model.loss import SigmoidContrastiveLoss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SigLIPMedical(nn.Module):
    """
    Wrapper for SigLIP model with sigmoid contrastive loss
    """
    
    def __init__(self, config):
        """
        Args:
            config: ModelConfig instance
        """
        super().__init__()
        self.config = config
        
        # Load pretrained SigLIP model
        logger.info(f"Loading pretrained model: {config.model_name}")
        self.model = AutoModel.from_pretrained(config.model_name)
        
        # Get embedding dimensions
        self.vision_embed_dim = self.model.config.vision_config.hidden_size
        self.text_embed_dim = self.model.config.text_config.hidden_size
        
        # Initialize sigmoid contrastive loss
        self.loss_fn = SigmoidContrastiveLoss(
            init_temperature=config.init_temperature,
            init_bias=config.init_bias
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_loss: bool = True
    ):
        """
        Forward pass
        
        Args:
            pixel_values: [batch_size, 3, height, width]
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            return_loss: Whether to compute loss
        
        Returns:
            If return_loss=True: (loss, image_embeds, text_embeds)
            If return_loss=False: (image_embeds, text_embeds)
        """
        # Get embeddings from model
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        if return_loss:
            # Compute sigmoid contrastive loss
            loss = self.loss_fn(image_embeds, text_embeds)
            return loss, image_embeds, text_embeds
        else:
            return image_embeds, text_embeds
    
    def get_loss_info(self) -> dict:
        """Get current loss parameters"""
        return {
            'temperature': self.loss_fn.get_temperature(),
            'bias': self.loss_fn.get_bias()
        }