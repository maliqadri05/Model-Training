"""
Sigmoid contrastive loss for SigLIP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidContrastiveLoss(nn.Module):
    """
    Pairwise Sigmoid Contrastive Loss for Language-Image Pre-training
    Based on Algorithm 1 in SigLIP paper (arXiv:2303.15343)
    """
    
    def __init__(self, init_temperature: float, init_bias: float):
        """
        Args:
            init_temperature: Initial value for learnable temperature (t')
            init_bias: Initial value for learnable bias
        """
        super().__init__()
        
        # Learnable temperature (stored as t', actual temp is exp(t'))
        self.t_prime = nn.Parameter(torch.tensor(init_temperature))
        
        # Learnable bias
        self.bias = nn.Parameter(torch.tensor(init_bias))
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sigmoid contrastive loss
        
        Args:
            image_embeds: [batch_size, embedding_dim]
            text_embeds: [batch_size, embedding_dim]
        
        Returns:
            loss: scalar tensor
        """
        batch_size = image_embeds.shape[0]
        
        # Get temperature (exponential of t')
        temperature = torch.exp(self.t_prime)
        
        # L2 normalize embeddings
        image_embeds_norm = F.normalize(image_embeds, p=2, dim=1)
        text_embeds_norm = F.normalize(text_embeds, p=2, dim=1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        logits = torch.matmul(image_embeds_norm, text_embeds_norm.t())
        logits = logits * temperature + self.bias
        
        # Create labels: 1 on diagonal (positive pairs), -1 elsewhere
        labels = 2 * torch.eye(batch_size, device=logits.device) - 1
        
        # Compute sigmoid loss: -log(sigmoid(labels * logits))
        loss = -F.logsigmoid(labels * logits).sum() / batch_size
        
        return loss
    
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return torch.exp(self.t_prime).item()
    
    def get_bias(self) -> float:
        """Get current bias value"""
        return self.bias.item()