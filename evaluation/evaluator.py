"""
Evaluation utilities for SigLIP medical model
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SigLIPEvaluator:
    """Evaluator for SigLIP medical model"""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def compute_embeddings(self, dataloader, modality='both'):
        """
        Compute embeddings for a dataset
        
        Args:
            dataloader: DataLoader
            modality: 'image', 'text', or 'both'
        
        Returns:
            Dict with embeddings and metadata
        """
        image_embeds_list = []
        text_embeds_list = []
        sources = []
        
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get embeddings
            image_embeds, text_embeds = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=False
            )
            
            if modality in ['image', 'both']:
                image_embeds_list.append(image_embeds.cpu())
            if modality in ['text', 'both']:
                text_embeds_list.append(text_embeds.cpu())
            
            sources.extend(batch['source'])
        
        result = {'sources': sources}
        
        if modality in ['image', 'both']:
            result['image_embeds'] = torch.cat(image_embeds_list, dim=0)
        if modality in ['text', 'both']:
            result['text_embeds'] = torch.cat(text_embeds_list, dim=0)
        
        return result
    
    def compute_retrieval_metrics(self, image_embeds, text_embeds, k_values=[1, 5, 10]):
        """
        Compute image-text retrieval metrics
        
        Args:
            image_embeds: [N, D] tensor
            text_embeds: [N, D] tensor
            k_values: List of k for Recall@k
        
        Returns:
            Dict with retrieval metrics
        """
        # Normalize embeddings
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(image_embeds, text_embeds.t())
        
        n = similarity.shape[0]
        
        # Image to text retrieval
        i2t_ranks = []
        for i in range(n):
            # Get ranking of correct text for this image
            sims = similarity[i]
            sorted_indices = torch.argsort(sims, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            i2t_ranks.append(rank + 1)  # 1-indexed
        
        # Text to image retrieval
        t2i_ranks = []
        for i in range(n):
            sims = similarity[:, i]
            sorted_indices = torch.argsort(sims, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            t2i_ranks.append(rank + 1)
        
        # Compute Recall@K
        metrics = {}
        for k in k_values:
            i2t_recall = np.mean([1 if r <= k else 0 for r in i2t_ranks])
            t2i_recall = np.mean([1 if r <= k else 0 for r in t2i_ranks])
            
            metrics[f'image_to_text_R@{k}'] = i2t_recall
            metrics[f'text_to_image_R@{k}'] = t2i_recall
        
        # Median rank
        metrics['image_to_text_median_rank'] = np.median(i2t_ranks)
        metrics['text_to_image_median_rank'] = np.median(t2i_ranks)
        
        # Mean rank
        metrics['image_to_text_mean_rank'] = np.mean(i2t_ranks)
        metrics['text_to_image_mean_rank'] = np.mean(t2i_ranks)
        
        return metrics
    
    def evaluate(self, dataloader):
        """
        Full evaluation pipeline
        
        Returns:
            Dict with all metrics
        """
        logger.info("Starting evaluation...")
        
        # Compute embeddings
        embeds = self.compute_embeddings(dataloader, modality='both')
        
        # Compute retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(
            embeds['image_embeds'],
            embeds['text_embeds']
        )
        
        logger.info("Evaluation completed")
        logger.info("\nRetrieval Metrics:")
        for key, value in retrieval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return {
            'retrieval': retrieval_metrics,
            'embeddings': embeds
        }