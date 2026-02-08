"""Advanced Training Techniques for MaxSight 3.0 (Production v2)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MAELoss(nn.Module):
    """Masked Autoencoder reconstruction loss (production v2)."""
    
    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MAE reconstruction loss."""
        loss = (recon - target) ** 2
        loss = loss.mean(dim=-1)          # Per-patch MSE.
        loss = (loss * mask.float()).sum() / (mask.sum().float() + 1e-8)  # Only masked patches.
        return loss


class SimCLRLoss(nn.Module):
    """NT-Xent contrastive loss (SimCLR) - production v2. Batch-stable, AMP-safe, correct positives/negatives. Args: temperature: Temperature for softmax (default: 0.07)"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent contrastive loss."""
        B = z1.size(0)
        
        # Normalize embeddings.
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate all representations.
        representations = torch.cat([z1, z2], dim=0)  # [2*B, D].
        
        # Compute similarity matrix.
        similarity = torch.matmul(representations, representations.T) / self.temperature  # [2*B, 2*B].
        
        # Create labels: positive pairs are (i, i+B) for i in [0, B-1].
        labels = torch.arange(B, device=z1.device)
        labels = torch.cat([labels + B, labels])  # [2*B].
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=z1.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float("-inf"))
        
        # Cross-entropy loss.
        loss = F.cross_entropy(similarity, labels)
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """Standard teacher-student knowledge distillation loss (production v2)."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute knowledge distillation loss."""
        # Teacher forward pass is frozen.
        with torch.no_grad():
            teacher_soft = F.softmax(
                teacher_logits / self.temperature, dim=1
            )
        
        # Student log probabilities.
        student_log_soft = F.log_softmax(
            student_logits / self.temperature, dim=1
        )
        
        # KL divergence loss (scaled by temperature^2)
        kd_loss = F.kl_div(
            student_log_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy loss.
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss.
        total = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return {
            "total_loss": total,
            "kd_loss": kd_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for continual learning (production v2)."""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optimal_params = {}
    
    @torch.no_grad()
    def consolidate(self):
        """Save current model parameters as optimal (call after training on task)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.clone()
    
    def compute_fisher(self, dataloader, loss_fn, device):
        """Compute Fisher information matrix from dataloader."""
        self.model.eval()
        fisher = {}
        
        # Initialize Fisher matrices.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        # Accumulate Fisher information.
        num_samples = 0
        for batch in dataloader:
            self.model.zero_grad()
            
            # Parse batch (supports dict or tuple)
            if isinstance(batch, dict):
                inputs = batch.get('input', batch.get('images', batch.get('image')))
                targets = batch.get('target', batch.get('labels', batch.get('targets')))
            elif isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            # Forward pass.
            outputs = self.model(inputs)
            
            # Compute loss.
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                # If no targets, assume outputs is loss dict.
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs.get('loss', torch.tensor(0.0, device=device))
            
            # Backward pass.
            loss.backward()
            
            # Accumulate Fisher (gradient squared)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2)
            
            num_samples += 1
        
        # Normalize Fisher by number of samples.
        for name in fisher:
            fisher[name] /= max(num_samples, 1)
        
        self.fisher = fisher
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty term (additive, differentiable). Returns: Scalar penalty tensor."""
        loss = torch.tensor(0.0, device=next(iter(self.model.parameters())).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.optimal_params:
                fisher = self.fisher[name]
                optimal = self.optimal_params[name]
                diff = (param - optimal) ** 2
                loss = loss + (fisher * diff).sum()
        
        return self.lambda_ewc * loss


# Backward compatibility aliases.
ReconstructionLoss = MAELoss
MaskingSIM = SimCLRLoss
KnowledgeDistillation = KnowledgeDistillationLoss

# Test compatibility aliases (old class names)
MAE = MAELoss  # For tests - but MAE should be a model, not a loss.
SimCLR = SimCLRLoss  # For tests - but SimCLR should be a model, not a loss.






