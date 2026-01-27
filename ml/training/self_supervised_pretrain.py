"""Advanced Training Techniques for MaxSight 3.0

Includes self-supervised pretraining, knowledge distillation, and continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MAE(nn.Module):
    """Masked Autoencoder for vision pretraining."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with masking."""
        B, C, H, W = x.shape
        num_patches = (H // 16) * (W // 16)
        num_masked = int(num_patches * self.mask_ratio)
        mask = torch.rand(B, num_patches, device=x.device).topk(num_masked, dim=1).indices
        encoded = self.encoder(x, mask)
        decoded = self.decoder(encoded)
        return {'reconstruction': decoded, 'mask': mask}


class SimCLR(nn.Module):
    """SimCLR contrastive learning."""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.temperature = temperature
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Contrastive learning forward."""
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        similarity = torch.matmul(z1, z2.t()) / self.temperature
        return {'similarity': similarity}


class KnowledgeDistillation(nn.Module):
    """Teacher-student knowledge distillation."""
    
    def __init__(self, teacher: nn.Module, student: nn.Module, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                        labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute distillation loss."""
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        return {'total_loss': total_loss, 'kd_loss': kd_loss, 'ce_loss': ce_loss}


class ElasticWeightConsolidation(nn.Module):
    """EWC for continual learning."""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        super().__init__()
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optimal_params = {}
    
    def compute_fisher(self, dataloader, criterion):
        """Compute Fisher information matrix."""
        self.model.eval()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_info[name] = torch.zeros_like(param.data)
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(batch['input'])
            loss = criterion(output, batch['target'])
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_info[name] += param.grad.data ** 2
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty."""
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                ewc_loss += (self.fisher_info[name] * (param - self.optimal_params[name]) ** 2).sum()
        return self.lambda_ewc * ewc_loss

