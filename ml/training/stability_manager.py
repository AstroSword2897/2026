"""Adaptive Training Stability Manager for MaxSight."""

import logging
import math
from typing import Dict, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class StabilityMetrics:
    """Metrics for stability assessment."""
    epoch: int
    train_loss: float
    val_loss: float
    train_val_gap: float  # Overfitting indicator.
    loss_spike: bool  # Sudden loss increase.
    loss_unstable: bool  # NaN or Inf.
    task_imbalance: float  # GradNorm weight variance.
    lr_current: float
    weight_decay_current: float
    # Set when check_and_adjust applies an adjustment (for logging in train_loop)
    lr_adjusted: bool = False
    wd_adjusted: bool = False
    new_lr: Optional[float] = None
    new_wd: Optional[float] = None


class StabilityManager:
    """Monitors training stability and auto-adjusts hyperparameters."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        gradnorm_loss: Optional[Any] = None,
        # Thresholds.
        spike_threshold: float = 0.3,  # Loss increase > 30% = spike.
        overfit_threshold: float = 0.25,  # Val > train by 25% = overfit.
        task_imbalance_threshold: float = 3.0,  # Max/min weight ratio.
        # Adjustments.
        lr_reduce_factor: float = 0.5,  # Halve LR on spike.
        wd_increase_factor: float = 1.5,  # 1.5x weight decay on overfit.
        max_wd: float = 0.5,  # Cap weight decay.
        min_lr: float = 1e-7,  # Floor LR.
        # Logging.
        log_every: int = 1,
    ):
        """Initialize stability manager."""
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradnorm_loss = gradnorm_loss
        
        self.spike_threshold = spike_threshold
        self.overfit_threshold = overfit_threshold
        self.task_imbalance_threshold = task_imbalance_threshold
        
        self.lr_reduce_factor = lr_reduce_factor
        self.wd_increase_factor = wd_increase_factor
        self.max_wd = max_wd
        self.min_lr = min_lr
        
        self.log_every = log_every
        
        # History for spike detection.
        self.loss_history = []
        self.stability_history = []
        
        # State.
        self.adjustments_made = 0
        self.last_adjustment_epoch = -1
    
    def check_and_adjust(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> StabilityMetrics:
        """Check stability and auto-adjust hyperparameters."""
        # Get current LR and weight decay (guard empty param_groups)
        if not self.optimizer.param_groups:
            logger.warning("StabilityManager: optimizer has no param_groups, skipping check")
            return StabilityMetrics(
                epoch=epoch,
                train_loss=float(train_loss) if train_loss is not None else 0.0,
                val_loss=float(val_loss) if val_loss is not None else 0.0,
                train_val_gap=0.0,
                loss_spike=False,
                loss_unstable=True,
                task_imbalance=0.0,
                lr_current=0.0,
                weight_decay_current=0.0,
            )
        lr_current = self.optimizer.param_groups[0]['lr']
        wd_current = self.optimizer.param_groups[0].get('weight_decay', 0.0)
        train_loss_f = float(train_loss) if train_loss is not None else 0.0
        val_loss_f = float(val_loss) if val_loss is not None else 0.0
        # Detect instability (safe for NaN/Inf)
        try:
            loss_unstable = (
                not math.isfinite(train_loss_f) or
                not math.isfinite(val_loss_f)
            )
        except (TypeError, ValueError):
            loss_unstable = True
        
        # Detect spike (loss increased sharply from recent average)
        loss_spike = False
        if len(self.loss_history) >= 3:
            recent_avg = sum(self.loss_history[-3:]) / 3
            if train_loss_f > recent_avg * (1 + self.spike_threshold):
                loss_spike = True
        
        # Detect overfitting.
        train_val_gap = (val_loss_f - train_loss_f) / (train_loss_f + 1e-8)
        overfitting = train_val_gap > self.overfit_threshold
        
        # Detect task imbalance (GradNorm)
        task_imbalance = 0.0
        try:
            if self.gradnorm_loss is not None and hasattr(self.gradnorm_loss, 'task_weights'):
                weights = self.gradnorm_loss.task_weights.detach().cpu()
                if len(weights) > 1:
                    task_imbalance = weights.max().item() / (weights.min().item() + 1e-8)
        except (AttributeError, RuntimeError) as e:
            logger.debug("StabilityManager: could not compute task_imbalance: %s", e)
        
        # Build metrics (adjustment fields set below when we apply changes)
        metrics = StabilityMetrics(
            epoch=epoch,
            train_loss=train_loss_f,
            val_loss=val_loss_f,
            train_val_gap=train_val_gap,
            loss_spike=loss_spike,
            loss_unstable=loss_unstable,
            task_imbalance=task_imbalance,
            lr_current=lr_current,
            weight_decay_current=wd_current,
        )
        
        # Apply adjustments.
        adjustments = []
        
        # 1. Handle NaN/Inf: aggressive LR reduction.
        if loss_unstable:
            new_lr = max(lr_current * 0.1, self.min_lr)
            self._set_lr(new_lr)
            metrics.lr_adjusted = True
            metrics.new_lr = new_lr
            adjustments.append(f"NaN/Inf detected, reduced LR: {lr_current:.2e} → {new_lr:.2e}")
            self.adjustments_made += 1
            self.last_adjustment_epoch = epoch
        
        # 2. Handle loss spike: moderate LR reduction.
        elif loss_spike and epoch > 5:  # Skip early epochs (warmup)
            new_lr = max(lr_current * self.lr_reduce_factor, self.min_lr)
            self._set_lr(new_lr)
            metrics.lr_adjusted = True
            metrics.new_lr = new_lr
            adjustments.append(
                f"Loss spike detected ({train_loss_f:.4f} vs recent {sum(self.loss_history[-3:])/3:.4f}), "
                f"reduced LR: {lr_current:.2e} → {new_lr:.2e}"
            )
            self.adjustments_made += 1
            self.last_adjustment_epoch = epoch
        
        # 3. Handle overfitting: increase weight decay.
        if overfitting and epoch > 10:
            new_wd = min(wd_current * self.wd_increase_factor, self.max_wd)
            if new_wd > wd_current:
                self._set_weight_decay(new_wd)
                metrics.wd_adjusted = True
                metrics.new_wd = new_wd
                adjustments.append(
                    f"Overfitting detected (val-train gap: {train_val_gap:.2%}), "
                    f"increased weight decay: {wd_current:.2e} → {new_wd:.2e}"
                )
                self.adjustments_made += 1
        
        # 4. Handle task imbalance: warn (GradNorm self-corrects)
        if task_imbalance > self.task_imbalance_threshold and epoch > 15:
            adjustments.append(
                f"Task imbalance detected (weight ratio: {task_imbalance:.2f}). "
                "GradNorm will rebalance; monitor for improvement."
            )
        
        # Log adjustments.
        if adjustments:
            logger.warning(f"Epoch {epoch} stability adjustments:")
            for adj in adjustments:
                logger.warning(f"  - {adj}")
        
        # Log stability metrics periodically.
        if epoch % self.log_every == 0:
            logger.info(
                f"Stability check (epoch {epoch}): "
                f"train_loss={train_loss_f:.4f}, val_loss={val_loss_f:.4f}, "
                f"gap={train_val_gap:.2%}, LR={lr_current:.2e}, WD={wd_current:.2e}, "
                f"adjustments_total={self.adjustments_made}"
            )
        
        # Update history.
        self.loss_history.append(train_loss_f)
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
        self.stability_history.append(metrics)
        
        return metrics
    
    def _set_lr(self, new_lr: float):
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def _set_weight_decay(self, new_wd: float):
        """Set weight decay for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_wd
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stability management."""
        return {
            'total_adjustments': self.adjustments_made,
            'last_adjustment_epoch': self.last_adjustment_epoch,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'current_wd': self.optimizer.param_groups[0].get('weight_decay', 0.0),
            'history_length': len(self.stability_history),
        }






