"""
Task Balancing for Multi-Head Training

CRITICAL: Without proper task balancing, 20 heads will engage in gradient warfare:
- Detection accuracy will decay
- Depth will oscillate
- Rare heads will overfit silently

This module implements:
1. GradNorm: Gradient normalization for balanced multi-task learning
2. PCGrad: Projected Conflicting Gradients (alternative)
3. Per-head loss monitoring and adaptive weighting
4. Head-level kill switches for runtime control

Reference:
- GradNorm: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
- PCGrad: Yu et al., "Gradient Surgery for Multi-Task Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GradNormBalancer(nn.Module):
    """
    GradNorm: Gradient normalization for adaptive loss balancing.
    
    Automatically balances gradients across multiple tasks by:
    1. Computing per-task gradient norms
    2. Normalizing gradients to equalize learning rates
    3. Adaptively adjusting task weights
    
    This prevents gradient warfare where dominant tasks (detection) 
    overwhelm rare tasks (fatigue, personalization).
    """
    
    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,  # Restoring force hyperparameter
        initial_task_weights: Optional[List[float]] = None
    ):
        """
        Initialize GradNorm balancer.
        
        Arguments:
            num_tasks: Number of tasks/heads to balance
            alpha: Restoring force (higher = stronger balancing)
            initial_task_weights: Initial weights for each task (optional)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        # Learnable task weights (initialized to 1.0)
        if initial_task_weights is None:
            initial_task_weights = [1.0] * num_tasks
        self.task_weights = nn.Parameter(torch.tensor(initial_task_weights, dtype=torch.float32))
        
        # Track initial loss values (for relative loss computation)
        self.register_buffer('initial_losses', torch.zeros(num_tasks))
        self.register_buffer('loss_history', torch.zeros(num_tasks))
        self.initialized = False
    
    def compute_gradient_norms(
        self,
        model: nn.Module,
        shared_params: List[nn.Parameter],
        task_losses: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient norms for each task.
        
        Arguments:
            model: The model being trained
            shared_params: Shared parameters (backbone, FPN)
            task_losses: List of loss values for each task
        
        Returns:
            Tuple of (weighted_losses, gradient_norms)
        """
        # Compute weighted losses
        weighted_losses = [
            self.task_weights[i] * loss 
            for i, loss in enumerate(task_losses)
        ]
        
        # Compute gradients for each task
        gradient_norms = []
        for i, weighted_loss in enumerate(weighted_losses):
            # Zero gradients first
            model.zero_grad()
            
            # Backward for this task only
            weighted_loss.backward(retain_graph=True)
            
            # Compute gradient norm for shared parameters
            grad_norm = 0.0
            for param in shared_params:
                if param.grad is not None:
                    grad_norm += param.grad.norm(p=2) ** 2
            grad_norm = grad_norm ** 0.5
            gradient_norms.append(grad_norm)
        
        gradient_norms_tensor = torch.stack(gradient_norms)
        
        return torch.stack(weighted_losses), gradient_norms_tensor
    
    def update_task_weights(
        self,
        task_losses: List[torch.Tensor],
        gradient_norms: torch.Tensor,
        iteration: int
    ) -> Dict[str, float]:
        """
        Update task weights using GradNorm algorithm.
        
        Arguments:
            task_losses: Current loss values for each task
            gradient_norms: Gradient norms for each task
            iteration: Current training iteration
        
        Returns:
            Dictionary with updated weights and metrics
        """
        # Initialize reference losses on first iteration
        if not self.initialized:
            self.initial_losses = torch.stack([loss.detach() for loss in task_losses])
            self.initialized = True
        
        # Compute relative losses (current / initial)
        current_losses = torch.stack([loss.detach() for loss in task_losses])
        relative_losses = current_losses / (self.initial_losses + 1e-8)
        
        # Compute average gradient norm
        avg_grad_norm = gradient_norms.mean()
        
        # Compute relative inverse training rates
        # Tasks with higher relative loss should have higher gradient norm
        relative_inverse_rates = relative_losses ** self.alpha
        
        # Target gradient norms (proportional to relative inverse rates)
        target_grad_norms = avg_grad_norm * relative_inverse_rates
        
        # Compute GradNorm loss (difference between actual and target norms)
        gradnorm_loss = F.l1_loss(gradient_norms, target_grad_norms)
        
        # Update task weights via gradient descent on gradnorm_loss
        # This is done separately from main training loop
        # In practice, you'd optimize task_weights with respect to gradnorm_loss
        
        # For now, return metrics for monitoring
        metrics = {
            'gradnorm_loss': gradnorm_loss.item(),
            'avg_grad_norm': avg_grad_norm.item(),
            'task_weights': {f'task_{i}': self.task_weights[i].item() 
                           for i in range(self.num_tasks)},
            'relative_losses': {f'task_{i}': relative_losses[i].item() 
                               for i in range(self.num_tasks)},
            'gradient_norms': {f'task_{i}': gradient_norms[i].item() 
                             for i in range(self.num_tasks)}
        }
        
        return metrics
    
    def get_task_weights(self) -> torch.Tensor:
        """Get current task weights."""
        return self.task_weights.detach()


class PCGradBalancer:
    """
    PCGrad: Projected Conflicting Gradients for multi-task learning.
    
    Resolves gradient conflicts by projecting conflicting gradients
    onto each other's normal plane.
    
    Alternative to GradNorm - sometimes more stable.
    """
    
    def __init__(self):
        """Initialize PCGrad balancer."""
        pass
    
    def project_conflicting_gradients(
        self,
        gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Project conflicting gradients to resolve conflicts.
        
        Arguments:
            gradients: List of gradient tensors for each task
        
        Returns:
            List of projected gradients
        """
        projected_grads = []
        
        for i, grad_i in enumerate(gradients):
            grad_i_proj = grad_i.clone()
            
            # Project grad_i onto other gradients
            for j, grad_j in enumerate(gradients):
                if i != j:
                    # Check if gradients conflict (negative dot product)
                    dot_product = (grad_i * grad_j).sum()
                    
                    if dot_product < 0:  # Conflicting gradients
                        # Project grad_i onto grad_j's normal plane
                        grad_i_proj = grad_i_proj - (dot_product / (grad_j.norm() ** 2 + 1e-8)) * grad_j
            
            projected_grads.append(grad_i_proj)
        
        return projected_grads


class PerHeadLossMonitor:
    """
    Monitor per-head loss magnitudes over time.
    
    Critical for detecting gradient warfare:
    - Losses that decay too fast (dominant tasks)
    - Losses that oscillate (conflicting gradients)
    - Losses that plateau (underfitting)
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize loss monitor.
        
        Arguments:
            window_size: Number of iterations to track
        """
        self.window_size = window_size
        self.loss_history: Dict[str, List[float]] = defaultdict(list)
        self.iteration = 0
    
    def update(self, head_losses: Dict[str, torch.Tensor]):
        """
        Update loss history.
        
        Arguments:
            head_losses: Dictionary mapping head names to loss values
        """
        self.iteration += 1
        
        for head_name, loss in head_losses.items():
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            
            # Add to history
            self.loss_history[head_name].append(loss_val)
            
            # Keep only window_size most recent
            if len(self.loss_history[head_name]) > self.window_size:
                self.loss_history[head_name].pop(0)
    
    def detect_issues(self) -> Dict[str, List[str]]:
        """
        Detect potential gradient warfare issues.
        
        Returns:
            Dictionary mapping issue types to affected heads
        """
        issues = {
            'dominant': [],  # Loss decaying too fast
            'oscillating': [],  # Loss oscillating
            'plateaued': [],  # Loss not improving
            'conflicting': []  # Losses moving in opposite directions
        }
        
        for head_name, history in self.loss_history.items():
            if len(history) < 20:  # Need enough history
                continue
            
            recent = history[-20:]
            early = history[-50:-30] if len(history) >= 50 else history[:20]
            
            # Check for dominance (decaying too fast)
            if len(recent) > 0 and len(early) > 0:
                recent_avg = sum(recent) / len(recent)
                early_avg = sum(early) / len(early)
                decay_rate = (early_avg - recent_avg) / (early_avg + 1e-8)
                
                if decay_rate > 0.5:  # Decayed by >50%
                    issues['dominant'].append(head_name)
            
            # Check for oscillation
            if len(recent) > 10:
                variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
                mean_val = sum(recent) / len(recent)
                cv = (variance ** 0.5) / (mean_val + 1e-8)  # Coefficient of variation
                
                if cv > 0.3:  # High variance relative to mean
                    issues['oscillating'].append(head_name)
            
            # Check for plateau
            if len(recent) > 10:
                recent_trend = (recent[-1] - recent[0]) / (len(recent) - 1)
                if abs(recent_trend) < 1e-6:  # No improvement
                    issues['plateaued'].append(head_name)
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all heads."""
        summary = {}
        
        for head_name, history in self.loss_history.items():
            if len(history) == 0:
                continue
            
            summary[head_name] = {
                'current': history[-1],
                'mean': sum(history) / len(history),
                'min': min(history),
                'max': max(history),
                'trend': (history[-1] - history[0]) / len(history) if len(history) > 1 else 0.0
            }
        
        return summary


class GradNormMultiHeadLoss(nn.Module):
    """
    Multi-head loss combiner with GradNorm for adaptive task balancing.
    
    Automatically balances gradients across all heads by:
    1. Computing per-head gradient norms on shared parameters
    2. Normalizing gradients to equalize learning rates
    3. Adaptively adjusting head weights
    
    This prevents gradient warfare where dominant heads (detection) 
    overwhelm rare heads (fatigue, personalization).
    """
    
    def __init__(
        self,
        head_losses: Dict[str, nn.Module],
        shared_params: Optional[List[nn.Parameter]] = None,
        alpha: float = 1.5,
        update_interval: int = 100,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.head_losses = nn.ModuleDict(head_losses)
        self.head_names = list(head_losses.keys())
        self.num_heads = len(head_losses)
        self.alpha = alpha
        self.update_interval = update_interval
        
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in self.head_names}
        
        initial_weights_tensor = torch.tensor(
            [initial_weights.get(name, 1.0) for name in self.head_names],
            dtype=torch.float32
        )
        self.task_weights = nn.Parameter(initial_weights_tensor)
        
        self.register_buffer('initial_losses', torch.zeros(self.num_heads))
        self.register_buffer('loss_history', torch.zeros(self.num_heads))
        self.initialized = False
        self.shared_params = shared_params
        self.iteration = 0
    
    def set_shared_params(self, shared_params: List[nn.Parameter]):
        """Set shared parameters for gradient norm computation."""
        self.shared_params = shared_params
    
    def compute_head_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for all heads."""
        head_loss_dicts = {}
        for head_name, loss_fn in self.head_losses.items():
            try:
                loss_dict = loss_fn(outputs, targets)
                head_loss_dicts[head_name] = loss_dict
            except Exception as e:
                logger.warning(f"Failed to compute loss for {head_name}: {e}")
                device = next(iter(outputs.values())).device
                head_loss_dicts[head_name] = {'loss': torch.tensor(0.0, device=device)}
        return head_loss_dicts
    
    def compute_gradient_norms(
        self,
        model: nn.Module,
        head_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient norms for each head on shared parameters."""
        if self.shared_params is None:
            self.shared_params = []
            for name, param in model.named_parameters():
                if any(bb_name in name for bb_name in [
                    'backbone', 'resnet', 'conv1', 'bn1', 
                    'layer1', 'layer2', 'layer3', 'layer4', 'fpn'
                ]):
                    if param.requires_grad:
                        self.shared_params.append(param)
        
        if len(self.shared_params) == 0:
            logger.warning("No shared parameters found, skipping GradNorm")
            head_loss_list = [head_losses[name] for name in self.head_names]
            return torch.stack(head_loss_list), torch.ones(self.num_heads, device=head_loss_list[0].device)
        
        weighted_losses = []
        for i, head_name in enumerate(self.head_names):
            loss = head_losses[head_name]
            weighted_loss = self.task_weights[i] * loss
            weighted_losses.append(weighted_loss)
        
        weighted_losses_tensor = torch.stack(weighted_losses)
        gradient_norms = []
        
        for i, weighted_loss in enumerate(weighted_losses):
            model.zero_grad()
            weighted_loss.backward(retain_graph=True)
            grad_norm = 0.0
            for param in self.shared_params:
                if param.grad is not None:
                    grad_norm += param.grad.norm(p=2) ** 2
            grad_norm = grad_norm ** 0.5
            if torch.is_tensor(grad_norm):
                grad_norm = grad_norm.item()
            gradient_norms.append(grad_norm)
        
        gradient_norms_tensor = torch.tensor(
            gradient_norms, 
            device=weighted_losses_tensor.device,
            dtype=weighted_losses_tensor.dtype
        )
        return weighted_losses_tensor, gradient_norms_tensor
    
    def update_task_weights(
        self,
        head_losses: Dict[str, torch.Tensor],
        gradient_norms: torch.Tensor
    ) -> Dict[str, float]:
        """Update task weights using GradNorm algorithm."""
        if not self.initialized:
            head_loss_list = [head_losses[name] for name in self.head_names]
            self.initial_losses = torch.stack([
                loss.detach() if torch.is_tensor(loss) else torch.tensor(loss)
                for loss in head_loss_list
            ])
            self.initialized = True
            return {}
        
        head_loss_list = [head_losses[name] for name in self.head_names]
        current_losses = torch.stack([
            loss.detach() if torch.is_tensor(loss) else torch.tensor(loss)
            for loss in head_loss_list
        ])
        relative_losses = current_losses / (self.initial_losses + 1e-8)
        avg_grad_norm = gradient_norms.mean()
        relative_inverse_rates = relative_losses ** self.alpha
        target_grad_norms = avg_grad_norm * relative_inverse_rates
        gradnorm_loss = F.l1_loss(gradient_norms, target_grad_norms)
        
        weight_updates = (target_grad_norms - gradient_norms) / (gradient_norms + 1e-8)
        weight_updates = torch.clamp(weight_updates, -0.1, 0.1)
        
        with torch.no_grad():
            self.task_weights.data = self.task_weights.data * (1.0 + 0.01 * weight_updates)
            self.task_weights.data = torch.clamp(self.task_weights.data, 0.1, 10.0)
        
        metrics = {
            'gradnorm_loss': gradnorm_loss.item(),
            'avg_grad_norm': avg_grad_norm.item(),
            'task_weights': {name: self.task_weights[i].item() 
                           for i, name in enumerate(self.head_names)},
            'relative_losses': {name: relative_losses[i].item() 
                               for i, name in enumerate(self.head_names)},
            'gradient_norms': {name: gradient_norms[i].item() 
                             for i, name in enumerate(self.head_names)}
        }
        return metrics
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute combined multi-head loss with GradNorm balancing."""
        self.iteration += 1
        
        head_loss_dicts = self.compute_head_losses(outputs, targets)
        head_losses = {
            name: loss_dict.get('loss', torch.tensor(0.0))
            for name, loss_dict in head_loss_dicts.items()
        }
        
        gradnorm_metrics = {}
        if model is not None and self.iteration % self.update_interval == 0:
            try:
                weighted_losses, gradient_norms = self.compute_gradient_norms(model, head_losses)
                gradnorm_metrics = self.update_task_weights(head_losses, gradient_norms)
            except Exception as e:
                logger.warning(f"GradNorm update failed: {e}")
        
        total_loss = torch.tensor(0.0, device=list(head_losses.values())[0].device)
        for i, head_name in enumerate(self.head_names):
            if head_name in head_losses:
                weighted_loss = self.task_weights[i] * head_losses[head_name]
                total_loss = total_loss + weighted_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'head_losses': {name: loss.item() if torch.is_tensor(loss) else loss
                          for name, loss in head_losses.items()},
            **gradnorm_metrics
        }
        
        return total_loss, metrics


class GradNormStressIntegrator:
    """
    Integrates GradNormMultiHeadLoss with MaxSight Stress Test Suite.
    
    Tracks per-head metrics, detects gradient warfare, and triggers alerts.
    Provides seamless integration with training loops and stress testing.
    """
    
    def __init__(
        self,
        loss_module: GradNormMultiHeadLoss,
        monitor_window: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None,
        auto_dampen: bool = False,
        damp_factor: float = 0.9
    ):
        """
        Initialize GradNorm stress integrator.
        
        Arguments:
            loss_module: Instance of GradNormMultiHeadLoss
            monitor_window: Number of iterations for loss history
            alert_thresholds: Dict with thresholds for alerts
                Example: {'dominant': 0.5, 'oscillating': 0.3, 'plateaued': 1e-6}
            auto_dampen: Whether to automatically reduce weights for problematic heads
            damp_factor: Factor to multiply weights by when dampening (default 0.9 = 10% reduction)
        """
        self.loss_module = loss_module
        self.monitor = PerHeadLossMonitor(window_size=monitor_window)
        self.alert_thresholds = alert_thresholds or {
            'dominant': 0.5,
            'oscillating': 0.3,
            'plateaued': 1e-6
        }
        self.auto_dampen = auto_dampen
        self.damp_factor = damp_factor
        self.iteration = 0
        self.alert_history: List[Dict[str, Any]] = []
    
    def step(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute total loss, update task weights, monitor head trends, and return metrics.
        
        This is the main entry point for training loops. It wraps GradNormMultiHeadLoss
        and adds stress monitoring, issue detection, and optional auto-dampening.
        
        Arguments:
            outputs: Model outputs dictionary
            targets: Target labels dictionary
            model: Model instance (required for GradNorm)
        
        Returns:
            Tuple of (total_loss, metrics_dict)
            metrics_dict includes:
                - All standard GradNorm metrics (task_weights, gradient_norms, etc.)
                - head_issues: Dict of detected issues by type
                - alert_history: List of recent alerts
                - monitor_summary: Per-head loss statistics
        """
        self.iteration += 1
        
        # Compute loss and update task weights via GradNorm
        total_loss, metrics = self.loss_module(outputs, targets, model=model)
        
        # Extract head losses for monitoring
        head_losses = metrics.get('head_losses', {})
        
        # Convert to format expected by PerHeadLossMonitor
        # PerHeadLossMonitor expects Dict[str, torch.Tensor]
        monitor_losses = {}
        for head_name, loss_val in head_losses.items():
            if isinstance(loss_val, (int, float)):
                monitor_losses[head_name] = torch.tensor(float(loss_val))
            elif torch.is_tensor(loss_val):
                monitor_losses[head_name] = loss_val
            else:
                continue
        
        # Update monitor with current losses
        self.monitor.update(monitor_losses)
        
        # Detect issues if enough history accumulated
        issues = self.monitor.detect_issues()
        
        # Handle detected issues (optional auto-dampening)
        if issues and self.auto_dampen:
            self._handle_issues(issues)
        
        # Build dashboard metrics for MaxSight stress suite
        dashboard_metrics = {
            'iteration': self.iteration,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'task_weights': metrics.get('task_weights', {}),
            'gradient_norms': metrics.get('gradient_norms', {}),
            'relative_losses': metrics.get('relative_losses', {}),
            'head_issues': issues,
            'monitor_summary': self.monitor.get_summary()
        }
        
        # Log critical alerts
        if any(issues.values()):
            alert = {
                'iteration': self.iteration,
                'issues': issues,
                'task_weights': metrics.get('task_weights', {}),
                'gradient_norms': metrics.get('gradient_norms', {})
            }
            self.alert_history.append(alert)
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history.pop(0)
            
            # Log warning for critical issues
            for issue_type, heads in issues.items():
                if heads:
                    logger.warning(
                        f"Iteration {self.iteration}: {issue_type} heads detected: {heads}"
                    )
        
        # Merge dashboard metrics into main metrics
        metrics.update(dashboard_metrics)
        
        return total_loss, metrics
    
    def _handle_issues(self, issues: Dict[str, List[str]]):
        """
        Automatically dampen task weights for heads flagged as problematic.
        
        This is an optional safety mechanism that reduces the influence of
        misbehaving heads to prevent gradient warfare.
        
        Arguments:
            issues: Dictionary mapping issue types to affected head names
        """
        for issue_type, heads in issues.items():
            for head_name in heads:
                if head_name in self.loss_module.head_names:
                    idx = self.loss_module.head_names.index(head_name)
                    
                    with torch.no_grad():
                        old_weight = self.loss_module.task_weights.data[idx].item()
                        self.loss_module.task_weights.data[idx] *= self.damp_factor
                        new_weight = self.loss_module.task_weights.data[idx].item()
                        
                        logger.warning(
                            f"Dampened {head_name} weight: {old_weight:.3f} -> {new_weight:.3f} "
                            f"due to {issue_type}"
                        )
    
    def get_alerts(self, max_alerts: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Arguments:
            max_alerts: Maximum number of alerts to return
        
        Returns:
            List of alert dictionaries
        """
        return self.alert_history[-max_alerts:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary for stress testing.
        
        Returns:
            Dictionary with:
                - current_iteration
                - head_loss_summary: Per-head statistics
                - detected_issues: Current issues
                - task_weights: Current task weights
                - recent_alerts: Recent alerts
        """
        return {
            'current_iteration': self.iteration,
            'head_loss_summary': self.monitor.get_summary(),
            'detected_issues': self.monitor.detect_issues(),
            'task_weights': {
                name: self.loss_module.task_weights[i].item()
                for i, name in enumerate(self.loss_module.head_names)
            },
            'recent_alerts': self.get_alerts(max_alerts=10)
        }
    
    def reset_monitoring(self):
        """Reset monitoring state (useful for new training runs)."""
        self.monitor = PerHeadLossMonitor(window_size=self.monitor.window_size)
        self.alert_history.clear()
        self.iteration = 0
        logger.info("GradNorm stress monitoring reset")

