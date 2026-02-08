"""Task Balancing for Multi-Head Training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GradNormBalancer(nn.Module):
    """GradNorm: Gradient normalization for adaptive loss balancing."""
    
    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,  # Restoring force hyperparameter.
        initial_task_weights: Optional[List[float]] = None
    ):
        """Initialize GradNorm balancer."""
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
        """Compute gradient norms for each task."""
        # Apply learnable task weights to balance loss magnitudes.
        weighted_losses = [
            self.task_weights[i] * loss 
            for i, loss in enumerate(task_losses)
        ]
        
        # Compute gradient norms for each task. Retain_graph=True keeps computation graph alive for multiple backward passes.
        gradient_norms = []
        for i, weighted_loss in enumerate(weighted_losses):
            model.zero_grad()
            weighted_loss.backward(retain_graph=True)
            
            # Compute L2 norm of gradients on shared parameters.
            grad_norm = 0.0
            for param in shared_params:
                if param.grad is not None:
                    grad_norm += param.grad.norm(p=2) ** 2
            grad_norm = grad_norm ** 0.5
            gradient_norms.append(grad_norm)
        
        model.zero_grad()  # Clear gradients after computing norms.
        gradient_norms_tensor = torch.stack(gradient_norms)
        return torch.stack(weighted_losses), gradient_norms_tensor
    
    def update_task_weights(
        self,
        task_losses: List[torch.Tensor],
        gradient_norms: torch.Tensor,
        iteration: int
    ) -> Dict[str, float]:
        """Update task weights using GradNorm algorithm."""
        # Initialize reference losses on first iteration.
        if not self.initialized:
            self.initial_losses = torch.stack([loss.detach() for loss in task_losses])
            self.initialized = True
        
        # Compute relative losses (current / initial) to track training progress per task.
        current_losses = torch.stack([loss.detach() for loss in task_losses])
        relative_losses = current_losses / (self.initial_losses + 1e-8)
        
        # Average gradient norm across all tasks (target for balancing)
        avg_grad_norm = gradient_norms.mean()
        
        # Relative inverse training rates: tasks with higher relative loss need more learning.
        # Alpha controls restoring force (higher = stronger rebalancing)
        relative_inverse_rates = relative_losses ** self.alpha
        
        # Target gradient norms: tasks that are behind get larger gradient targets.
        target_grad_norms = avg_grad_norm * relative_inverse_rates
        
        # GradNorm loss: minimize difference between actual and target gradient norms.
        gradnorm_loss = F.l1_loss(gradient_norms, target_grad_norms)
        
        # Update task weights via gradient descent on gradnorm_loss.
        # Run separately from the main training loop.
        # In practice, you'd optimize task_weights with respect to gradnorm_loss.
        
        # For now, return metrics for monitoring.
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
    """PCGrad: Projected Conflicting Gradients for multi-task learning."""
    
    def __init__(self):
        """Initialize PCGrad balancer."""
        pass
    
    def project_conflicting_gradients(
        self,
        gradients: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Project conflicting gradients to resolve conflicts."""
        projected_grads = []
        
        for i, grad_i in enumerate(gradients):
            grad_i_proj = grad_i.clone()
            
            # Project grad_i onto other gradients to resolve conflicts.
            for j, grad_j in enumerate(gradients):
                if i != j:
                    # Negative dot product indicates conflicting update directions.
                    dot_product = (grad_i * grad_j).sum()
                    
                    if dot_product < 0:  # Gradients point in opposite directions.
                        # Project grad_i onto plane orthogonal to grad_j (remove conflict component)
                        grad_i_proj = grad_i_proj - (dot_product / (grad_j.norm() ** 2 + 1e-8)) * grad_j
            
            projected_grads.append(grad_i_proj)
        
        return projected_grads


class PerHeadLossMonitor:
    """Monitor per-head loss magnitudes over time."""
    
    def __init__(self, window_size: int = 100):
        """Initialize loss monitor. Arguments: window_size: Number of iterations to track."""
        self.window_size = window_size
        self.loss_history: Dict[str, List[float]] = defaultdict(list)
        self.iteration = 0
    
    def update(self, head_losses: Dict[str, torch.Tensor]):
        """Update loss history. Arguments: head_losses: Dictionary mapping head names to loss values."""
        self.iteration += 1
        
        for head_name, loss in head_losses.items():
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            
            # Add to history.
            self.loss_history[head_name].append(loss_val)
            
            # Keep only window_size most recent.
            if len(self.loss_history[head_name]) > self.window_size:
                self.loss_history[head_name].pop(0)
    
    def detect_issues(self) -> Dict[str, List[str]]:
        """Detect potential gradient warfare issues. Returns: Dictionary mapping issue types to affected heads."""
        issues = {
            'dominant': [],  # Loss decaying too fast.
            'oscillating': [],  # Loss oscillating.
            'plateaued': [],  # Loss not improving.
            'conflicting': []  # Losses moving in opposite directions.
        }
        
        for head_name, history in self.loss_history.items():
            if len(history) < 20:  # Need enough history.
                continue
            
            recent = history[-20:]
            early = history[-50:-30] if len(history) >= 50 else history[:20]
            
            # Check for dominance (decaying too fast)
            if len(recent) > 0 and len(early) > 0:
                recent_avg = sum(recent) / len(recent)
                early_avg = sum(early) / len(early)
                decay_rate = (early_avg - recent_avg) / (early_avg + 1e-8)
                
                if decay_rate > 0.5:  # Decayed by >50%.
                    issues['dominant'].append(head_name)
            
            # Check for oscillation.
            if len(recent) > 10:
                variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
                mean_val = sum(recent) / len(recent)
                cv = (variance ** 0.5) / (mean_val + 1e-8)  # Coefficient of variation.
                
                if cv > 0.3:  # High variance relative to mean.
                    issues['oscillating'].append(head_name)
            
            # Check for plateau.
            if len(recent) > 10:
                recent_trend = (recent[-1] - recent[0]) / (len(recent) - 1)
                if abs(recent_trend) < 1e-6:  # No improvement.
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
    """Multi-head loss combiner with GradNorm for adaptive task balancing."""
    
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
        self._nan_warned_heads: set = set()  # Debounce nan/inf warnings per head.
    
    def set_shared_params(self, shared_params: List[nn.Parameter]):
        """Set shared parameters for gradient norm computation."""
        self.shared_params = shared_params
    
    OUTPUT_KEY_MAP = {
        'objectness': 'objectness',
        'classification': 'classifications',
        'box': 'boxes',
        'distance': 'distance_zones',
        'urgency': 'urgency_scores',
    }
    TARGET_KEY_MAP = {
        'objectness': 'objectness',  # Batch may not have; build from labels or skip.
        'classification': 'labels',
        'box': 'boxes',
        'distance': 'distance',
        'urgency': 'urgency',
    }

    def compute_head_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for all heads. Uses Hungarian matching to align detection pred/targets, then per-head losses."""
        head_loss_dicts = {}
        device = next((t.device for t in outputs.values() if torch.is_tensor(t)), torch.device('cpu'))
        aligned_pred, aligned_target = None, None
        if ('boxes' in outputs and 'classifications' in outputs and
                'boxes' in targets and 'labels' in targets and
                outputs['boxes'].dim() == 3 and targets['boxes'].dim() == 3):
            try:
                from ml.training.matching import build_matched_pred_targets
                aligned_pred, aligned_target = build_matched_pred_targets(outputs, targets)
            except Exception as e:
                logger.warning(f"Matching failed, using direct keys: {e}")

        for head_name, loss_fn in self.head_losses.items():
            try:
                pred, targ = None, None
                if head_name == 'objectness' and aligned_pred is not None:
                    pred = aligned_pred.get('objectness')
                    targ = aligned_target.get('objectness')
                elif head_name == 'classification' and aligned_pred is not None:
                    pred = aligned_pred.get('classification')
                    targ = aligned_target.get('labels')
                    if pred is not None and pred.numel() == 0:
                        # Create zero loss with requires_grad=True for GradNorm compatibility.
                        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        head_loss_dicts[head_name] = {'loss': zero_loss}
                        continue
                    # ClassificationLoss expects [B, N, C] and [B, N]; matched are [N, C] and [N].
                    if pred is not None and targ is not None and pred.dim() == 2:
                        pred = pred.unsqueeze(0)
                        targ = targ.unsqueeze(0)
                elif head_name == 'box' and aligned_pred is not None:
                    pred = aligned_pred.get('box')
                    targ = aligned_target.get('boxes')
                    if pred is not None and pred.numel() == 0:
                        # Create zero loss with requires_grad=True for GradNorm compatibility.
                        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        head_loss_dicts[head_name] = {'loss': zero_loss}
                        continue
                elif head_name == 'distance' and aligned_pred is not None:
                    pred = aligned_pred.get('distance')
                    targ = aligned_target.get('distance')
                    if pred is not None and pred.numel() == 0:
                        # Create zero loss with requires_grad=True for GradNorm compatibility.
                        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        head_loss_dicts[head_name] = {'loss': zero_loss}
                        continue
                elif head_name == 'urgency':
                    pred = outputs.get('urgency_scores')
                    targ = targets.get('urgency')

                if pred is None or targ is None:
                    out_key = self.OUTPUT_KEY_MAP.get(head_name, head_name)
                    targ_key = self.TARGET_KEY_MAP.get(head_name, head_name)
                    pred = outputs.get(out_key) if isinstance(outputs.get(out_key), torch.Tensor) else None
                    targ = targets.get(targ_key) if isinstance(targets.get(targ_key), torch.Tensor) else None
                if pred is None or targ is None:
                    # Create zero loss with requires_grad=True for GradNorm compatibility.
                    zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    head_loss_dicts[head_name] = {'loss': zero_loss}
                    continue
                # ObjectnessLoss expects logits; model may return sigmoid scores.
                if head_name == 'objectness' and pred.dtype == torch.float32 and pred.min() >= 0 and pred.max() <= 1:
                    pred = torch.logit(torch.clamp(pred, 1e-4, 1.0 - 1e-4))
                loss = loss_fn(pred, targ)
                if torch.is_tensor(loss):
                    head_loss_dicts[head_name] = {'loss': loss}
                elif isinstance(loss, dict) and 'loss' in loss:
                    head_loss_dicts[head_name] = loss
                else:
                    # Create zero loss with requires_grad=True for GradNorm compatibility.
                    zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    head_loss_dicts[head_name] = {'loss': zero_loss}
            except Exception as e:
                logger.warning(f"Failed to compute loss for {head_name}: {e}")
                # Create zero loss with requires_grad=True for GradNorm compatibility.
                zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                head_loss_dicts[head_name] = {'loss': zero_loss}
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
            # Ensure loss requires grad for GradNorm computation.
            if torch.is_tensor(loss) and not loss.requires_grad:
                loss = loss.detach().requires_grad_(True)
            weighted_loss = self.task_weights[i] * loss
            weighted_losses.append(weighted_loss)
        
        weighted_losses_tensor = torch.stack(weighted_losses)
        gradient_norms = []
        
        for i, weighted_loss in enumerate(weighted_losses):
            # Ensure weighted_loss requires grad before backward pass.
            if not weighted_loss.requires_grad:
                logger.warning(f"Head {self.head_names[i]} loss does not require grad, skipping gradient norm computation")
                gradient_norms.append(0.0)
                continue
            
            model.zero_grad()
            if weighted_loss.dtype == torch.float16:
                weighted_loss = weighted_loss.clone().float()
            try:
                weighted_loss.backward(retain_graph=True)
            except RuntimeError as e:
                if "inplace operation" in str(e) or "version" in str(e):
                    logger.warning(
                        f"GradNorm backward failed for head {self.head_names[i]} due to inplace operation: {e}. "
                        "This may indicate mixed precision or inplace ops in the model. Skipping this head."
                    )
                    gradient_norms.append(0.0)
                    continue
                else:
                    raise
            
            grad_norm = 0.0
            for param in self.shared_params:
                if param.grad is not None:
                    # Use detach() to avoid inplace operation issues.
                    grad_norm += param.grad.detach().norm(p=2) ** 2
            grad_norm = grad_norm ** 0.5
            if torch.is_tensor(grad_norm):
                grad_norm = grad_norm.item()
            gradient_norms.append(grad_norm)
        
        model.zero_grad()
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
        """Update task weights using GradNorm algorithm. On MPS, GradNorm math runs on CPU to avoid unsupported ops; weights are then copied back."""
        orig_device = self.task_weights.device
        use_cpu_fallback = orig_device.type == 'mps'
        if use_cpu_fallback:
            gradient_norms = gradient_norms.detach().cpu()

        if not self.initialized:
            head_loss_list = [head_losses[name] for name in self.head_names]
            self.initial_losses = torch.stack([
                loss.detach() if torch.is_tensor(loss) else torch.tensor(loss)
                for loss in head_loss_list
            ])
            if use_cpu_fallback:
                self.initial_losses = self.initial_losses.cpu()
            self.initial_losses = self.initial_losses.to(gradient_norms.device)
            self.initialized = True
            return {}

        head_loss_list = [head_losses[name] for name in self.head_names]
        current_losses = torch.stack([
            loss.detach() if torch.is_tensor(loss) else torch.tensor(loss)
            for loss in head_loss_list
        ])
        current_losses = current_losses.to(gradient_norms.device)
        init_losses = self.initial_losses.to(gradient_norms.device)

        relative_losses = current_losses / (init_losses + 1e-8)
        avg_grad_norm = gradient_norms.mean()
        relative_inverse_rates = relative_losses ** self.alpha
        target_grad_norms = avg_grad_norm * relative_inverse_rates
        gradnorm_loss = F.l1_loss(gradient_norms, target_grad_norms)

        weight_updates = (target_grad_norms - gradient_norms) / (gradient_norms + 1e-8)
        weight_updates = torch.clamp(weight_updates, -0.1, 0.1)

        with torch.no_grad():
            weights = self.task_weights.data.to(gradient_norms.device)
            weights = weights * (1.0 + 0.01 * weight_updates)
            weights = torch.clamp(weights, 0.1, 10.0)
            self.task_weights.data = weights.to(orig_device)
            if use_cpu_fallback:
                self.initial_losses = init_losses.to(orig_device)

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
        device = next((t.device for t in outputs.values() if torch.is_tensor(t)), torch.device('cpu'))
        head_losses = {}
        for name, loss_dict in head_loss_dicts.items():
            loss = loss_dict.get('loss', None)
            if loss is None:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            elif torch.is_tensor(loss) and not loss.requires_grad:
                # If loss doesn't require grad, create a new tensor that does.
                loss = loss.detach().requires_grad_(True)
            elif not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), device=device, requires_grad=True)
            head_losses[name] = loss
        
        for name in list(head_losses.keys()):
            l = head_losses[name]
            if torch.is_tensor(l) and (torch.isnan(l) | torch.isinf(l)).any().item():
                if name not in self._nan_warned_heads:
                    logger.warning(
                        "GradNorm: head %r produced nan/inf loss, using 0 for this step (check data or loss for this head)",
                        name,
                    )
                    self._nan_warned_heads.add(name)
                head_losses[name] = torch.tensor(0.0, device=device, requires_grad=True)
        
        gradnorm_metrics = {}
        if model is not None and self.iteration % self.update_interval == 0:
            try:
                weighted_losses, gradient_norms = self.compute_gradient_norms(model, head_losses)
                gradnorm_metrics = self.update_task_weights(head_losses, gradient_norms)
            except Exception as e:
                logger.warning(f"GradNorm update failed: {e}")
        
        total_loss = torch.tensor(0.0, device=device)
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
    """Integrates GradNormMultiHeadLoss with MaxSight Stress Test Suite."""
    
    def __init__(
        self,
        loss_module: GradNormMultiHeadLoss,
        monitor_window: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None,
        auto_dampen: bool = False,
        damp_factor: float = 0.9
    ):
        """Initialize GradNorm stress integrator."""
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
        """Compute total loss, update task weights, monitor head trends, and return metrics."""
        self.iteration += 1
        
        # Compute loss and update task weights via GradNorm.
        total_loss, metrics = self.loss_module(outputs, targets, model=model)
        
        # Extract head losses for monitoring.
        head_losses = metrics.get('head_losses', {})
        
        # Convert to format expected by PerHeadLossMonitor. PerHeadLossMonitor expects Dict[str, torch.Tensor].
        monitor_losses = {}
        for head_name, loss_val in head_losses.items():
            if isinstance(loss_val, (int, float)):
                monitor_losses[head_name] = torch.tensor(float(loss_val))
            elif torch.is_tensor(loss_val):
                monitor_losses[head_name] = loss_val
            else:
                continue
        
        # Update monitor with current losses.
        self.monitor.update(monitor_losses)
        
        # Detect issues if enough history accumulated.
        issues = self.monitor.detect_issues()
        
        # Handle detected issues (optional auto-dampening)
        if issues and self.auto_dampen:
            self._handle_issues(issues)
        
        # Build dashboard metrics for MaxSight stress suite.
        dashboard_metrics = {
            'iteration': self.iteration,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'task_weights': metrics.get('task_weights', {}),
            'gradient_norms': metrics.get('gradient_norms', {}),
            'relative_losses': metrics.get('relative_losses', {}),
            'head_issues': issues,
            'monitor_summary': self.monitor.get_summary()
        }
        
        # Log critical alerts.
        if any(issues.values()):
            alert = {
                'iteration': self.iteration,
                'issues': issues,
                'task_weights': metrics.get('task_weights', {}),
                'gradient_norms': metrics.get('gradient_norms', {})
            }
            self.alert_history.append(alert)
            # Keep only last 100 alerts.
            if len(self.alert_history) > 100:
                self.alert_history.pop(0)
            
            # Log warning for critical issues.
            for issue_type, heads in issues.items():
                if heads:
                    logger.warning(
                        f"Iteration {self.iteration}: {issue_type} heads detected: {heads}"
                    )
        
        # Merge dashboard metrics into main metrics.
        metrics.update(dashboard_metrics)
        
        return total_loss, metrics
    
    def _handle_issues(self, issues: Dict[str, List[str]]):
        """Automatically dampen task weights for heads flagged as problematic."""
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
        """Get recent alerts. Arguments: max_alerts: Maximum number of alerts to return Returns: List of alert dictionaries."""
        return self.alert_history[-max_alerts:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for stress testing."""
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







