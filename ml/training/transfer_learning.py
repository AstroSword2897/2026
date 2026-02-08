"""Tier Transfer Learning for MaxSight."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class TierTransferManager:
    """Manages tier-to-tier transfer learning."""
    
    def __init__(
        self,
        source_checkpoint: Path,
        target_model: nn.Module,
        transfer_config: Dict[str, Any]
    ):
        """Initialize transfer manager."""
        self.source_checkpoint = Path(source_checkpoint)
        self.target_model = target_model
        self.config = transfer_config
        
        # Validates source checkpoint exists.
        if not self.source_checkpoint.exists():
            raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
    
    def validate_source_checkpoint(self) -> bool:
        """Validate T2 checkpoint meets transfer prerequisites. Returns: True if checkpoint is valid for transfer."""
        logger.info("Validating source checkpoint...")
        
        checkpoint = torch.load(self.source_checkpoint, map_location='cpu')
        
        # Checks required keys.
        required_keys = ['model_state_dict', 'epoch', 'val_loss']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            logger.error(f"Missing required keys: {missing_keys}")
            return False
        
        # Check epoch count (expect trained model)
        epoch = checkpoint.get('epoch', 0)
        if epoch < 50:
            logger.warning(f"Source checkpoint only at epoch {epoch}, recommend ≥50")
        
        # Checks for NaNs in state dict.
        state_dict = checkpoint['model_state_dict']
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                logger.error(f"NaN found in {name}")
                return False
        
        logger.info("OK Source checkpoint validated")
        return True
    
    def transfer_weights(
        self,
        strict: bool = False
    ) -> Dict[str, int]:
        """Transfer compatible weights from T2 to T5."""
        logger.info("Loading source checkpoint...")
        source_ckpt = torch.load(self.source_checkpoint, map_location='cpu')
        source_state = source_ckpt['model_state_dict']
        
        target_state = self.target_model.state_dict()
        
        # Components to transfer (spatial representation only)
        transfer_patterns = [
            'backbone',  # CNN backbone.
            'fpn',  # FPN.
            'vit',  # ViT blocks.
            'se_attention',  # SE attention.
            'cbam_attention',  # CBAM attention.
            'dynamic_conv',  # Dynamic convolution.
            'detection_head',  # Detection head.
            'box_head',  # Box regression head.
            'classification_head',  # Classification head.
            'distance_head',  # Distance head.
            'urgency_head',  # Urgency head.
        ]
        
        # Components to skip (coordination, not representation)
        skip_patterns = [
            'temporal',  # Temporal modeling.
            'cross_task_attention',  # Cross-task attention.
            'cross_modal_attention',  # Cross-modal attention.
            'retrieval',  # Retrieval modules.
            'scene_graph',  # Scene graph (new in T5)
            'ocr_head',  # OCR (new in T5)
            'sound_event_head',  # Sound events (new in T5)
            'personalization_head',  # Personalization (new in T5)
            'predictive_alert_head',  # Predictive alerts (new in T5)
        ]
        
        transferred = 0
        skipped = 0
        shape_mismatch = 0
        
        for target_name, target_param in target_state.items():
            # Detect whether the parameter is transferred from the source model.
            should_transfer = any(pattern in target_name for pattern in transfer_patterns)
            should_skip = any(pattern in target_name for pattern in skip_patterns)
            
            if should_skip:
                skipped += 1
                continue
            
            if should_transfer:
                # Find matching source parameter.
                source_name = target_name
                if source_name in source_state:
                    source_param = source_state[source_name]
                    
                    # Checks shape compatibility.
                    if source_param.shape == target_param.shape:
                        target_state[target_name] = source_param.clone()
                        transferred += 1
                        logger.debug(f"Transferred: {target_name}")
                    else:
                        shape_mismatch += 1
                        logger.warning(
                            f"Shape mismatch: {target_name} "
                            f"source={source_param.shape} target={target_param.shape}"
                        )
                elif strict:
                    raise KeyError(f"Required key not found in source: {source_name}")
        
        # Load updated state dict.
        self.target_model.load_state_dict(target_state, strict=False)
        
        stats = {
            'transferred': transferred,
            'skipped': skipped,
            'shape_mismatch': shape_mismatch,
            'total_target': len(target_state)
        }
        
        logger.info(f"Transfer complete: {transferred} transferred, {skipped} skipped, "
                   f"{shape_mismatch} shape mismatches")
        
        return stats
    
    def create_parameter_groups(
        self,
        base_lr: float
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with different learning rates."""
        param_groups = []
        
        # Group 1: CNN backbone (lowest LR)
        cnn_params = []
        cnn_lr = base_lr * 0.2  # More conservative.
        
        # Group 2: ViT backbone.
        vit_params = []
        vit_lr = base_lr * 0.5  # Needs more plasticity for temporal.
        
        # Group 3: Detection/box heads.
        detection_params = []
        detection_lr = base_lr * 0.6
        
        # Group 4: Temporal modules (full LR)
        temporal_params = []
        temporal_lr = base_lr * 1.0
        
        # Group 5: Cross-task/modal attention (full LR)
        cross_attention_params = []
        cross_attention_lr = base_lr * 1.0
        
        # Group 6: New Tier-5 heads (highest LR)
        new_head_params = []
        new_head_lr = base_lr * 1.3  # Must move fastest or lag permanently.
        
        # Categorize parameters.
        for name, param in self.target_model.named_parameters():
            if 'backbone' in name and 'vit' not in name:
                cnn_params.append(param)
            elif 'vit' in name or 'transformer' in name:
                vit_params.append(param)
            elif any(x in name for x in ['detection_head', 'box_head', 'classification_head']):
                detection_params.append(param)
            elif 'temporal' in name:
                temporal_params.append(param)
            elif any(x in name for x in ['cross_task_attention', 'cross_modal_attention']):
                cross_attention_params.append(param)
            elif any(x in name for x in [
                'scene_graph', 'ocr_head', 'sound_event_head',
                'personalization_head', 'predictive_alert_head'
            ]):
                new_head_params.append(param)
            else:
                # Default: put in detection group.
                detection_params.append(param)
        
        # Create parameter groups.
        if cnn_params:
            param_groups.append({'params': cnn_params, 'lr': cnn_lr, 'name': 'cnn_backbone'})
        if vit_params:
            param_groups.append({'params': vit_params, 'lr': vit_lr, 'name': 'vit_backbone'})
        if detection_params:
            param_groups.append({'params': detection_params, 'lr': detection_lr, 'name': 'detection_heads'})
        if temporal_params:
            param_groups.append({'params': temporal_params, 'lr': temporal_lr, 'name': 'temporal'})
        if cross_attention_params:
            param_groups.append({'params': cross_attention_params, 'lr': cross_attention_lr, 'name': 'cross_attention'})
        if new_head_params:
            param_groups.append({'params': new_head_params, 'lr': new_head_lr, 'name': 'new_heads'})
        
        logger.info(f"Created {len(param_groups)} parameter groups:")
        for group in param_groups:
            logger.info(f"  {group['name']}: {len(group['params'])} params, LR={group['lr']:.2e}")
        
        return param_groups
    
    def get_freeze_schedule(self, epoch: int) -> Dict[str, bool]:
        """Get freeze/unfreeze schedule based on epoch (corrected timing)."""
        freeze_map = {}
        
        for name, param in self.target_model.named_parameters():
            # Detect new T5 head (not present in T2).
            is_new_head = any(x in name for x in [
                'temporal', 'cross_task_attention', 'cross_modal_attention',
                'scene_graph', 'ocr_head', 'sound_event_head',
                'personalization_head', 'predictive_alert_head'
            ])
            
            if epoch < 5:
                # Epochs 0-5: Freeze CNN+ViT+detection, train new heads only.
                should_freeze = not is_new_head and any(x in name for x in [
                    'backbone', 'vit', 'transformer',
                    'detection_head', 'box_head', 'classification_head'
                ])
            elif epoch < 15:
                # Epochs 5-15: Unfreeze detection + classification. Still freeze: CNN+ViT backbone.
                if is_new_head:
                    should_freeze = False
                elif any(x in name for x in ['detection_head', 'classification_head']):
                    should_freeze = False
                else:
                    should_freeze = any(x in name for x in [
                        'backbone', 'vit', 'transformer'
                    ])
            elif epoch < 30:
                # Epochs 15-30: Unfreeze top 40% of ViT. Still freeze: CNN backbone, early ViT layers.
                if is_new_head:
                    should_freeze = False
                elif 'backbone' in name and 'vit' not in name:
                    should_freeze = True
                elif 'vit' in name or 'transformer' in name:
                    # Freeze first 60% of ViT layers (heuristic: layer 0-6 out of 12)
                    layer_num = self._extract_layer_number(name)
                    should_freeze = layer_num is not None and layer_num < 7
                else:
                    should_freeze = False
            elif epoch < 45:
                # Epochs 30-45: Unfreeze full ViT. Still freeze: CNN backbone.
                if is_new_head:
                    should_freeze = False
                elif 'backbone' in name and 'vit' not in name:
                    should_freeze = True
                else:
                    should_freeze = False
            else:
                # Epoch 45+: Unfreeze everything (including CNN)
                should_freeze = False
            
            freeze_map[name] = should_freeze
        
        return freeze_map
    
    def _extract_layer_number(self, name: str) -> Optional[int]:
        """Extract layer number from parameter name (heuristic)."""
        import re
        match = re.search(r'layer[_\s]*(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'blocks[_\s]*\[?(\d+)', name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def apply_freeze_schedule(self, epoch: int):
        """Apply freeze schedule for current epoch."""
        freeze_map = self.get_freeze_schedule(epoch)
        
        for name, param in self.target_model.named_parameters():
            param.requires_grad = not freeze_map.get(name, False)
        
        frozen_count = sum(1 for p in self.target_model.parameters() if not p.requires_grad)
        total_count = sum(1 for _ in self.target_model.parameters())
        
        logger.info(f"Epoch {epoch}: {frozen_count}/{total_count} parameters frozen")
    
    def get_loss_unlock_schedule(self, epoch: int) -> Dict[str, bool]:
        """Get loss unlock schedule based on epoch (aligned with representation readiness)."""
        if epoch < 10:
            # Phase 1: Detection only.
            return {
                'detection': True,
                'classification': True,
                'box_regression': True,
                'distance': False,
                'urgency': False,
                'motion': False,
                'therapy_state': False,
                'roi_priority': False,
                'navigation_difficulty': False,
                'scene_description': False,
                'ocr': False,
                'scene_graph': False,
                'sound_events': False,
                'personalization': False,
                'predictive_alerts': False,
            }
        elif epoch < 25:
            # Phase 2: + Navigation.
            return {
                'detection': True,
                'classification': True,
                'box_regression': True,
                'distance': True,
                'urgency': True,
                'motion': True,
                'roi_priority': True,
                'navigation_difficulty': True,
                'therapy_state': False,
                'scene_description': False,
                'ocr': False,
                'scene_graph': False,
                'sound_events': False,
                'personalization': False,
                'predictive_alerts': False,
            }
        elif epoch < 40:
            # Phase 3: + Therapy/urgency.
            return {
                'detection': True,
                'classification': True,
                'box_regression': True,
                'distance': True,
                'urgency': True,
                'motion': True,
                'therapy_state': True,
                'roi_priority': True,
                'navigation_difficulty': True,
                'scene_description': False,
                'ocr': False,
                'scene_graph': False,
                'sound_events': False,
                'personalization': False,
                'predictive_alerts': False,
            }
        else:
            # Phase 4: All tasks enabled.
            return {
                'detection': True,
                'classification': True,
                'box_regression': True,
                'distance': True,
                'urgency': True,
                'motion': True,
                'therapy_state': True,
                'roi_priority': True,
                'navigation_difficulty': True,
                'scene_description': True,
                'ocr': True,
                'scene_graph': True,
                'sound_events': True,
                'personalization': True,
                'predictive_alerts': True,
            }


def create_transfer_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.05
) -> torch.optim.AdamW:
    """Create optimizer with parameter-grouped learning rates for transfer."""
    transfer_mgr = TierTransferManager(
        source_checkpoint=Path("dummy"),  # Not used for optimizer creation.
        target_model=model,
        transfer_config={}
    )
    
    param_groups = transfer_mgr.create_parameter_groups(base_lr)
    
    # Add weight decay to all groups.
    for group in param_groups:
        group['weight_decay'] = weight_decay
    
    optimizer = torch.optim.AdamW(param_groups)
    
    return optimizer







