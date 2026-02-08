"""Personal Mode for Phase 6: Active Scene Exploration & Personalization Enhances MaxSight with user-specific adaptations and active exploration."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from ml.retrieval.fusion.meta_fusion import (
    MetaFusionWeights,
    ActiveSceneExploration,
    PredictiveNavigationGuidance,
    UserProfile
)


@dataclass
class PersonalizationState:
    """State for personalization system."""
    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)
    task_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    adaptation_count: int = 0


class PersonalMode:
    """Personal mode manager for Phase 6. Handles: - User preference learning - Active scene exploration - Predictive navigation guidance - Adaptive fusion weights."""
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_modalities: int = 3
    ):
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Initialize components.
        self.meta_fusion = MetaFusionWeights(
            num_modalities=num_modalities,
            embed_dim=embed_dim
        )
        
        self.active_exploration = ActiveSceneExploration(embed_dim=embed_dim)
        
        self.navigation_guidance = PredictiveNavigationGuidance(
            embed_dim=embed_dim
        )
        
        # User state tracking.
        self.user_states: Dict[str, PersonalizationState] = {}
        
    def get_user_state(self, user_id: str) -> PersonalizationState:
        """Get or create user state."""
        if user_id not in self.user_states:
            self.user_states[user_id] = PersonalizationState(user_id=user_id)
        return self.user_states[user_id]
    
    def update_preferences(
        self,
        user_id: str,
        task_type: str,
        performance_score: float,
        preferred_modalities: Optional[List[str]] = None
    ):
        """Update user preferences based on task performance."""
        state = self.get_user_state(user_id)
        
        # Update task history.
        state.task_history.append(task_type)
        if len(state.task_history) > 100:  # Keep last 100 tasks.
            state.task_history.pop(0)
        
        # Update performance metrics.
        if task_type not in state.performance_metrics:
            state.performance_metrics[task_type] = []
        state.performance_metrics[task_type].append(performance_score)
        
        # Update preferences (exponential moving average)
        if task_type not in state.preferences:
            state.preferences[task_type] = performance_score
        else:
            alpha = 0.1  # Learning rate.
            state.preferences[task_type] = (
                alpha * performance_score + (1 - alpha) * state.preferences[task_type]
            )
        
        # Adapt fusion weights.
        if state.adaptation_count % 10 == 0:  # Adapt every 10 tasks.
            user_profile = UserProfile(
                user_id=user_id,
                preferred_modalities=preferred_modalities or ['vision', 'audio'],
                task_preferences=state.preferences
            )
            adapted_weights = self.meta_fusion.adapt_to_user(
                user_profile,
                state.performance_metrics
            )
            # Store adapted weights (could be used in next forward pass)
        
        state.adaptation_count += 1
    
    def fuse_with_personalization(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
        urgency: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse modalities with personalized weights."""
        # Convert user_id to tensor if provided.
        user_id_tensor = None
        if user_id is not None:
            # Simple hash-based user ID (in production, use proper user ID mapping)
            user_hash = hash(user_id) % 1000
            user_id_tensor = torch.tensor([user_hash], device=next(iter(modality_embeddings.values())).device)
        
        return self.meta_fusion(
            modality_embeddings=modality_embeddings,
            user_id=user_id_tensor,
            task_type=task_type,
            urgency=urgency,
            confidence=confidence
        )
    
    def explore_scene(
        self,
        region_embeddings: torch.Tensor,
        uncertainties: torch.Tensor,
        user_id: Optional[str] = None,
        urgency: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine which regions to explore next. Returns: exploration_scores: Scores for each region selected_regions: Indices of regions to explore."""
        # Get user preference if available.
        user_preference = None
        if user_id:
            state = self.get_user_state(user_id)
            # Use average preference as exploration bias.
            if state.preferences:
                user_preference = sum(state.preferences.values()) / len(state.preferences)
        
        urgency_tensor = None
        if urgency is not None:
            urgency_tensor = torch.tensor([urgency], device=region_embeddings.device)
        
        return self.active_exploration(
            region_embeddings=region_embeddings,
            uncertainties=uncertainties,
            urgency=urgency_tensor,
            user_preference=user_preference
        )
    
    def predict_navigation(
        self,
        current_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
        scene_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict navigation path and generate guidance. Returns: Dictionary with navigation predictions."""
        return self.navigation_guidance(
            current_embedding=current_embedding,
            goal_embedding=goal_embedding,
            scene_context=scene_context
        )
    
    def get_personalized_outputs(
        self,
        model_outputs: Dict[str, torch.Tensor],
        user_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply personalization to model outputs."""
        personalized = model_outputs.copy()
        
        if user_id:
            state = self.get_user_state(user_id)
            
            # Adjust output verbosity based on preferences.
            if task_type in state.preferences:
                preference = state.preferences[task_type]
                
                # Scale outputs based on preference. Higher preference = more detailed outputs.
                scale_factor = 0.5 + preference  # [0.5, 1.5].
                
                # Apply to relevant outputs.
                if 'scene_description' in personalized:
                    # Could adjust description length, etc.
                    pass
        
        return personalized







