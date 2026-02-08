# Helpers for Hugging Face CLIP (e.g. openai/clip-vit-base-patch32).

import torch
from typing import Union, Any


def clip_image_features_to_tensor(out: Any) -> torch.Tensor:
    """Maps CLIP get_image_features() return value to a single tensor."""
    if isinstance(out, torch.Tensor):
        return out
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler
    last = getattr(out, "last_hidden_state", None)
    if last is not None:
        return last[:, 0]
    raise TypeError("CLIP image output has no pooler_output or last_hidden_state")






