"""Loading models with TransformerLens 4 (TransformerBridge)."""

import torch
from transformer_lens.model_bridge import TransformerBridge


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str | None = None) -> TransformerBridge:
    """Raw HF weights wrapped with TL hook names (blocks.{i}.hook_resid_post, ...), in eval mode."""
    model = TransformerBridge.boot_transformers(name, device=device or pick_device())
    model.eval()
    return model
