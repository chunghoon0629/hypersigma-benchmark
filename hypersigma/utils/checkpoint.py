"""Checkpoint loading utilities for HyperSIGMA."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    prefix: str = "",
    strict: bool = False,
    remove_keys: Optional[list] = None,
) -> Dict:
    """
    Load pretrained weights into a model.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to the checkpoint file.
        prefix: Prefix to add to state dict keys (e.g., "spat_encoder.").
        strict: Whether to require an exact match.
        remove_keys: List of key patterns to remove from state dict.

    Returns:
        Dictionary containing loading information.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint with weights_only=False for pickle compatibility
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Strip module prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Remove specified keys
    if remove_keys:
        for pattern in remove_keys:
            for k in list(state_dict.keys()):
                if pattern in k:
                    del state_dict[k]

    # Add prefix if specified
    if prefix:
        state_dict = {prefix + k: v for k, v in state_dict.items()}

    # Get model state dict
    model_dict = model.state_dict()

    # Filter to only matching keys
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Update model dict
    model_dict.update(filtered_dict)

    # Load into model
    msg = model.load_state_dict(model_dict, strict=strict)

    return {
        'loaded_keys': list(filtered_dict.keys()),
        'missing_keys': msg.missing_keys if hasattr(msg, 'missing_keys') else [],
        'unexpected_keys': msg.unexpected_keys if hasattr(msg, 'unexpected_keys') else [],
    }


def load_hypersigma_weights(
    model: nn.Module,
    spat_weights: str,
    spec_weights: str,
    strict: bool = False,
) -> None:
    """
    Load both spatial and spectral weights for HyperSIGMA model.

    Args:
        model: The HyperSIGMA model with spat_encoder and spec_encoder.
        spat_weights: Path to spatial encoder weights.
        spec_weights: Path to spectral encoder weights.
        strict: Whether to require exact match.
    """
    # Load spatial weights
    spat_info = load_pretrained_weights(
        model,
        spat_weights,
        prefix='spat_encoder.',
        strict=False,
        remove_keys=['patch_embed.proj', 'spat_map', 'pos_embed']
    )
    print(f"Loaded {len(spat_info['loaded_keys'])} spatial encoder weights")

    # Load spectral weights
    spec_info = load_pretrained_weights(
        model,
        spec_weights,
        prefix='spec_encoder.',
        strict=False,
        remove_keys=['patch_embed.proj', 'spat_map', 'fpn1.0.weight', 'patch_embed']
    )
    print(f"Loaded {len(spec_info['loaded_keys'])} spectral encoder weights")
