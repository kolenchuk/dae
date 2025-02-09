import torch
from typing import Dict, Any, Tuple, Optional
import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def save_model_weights(
        model: torch.nn.Module,
        save_dir: str,
) -> str:
    """
    Save only the model weights.

    Args:
        model: The model to save
        save_dir: Directory to save the weights

    Returns:
        Path to the saved weights file
    """
    os.makedirs(save_dir, exist_ok=True)

    file_name = "weights.pth"
    weights_path = os.path.join(save_dir, file_name)

    torch.save(model.state_dict(), weights_path)

    return file_name


def save_model_config(
        model: torch.nn.Module,
        save_dir: str,
        additional_config: Optional[Dict] = None
) -> str:
    """
    Save model configuration separately.

    Args:
        model: The model to extract configuration from
        save_dir: Directory to save the config
        additional_config: Optional additional configuration parameters

    Returns:
        Path to the saved config file
    """
    os.makedirs(save_dir, exist_ok=True)
    file_name = "config.json"
    config_path = os.path.join(save_dir, file_name)

    # Extract model configuration
    config = {
        'vocab_size': getattr(model, 'vocab_size',
                              model.embedding.num_embeddings if hasattr(model, 'embedding') else None),
        'd_model': getattr(model, 'd_model', None),
        'num_heads': getattr(model.layers[0].self_attn, 'num_heads', None)
        if hasattr(model, 'layers') else None,
        'num_layers': len(model.layers) if hasattr(model, 'layers') else None,
        'd_ff': getattr(model.layers[0].feed_forward.linear1, 'out_features', None)
        if hasattr(model, 'layers') and hasattr(model.layers[0], 'feed_forward') else None,
        'dropout': getattr(model.layers[0].dropout, 'p', None)
        if hasattr(model, 'layers') else None,
        'max_len': model.pos_encoding.pe.shape[1] if hasattr(model, 'pos_encoding') else None,
        'pad_idx': getattr(model, 'pad_idx', None)
    }

    if additional_config:
        config.update(additional_config)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return file_name


def save_metadata(
        save_dir: str,
        char_to_idx: Optional[Dict],
        idx_to_char: Optional[Dict],
        additional_info: Optional[Dict] = None
) -> str:
    """
    Save model metadata separately.

    Args:
        save_dir: Directory to save metadata
        char_to_idx: character to index mapping
        idx_to_char: index to character mapping
        additional_info: Optional additional metadata

    Returns:
        Path to the saved metadata file
    """
    os.makedirs(save_dir, exist_ok=True)
    file_name = "metadata.json"
    metadata_path = os.path.join(save_dir, file_name)

    metadata = {
        'save_dir': save_dir,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }

    if additional_info:
        metadata.update(additional_info)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    return file_name


def save_checkpoint_distributed(
        model: torch.nn.Module,
        save_dir: str,
        char_to_idx: Optional[Dict],
        idx_to_char: Optional[Dict],
        metrics: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Save model checkpoint with distributed file structure.

    Args:
        model: The model to save
        save_dir: Base directory for saving
        char_to_idx: character to index mapping
        idx_to_char: index to character mapping
        metrics: Optional training metrics
        additional_info: Optional additional information

    Returns:
        Dictionary with paths to all saved files
    """
    save_dir = Path(save_dir)

    # Save each component separately
    paths = {
        'weights': save_model_weights(model, str(save_dir)),
        'config': save_model_config(model, str(save_dir)),
        'metadata': save_metadata(str(save_dir), char_to_idx, idx_to_char, additional_info)
    }

    # Create a manifest file
    manifest = {
        'file_paths': paths,
    }

    manifest_path = save_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    return paths


def load_checkpoint_distributed(
        save_dir: str,
        model_class: type,
        device: str = "cuda"
) -> Tuple[Any, Any]:
    """
    Load model checkpoint from distributed file structure.

    Args:
        save_dir: Base directory containing the checkpoint
        model_class: The model class to instantiate
        device: Device to load the model to

    Returns:
        Tuple of (model, metadata)
    """
    save_dir = Path(save_dir)

    manifest_path = save_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(save_dir / manifest['file_paths']['config']) as f:
        config = json.load(f)

    model = model_class(**config).to(device)
    state_dict = torch.load(save_dir / manifest['file_paths']['weights'], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    with open(save_dir / manifest['file_paths']['metadata']) as f:
        metadata = json.load(f)

    return model, metadata