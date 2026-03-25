"""Model loading utilities for vision roundtrip experiments.

Handles loading VLM components (ViT, mmproj, full model) from
HuggingFace format models.
"""

import torch
from typing import Tuple, Optional
from pathlib import Path


def load_full_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple:
    """
    Load the full VLM for description generation and LLM inversion.

    Returns: (model, processor)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"  Type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    if hasattr(model.config, 'vision_config'):
        vc = model.config.vision_config
        print(f"  Vision: {vc.get('depth', '?')} layers, "
              f"hidden={vc.get('hidden_size', '?')}, "
              f"patch={vc.get('patch_size', '?')}, "
              f"out={vc.get('out_hidden_size', '?')}")

    return model, processor


def extract_vision_components(model) -> dict:
    """
    Extract the vision encoder and projection components from a loaded VLM.

    Returns dict with:
        visual_model: the ViT encoder
        mmproj_weight: projection weight matrix (or None if integrated)
        mmproj_bias: projection bias (or None)
        embed_table: the text embedding table
        image_token_id: the token ID used for image placeholders
        hidden_size: LLM hidden dimension
        vision_hidden_size: ViT hidden dimension
    """
    result = {}

    # Find visual model
    if hasattr(model, 'visual'):
        result['visual_model'] = model.visual
    elif hasattr(model, 'vision_model'):
        result['visual_model'] = model.vision_model
    elif hasattr(model, 'vision_tower'):
        result['visual_model'] = model.vision_tower
    else:
        raise ValueError("Could not find vision encoder in model")

    # Find mmproj / visual projection
    result['mmproj_weight'] = None
    result['mmproj_bias'] = None
    for name in ['visual_projection', 'mm_projector', 'multi_modal_projector']:
        if hasattr(model, name):
            proj = getattr(model, name)
            if isinstance(proj, torch.nn.Linear):
                result['mmproj_weight'] = proj.weight.data
                result['mmproj_bias'] = proj.bias.data if proj.bias is not None else None
            else:
                # MLP or more complex projection — store the module
                result['mmproj_module'] = proj
            break

    # Find embedding table
    for name, module in model.named_modules():
        if name.endswith('embed_tokens') or name.endswith('wte'):
            result['embed_table'] = module
            break

    # Config values
    result['image_token_id'] = getattr(model.config, 'image_token_id', 248056)
    result['hidden_size'] = model.config.hidden_size
    if hasattr(model.config, 'vision_config'):
        vc = model.config.vision_config
        result['vision_hidden_size'] = vc.get('hidden_size', None)
        result['patch_size'] = vc.get('patch_size', 16)
        result['merge_size'] = vc.get('spatial_merge_size', 2)

    return result


def check_model_ready(model_path: str) -> bool:
    """Check if a model is fully downloaded and ready to load."""
    path = Path(model_path)
    if not path.exists():
        return False

    # Check for safetensors files
    safetensors = list(path.glob("*.safetensors"))
    if not safetensors:
        return False

    # Check for config
    if not (path / "config.json").exists():
        return False

    # Check for model index (tells us how many shards to expect)
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        import json
        with open(index_file) as f:
            index = json.load(f)
        expected_files = set(index.get("weight_map", {}).values())
        for ef in expected_files:
            if not (path / ef).exists():
                return False

    return True
