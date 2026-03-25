"""Image encoding via VLM's vision transformer.

Handles image → ViT → patch embeddings → mmproj → LLM embeddings.
"""

import torch
from PIL import Image
from typing import Tuple, Optional


def get_vision_token_count(image_height: int, image_width: int,
                            patch_size: int = 16, merge_size: int = 2) -> int:
    """
    Calculate how many vision tokens an image produces.

    Qwen3.5 pipeline:
      patches = (H / patch_size) × (W / patch_size)
      tokens = patches / (merge_size²)

    Args:
        image_height: input image height in pixels
        image_width: input image width in pixels
        patch_size: ViT patch size (default 16 for Qwen3.5)
        merge_size: spatial merge factor (default 2 for Qwen3.5)

    Returns:
        Number of vision tokens in LLM embedding space
    """
    patches_h = image_height // patch_size
    patches_w = image_width // patch_size
    total_patches = patches_h * patches_w
    tokens = total_patches // (merge_size * merge_size)
    return tokens


def encode_image(
    image: Image.Image,
    visual_model: torch.nn.Module,
    processor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """
    Encode an image to vision token embeddings using the VLM's visual model.

    Args:
        image: PIL Image
        visual_model: the VLM's visual encoder (e.g., model.visual)
        processor: the VLM's processor (for image preprocessing)
        device: target device

    Returns:
        vision_embeddings: (1, n_tokens, hidden_dim) in LLM embedding space
        info: dict with metadata (n_tokens, patch_grid_size, etc.)
    """
    # Preprocess the image using the model's processor
    # Qwen3.5 uses the processor to handle dynamic resolution
    if hasattr(processor, 'image_processor'):
        pixel_values = processor.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"]
    else:
        pixel_values = processor(
            images=image, return_tensors="pt"
        )["pixel_values"]

    pixel_values = pixel_values.to(device=device, dtype=torch.float16)

    # Forward through visual model
    with torch.no_grad():
        vision_output = visual_model(pixel_values)

    # Extract embeddings — handle different output formats
    if isinstance(vision_output, tuple):
        embeddings = vision_output[0]
    elif hasattr(vision_output, 'last_hidden_state'):
        embeddings = vision_output.last_hidden_state
    else:
        embeddings = vision_output

    info = {
        "n_tokens": embeddings.shape[1],
        "hidden_dim": embeddings.shape[2],
        "pixel_values_shape": tuple(pixel_values.shape),
        "dtype": str(embeddings.dtype),
    }

    return embeddings, info


def extract_vision_embeddings_from_model_input(
    model,
    processor,
    image: Image.Image,
    text: str = "Describe this image.",
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Process an image through the full model's input pipeline and extract
    the vision token embeddings at their positions in the sequence.

    This captures the embeddings AFTER the full projection into LLM space,
    exactly as the transformer layers will see them.

    Args:
        model: the full VLM
        processor: the VLM's processor
        image: PIL Image
        text: text prompt to pair with the image
        device: target device

    Returns:
        vision_embeddings: (1, n_vision_tokens, hidden_dim)
        vision_mask: (seq_len,) boolean mask of vision token positions
        info: dict with metadata
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Find vision token positions
    image_token_id = getattr(model.config, 'image_token_id', 248056)
    vision_mask = (inputs.input_ids == image_token_id).squeeze(0)
    n_vision = vision_mask.sum().item()

    # Forward through just the embedding + visual encoding stage
    # We want the input embeddings the transformer will see
    with torch.no_grad():
        # Get text embeddings
        embed_table = None
        for name, module in model.named_modules():
            if name.endswith('embed_tokens') or name.endswith('wte'):
                embed_table = module
                break

        if embed_table is None:
            raise ValueError("Could not find embedding table in model")

        input_embeds = embed_table(inputs.input_ids)

        # Get vision embeddings from the visual model
        if hasattr(model, 'visual'):
            if 'pixel_values' in inputs:
                vis_out = model.visual(inputs['pixel_values'])
            elif 'pixel_values_videos' in inputs:
                vis_out = model.visual(inputs['pixel_values_videos'])
            else:
                raise ValueError("No pixel_values found in inputs")

            if isinstance(vis_out, tuple):
                vision_embeds = vis_out[0]
            else:
                vision_embeds = vis_out
        else:
            # Fall back to extracting from the input embeddings at vision positions
            vision_embeds = input_embeds[:, vision_mask, :]

    info = {
        "n_vision_tokens": n_vision,
        "total_seq_len": inputs.input_ids.shape[1],
        "hidden_dim": vision_embeds.shape[-1],
        "vision_positions": vision_mask.nonzero(as_tuple=True)[0].tolist(),
        "image_token_id": image_token_id,
    }

    return vision_embeds, vision_mask, info
