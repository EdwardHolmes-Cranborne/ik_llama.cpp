"""Exhaustive image description via VLM.

Generates a detailed, structured description of everything visible in an image.
The description is used as the target text for LLM inversion.
"""

import time
import torch
from PIL import Image
from typing import Tuple


# Fixed exhaustive description prompt — never changes, KV-cacheable
DESCRIBE_PROMPT = """Describe everything you see in this image in exhaustive detail. Include:

COMPOSITION: Overall layout, foreground/midground/background, framing
PEOPLE: Exact poses (limb positions, body angles), facial expressions, eye direction, gestures
CLOTHING: Every garment, colours, textures, patterns, accessories, footwear
OBJECTS: Every visible object, its position, size relative to other elements, condition
LIGHTING: Direction, colour temperature, shadows (direction, hardness), highlights, time of day
COLOURS: Dominant palette, accent colours, colour relationships, saturation levels
TEXTURES: Surface qualities of every material visible (skin, fabric, stone, wood, metal, etc.)
SPATIAL: Distances between elements, depth cues, perspective lines, vanishing points
ATMOSPHERE: Weather, haze, dust, mood conveyed by the visual elements
BACKGROUND: Everything behind the main subjects, architectural details, landscape, sky

Be precise about spatial positions (left/right, high/low, near/far).
Be precise about body poses (which arm, which direction, angle of head).
Describe as if the reader must recreate this exact image from your words alone."""


def describe_image(
    model,
    processor,
    image: Image.Image,
    device: str = "cuda",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    verbose: bool = True,
) -> Tuple[str, dict]:
    """
    Generate an exhaustive description of an image using the VLM.

    Args:
        model: the full VLM (e.g., Qwen3.5-9B)
        processor: the VLM's processor
        image: PIL Image to describe
        device: computation device
        max_new_tokens: maximum description length
        temperature: sampling temperature (low = precise/deterministic)
        verbose: print progress

    Returns:
        description: the generated text description
        info: dict with metadata (n_tokens, time, vision_mask, etc.)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DESCRIBE_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Find vision token positions
    image_token_id = getattr(model.config, 'image_token_id', 248056)
    vision_mask = (inputs.input_ids == image_token_id).squeeze(0)
    n_vision = vision_mask.sum().item()

    if verbose:
        print(f"  Input sequence: {inputs.input_ids.shape[1]} tokens "
              f"({n_vision} vision tokens)")
        print(f"  Generating description (max {max_new_tokens} tokens)...")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    elapsed = time.time() - t0

    # Decode only the new tokens
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    description = processor.decode(new_tokens, skip_special_tokens=True)

    info = {
        "n_vision_tokens": n_vision,
        "n_description_tokens": len(new_tokens),
        "input_seq_len": inputs.input_ids.shape[1],
        "generation_time_s": elapsed,
        "tokens_per_second": len(new_tokens) / elapsed if elapsed > 0 else 0,
        "vision_mask": vision_mask,
        "inputs": inputs,
    }

    if verbose:
        print(f"  Generated {len(new_tokens)} tokens in {elapsed:.1f}s "
              f"({info['tokens_per_second']:.0f} tok/s)")
        preview = description[:300]
        print(f"  Preview: {preview}{'...' if len(description) > 300 else ''}")

    return description, info
