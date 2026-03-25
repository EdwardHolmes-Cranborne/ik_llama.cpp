#!/usr/bin/env python3
"""
Phase 3: WIR Modification of Vision Tokens

Tests whether modifying vision token embeddings via the VLM's own reasoning
produces meaningful changes when decoded back to an image.

Pipeline:
  1. Encode image → vision token embeddings
  2. Construct a prompt: [text context] + [vision tokens] + [instruction]
     "Given this scene and that the viewer is feeling afraid, how would
      this scene look different?"
  3. Run a forward pass — the model's hidden states at the vision token
     positions represent its modified version of the image
  4. WIR soft-refine: blend original embeddings with model's output
  5. Decode modified embeddings back to image via feature inversion

Usage:
  python phase3_wir_modify.py \
    --model-path /path/to/qwen-vl \
    --image test.jpg \
    --emotion "terrified, seeing danger everywhere" \
    --output modified.png \
    --alpha 0.3 \
    --wir-passes 3
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from phase1_inversion import (
    load_vit_and_mmproj,
    encode_image,
    pseudoinverse_mmproj,
    feature_inversion,
    tensor_to_pil,
    compare_images,
)


def wir_soft_refine(original_embeddings, model_output_embeddings,
                    alpha=0.3, confidence_weighted=True):
    """
    WIR soft refinement in embedding space.

    Blend original vision token embeddings with the model's modified version.
    alpha controls how much of the model's modification to apply.

    If confidence_weighted, positions where the model's output differs most
    from the original get higher alpha (the model is more sure about the change).
    """
    diff = model_output_embeddings - original_embeddings
    diff_magnitude = torch.norm(diff, dim=-1, keepdim=True)  # per-token change magnitude

    if confidence_weighted:
        # Normalize magnitudes to [0, 1] range
        max_diff = diff_magnitude.max()
        if max_diff > 0:
            position_alpha = alpha * (diff_magnitude / max_diff)
        else:
            position_alpha = torch.full_like(diff_magnitude, alpha)
    else:
        position_alpha = torch.full_like(diff_magnitude, alpha)

    refined = original_embeddings + position_alpha * diff

    # Report
    mean_alpha = position_alpha.mean().item()
    max_change = diff_magnitude.max().item()
    changed_positions = (diff_magnitude.squeeze(-1) > 0.01).sum().item()
    total_positions = diff_magnitude.shape[1]
    print(f"  WIR refine: mean_alpha={mean_alpha:.4f}, max_change={max_change:.4f}, "
          f"changed={changed_positions}/{total_positions}")

    return refined


def get_model_modified_embeddings(full_model, processor,
                                  image: Image.Image,
                                  emotion_text: str,
                                  device="cuda"):
    """
    Run the image + emotional context through the full VLM and extract
    the model's hidden states at vision token positions.

    The hidden states (BEFORE the LM head) are in the same 4096-dim space
    as the input vision embeddings. They represent the model's "re-imagined"
    version of each vision token after attending to the emotional context.

    These hidden states are the WIR update signal — we blend them with the
    original vision embeddings to shift the image toward the model's
    subjective version.
    """
    prompt = (
        f"You are viewing this scene while feeling {emotion_text}. "
        f"Focus on what feels most significant given your emotional state."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Forward pass — we need hidden states, NOT logits
    with torch.no_grad():
        outputs = full_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Hidden states from the last transformer layer (before LM head)
    # Shape: (batch, seq_len, hidden_size=4096)
    last_hidden = outputs.hidden_states[-1]

    # Find vision token positions in the input sequence
    # Qwen3.5 marks image tokens with image_token_id=248056 in input_ids
    image_token_id = getattr(full_model.config, 'image_token_id', 248056)
    vision_mask = (inputs.input_ids == image_token_id).squeeze(0)  # (seq_len,)
    n_vision = vision_mask.sum().item()

    if n_vision == 0:
        # Try alternative: look for contiguous non-text regions
        # Some models use a different marking scheme
        print("  Warning: no image_token_id found in input_ids")
        print(f"  input_ids unique values (sample): {inputs.input_ids.unique()[:20].tolist()}")
        return None, None, 0

    # Extract hidden states at vision positions only
    vision_hidden = last_hidden[:, vision_mask, :]  # (1, n_vision, 4096)

    # Also extract the original vision embeddings that were fed to the model
    # These are the input embeddings at vision positions (before any transformer layers)
    input_embeds = outputs.hidden_states[0]  # first "hidden state" = input embeddings
    vision_input_embeds = input_embeds[:, vision_mask, :]  # (1, n_vision, 4096)

    print(f"  Vision tokens found: {n_vision}")
    print(f"  Vision hidden states shape: {vision_hidden.shape}")
    print(f"  Vision input embeds shape:  {vision_input_embeds.shape}")

    # The delta between input and output at vision positions is what the model
    # "wants to change" about the image given the emotional context
    delta = vision_hidden - vision_input_embeds
    delta_magnitude = delta.norm(dim=-1).mean().item()
    print(f"  Mean embedding delta magnitude: {delta_magnitude:.4f}")

    return vision_input_embeds, vision_hidden, n_vision


def simple_emotional_perturbation(embeddings, emotion: str, strength: float = 0.1):
    """
    Fallback: simple heuristic perturbation of embeddings based on emotion.
    Not as good as running through the model, but useful for testing the
    decode pipeline without a full VLM forward pass.
    """
    perturbed = embeddings.clone()

    emotion_lower = emotion.lower()

    if any(w in emotion_lower for w in ['afraid', 'fear', 'terrif', 'scar']):
        # Darken: reduce magnitude of embeddings slightly
        # Increase variance: amplify differences between patches
        mean = perturbed.mean(dim=-1, keepdim=True)
        perturbed = mean + (perturbed - mean) * (1 + strength)  # increase contrast
        perturbed *= (1 - strength * 0.3)  # slight overall darkening

    elif any(w in emotion_lower for w in ['happy', 'joy', 'content', 'warm']):
        # Brighten: increase magnitude slightly
        # Smooth: reduce variance between patches
        mean = perturbed.mean(dim=-1, keepdim=True)
        perturbed = mean + (perturbed - mean) * (1 - strength * 0.3)  # reduce contrast
        perturbed *= (1 + strength * 0.2)  # slight brightening

    elif any(w in emotion_lower for w in ['angry', 'rage', 'fury']):
        # Sharpen: increase variance strongly
        mean = perturbed.mean(dim=-1, keepdim=True)
        perturbed = mean + (perturbed - mean) * (1 + strength * 1.5)

    elif any(w in emotion_lower for w in ['sad', 'melanchol', 'depress']):
        # Desaturate: move toward mean
        mean = perturbed.mean(dim=-1, keepdim=True)
        perturbed = mean + (perturbed - mean) * (1 - strength * 0.5)

    else:
        # Generic perturbation — add scaled noise
        noise = torch.randn_like(perturbed) * strength * perturbed.std()
        perturbed += noise

    print(f"  Emotional perturbation: '{emotion}', strength={strength:.2f}")
    return perturbed


def main():
    parser = argparse.ArgumentParser(description="Phase 3: WIR modification of vision tokens")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to VLM (HuggingFace format)")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image")
    parser.add_argument("--emotion", type=str, default="afraid, seeing danger",
                        help="Emotional context for re-imagination")
    parser.add_argument("--output", type=str, default="modified.png",
                        help="Output path for modified image")
    parser.add_argument("--comparison", type=str, default="modification_comparison.png",
                        help="Output path for comparison")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="WIR blending strength (0=no change, 1=full model output)")
    parser.add_argument("--wir-passes", type=int, default=3,
                        help="Number of WIR soft refinement passes")
    parser.add_argument("--inversion-steps", type=int, default=20,
                        help="Feature inversion gradient steps")
    parser.add_argument("--use-heuristic", action="store_true",
                        help="Use simple heuristic perturbation instead of model forward pass")
    parser.add_argument("--perturbation-strength", type=float, default=0.15,
                        help="Strength for heuristic perturbation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=336)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    vit, mmproj, processor, config = load_vit_and_mmproj(args.model_path, device)

    # Encode original image
    print(f"\n=== Encoding {args.image} ===")
    original = Image.open(args.image).convert("RGB")
    llm_embeddings, patch_embeddings = encode_image(original, vit, mmproj, processor, device)

    # Modify embeddings
    if args.use_heuristic:
        print(f"\n=== Heuristic emotional perturbation ===")
        modified_embeddings = llm_embeddings.clone()
        for pass_n in range(args.wir_passes):
            print(f"  Pass {pass_n + 1}/{args.wir_passes}:")
            perturbed = simple_emotional_perturbation(
                modified_embeddings, args.emotion, args.perturbation_strength
            )
            modified_embeddings = wir_soft_refine(
                modified_embeddings, perturbed,
                alpha=args.alpha, confidence_weighted=True
            )
    else:
        print(f"\n=== Model-based WIR re-imagination ({args.wir_passes} passes) ===")

        # Load the full model for hidden state extraction
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            print(f"  Loading full model for WIR forward passes...")
            full_model = AutoModelForImageTextToText.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            full_processor = AutoProcessor.from_pretrained(
                args.model_path, trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Could not load full model: {e}")
            print(f"  Falling back to heuristic perturbation")
            args.use_heuristic = True

        if not args.use_heuristic:
            modified_embeddings = llm_embeddings.clone()

            for pass_n in range(args.wir_passes):
                print(f"\n  WIR Pass {pass_n + 1}/{args.wir_passes}:")

                # We need to re-encode with modified embeddings each pass
                # For the first pass, use the original image
                # For subsequent passes, we'd ideally inject modified embeddings
                # directly — but for now, we use the model's hidden states
                # as the refinement signal on the original image + text context

                vision_input, vision_hidden, n_vision = get_model_modified_embeddings(
                    full_model, full_processor,
                    original, args.emotion, device
                )

                if vision_hidden is None:
                    print("  Failed to extract vision hidden states. Falling back to heuristic.")
                    perturbed = simple_emotional_perturbation(
                        modified_embeddings, args.emotion, args.perturbation_strength
                    )
                    modified_embeddings = wir_soft_refine(
                        modified_embeddings, perturbed,
                        alpha=args.alpha, confidence_weighted=True
                    )
                else:
                    # The model's hidden states at vision positions are the
                    # "re-imagined" version. Blend with current embeddings.
                    # Match shapes — llm_embeddings and vision_hidden may differ
                    # if the model added/removed special tokens
                    n_match = min(modified_embeddings.shape[1], vision_hidden.shape[1])
                    modified_embeddings = wir_soft_refine(
                        modified_embeddings[:, :n_match, :],
                        vision_hidden[:, :n_match, :],
                        alpha=args.alpha,
                        confidence_weighted=True,
                    )

    # Compute modification magnitude
    diff = (modified_embeddings - llm_embeddings).norm(dim=-1)
    print(f"\n=== Modification summary ===")
    print(f"  Mean embedding change: {diff.mean().item():.4f}")
    print(f"  Max embedding change:  {diff.max().item():.4f}")
    print(f"  Positions changed >1%: {(diff > 0.01 * llm_embeddings.norm(dim=-1)).sum().item()}")

    # Decode: pseudoinverse + feature inversion
    print(f"\n=== Decoding original (for comparison) ===")
    recovered_orig = pseudoinverse_mmproj(llm_embeddings, mmproj)
    inverted_orig = feature_inversion(
        recovered_orig, vit, processor,
        n_steps=args.inversion_steps, lr=0.1,
        image_size=args.image_size, device=device
    )

    print(f"\n=== Decoding modified ===")
    recovered_mod = pseudoinverse_mmproj(modified_embeddings, mmproj)
    inverted_mod = feature_inversion(
        recovered_mod, vit, processor,
        n_steps=args.inversion_steps, lr=0.1,
        image_size=args.image_size, device=device
    )

    # Save
    orig_pil = tensor_to_pil(inverted_orig)
    mod_pil = tensor_to_pil(inverted_mod)
    mod_pil.save(args.output)
    print(f"\nModified image saved to {args.output}")

    # Three-way comparison: original photo | original inversion | modified inversion
    h = 512
    images = [original, orig_pil, mod_pil]
    labels = ["Original", "Inverted", f"Modified ({args.emotion})"]
    widths = [int(img.width * h / img.height) for img in images]
    gap = 10
    total_w = sum(widths) + gap * (len(images) - 1)

    combined = Image.new('RGB', (total_w, h), (40, 40, 40))
    x = 0
    for img, w in zip(images, widths):
        combined.paste(img.resize((w, h), Image.LANCZOS), (x, 0))
        x += w + gap
    combined.save(args.comparison)
    print(f"Comparison saved to {args.comparison}")

    print(f"\n=== Phase 3 complete ===")
    print(f"Inspect {args.comparison} to see if the emotional modification is visible.")
    print(f"If the modified image looks different from the original inversion")
    print(f"in a way that corresponds to '{args.emotion}' → Phase 3 PASS")


if __name__ == "__main__":
    main()
