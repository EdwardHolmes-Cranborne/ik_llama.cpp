#!/usr/bin/env python3
"""
Phase 2 — Fuyu WIR: The model directly refines its own image tokens.

Fuyu's image patches are linearly projected into the same token embedding
space as text tokens. They go through the same LM head. So in theory,
the model can predict replacement image tokens the same way it predicts
replacement text tokens.

This is WIR applied to image tokens:
  1. Encode image → image tokens (via Fuyu's linear projection)
  2. Model describes the image in detail
  3. Edit the description
  4. Feed [image tokens] + [modified description] back through the model
  5. At image token positions, read the LM head's prediction
  6. n-least-conf: replace the image tokens the model is least confident about
  7. Repeat until converged
  8. Decode final image tokens → pixels

Usage:
  # Describe and refine
  python phase2_fuyu_wir.py \
    --model-path D:/models/Fuyu-8B-HF \
    --image market_scene.jpg \
    --edit "change 'arm at his side' to 'arm raised above his head'" \
    --passes 5 \
    --output refined.png

  # Just see what tokens the model predicts at image positions (diagnostic)
  python phase2_fuyu_wir.py \
    --model-path D:/models/Fuyu-8B-HF \
    --image test.jpg \
    --diagnostic
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.editor import apply_edit
from phase1_fuyu import (
    load_fuyu_patch_projection,
    image_to_patches,
    patches_to_image,
)
from core.projector import forward_project, pseudoinverse


def load_fuyu_full(model_path: str, device: str = "cuda"):
    """Load the full Fuyu model for inference."""
    from transformers import FuyuForCausalLM, FuyuProcessor

    print(f"Loading Fuyu-8B from {model_path}...")
    processor = FuyuProcessor.from_pretrained(model_path)
    model = FuyuForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    print(f"  Loaded: {model.config.model_type}")
    return model, processor


def encode_image_for_fuyu(image: Image.Image, processor, device="cuda"):
    """
    Process an image through Fuyu's processor to get the model inputs.
    Returns the inputs dict and identifies which positions are image tokens.
    """
    # Fuyu processor handles image → patch → token conversion
    prompt = "Describe this image in detail.\n"
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # Find image token positions
    # Fuyu uses a special image token or a range of IDs for image patches
    # The processor inserts image tokens before the text
    input_ids = inputs["input_ids"].squeeze(0)

    # Fuyu typically uses token ID 0 or a specific range for image patches
    # We need to identify which positions are image vs text
    # The image_patches field tells us about the image content
    if "image_patches" in inputs:
        n_image_patches = inputs["image_patches"][0].shape[0] if inputs["image_patches"][0] is not None else 0
    else:
        n_image_patches = 0

    print(f"  Input length: {len(input_ids)} tokens")
    print(f"  Image patches: {n_image_patches}")
    print(f"  First 20 token IDs: {input_ids[:20].tolist()}")

    return inputs, n_image_patches


def get_image_token_predictions(model, inputs, n_image_tokens, device="cuda"):
    """
    Forward pass through Fuyu. Extract the model's predictions
    at image token positions.

    Returns:
        predicted_ids: what the model would predict at each image position
        confidences: how confident the model is in each prediction
        logits: raw logits at image positions (for n-least-conf)
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Image tokens are at the start of the sequence (before the text prompt)
    # The exact positions depend on Fuyu's processor
    # For now, assume image tokens are positions 0..n_image_tokens-1
    image_logits = logits[0, :n_image_tokens, :]  # (n_image_tokens, vocab_size)

    # Get top predictions
    probs = F.softmax(image_logits.float(), dim=-1)
    confidences, predicted_ids = probs.max(dim=-1)

    # Also get the probability of the CURRENT token at each position
    current_ids = inputs["input_ids"][0, :n_image_tokens]
    current_probs = probs.gather(1, current_ids.unsqueeze(-1)).squeeze(-1)

    return predicted_ids, confidences, current_probs, image_logits


def n_least_confident_replace(current_ids, predicted_ids, current_probs, confidences, n=10):
    """
    WIR n-least-conf: replace the N image tokens where the model is
    LEAST confident in the current token.

    These are the tokens the model thinks are most "wrong" given the context.
    """
    # Sort by current token probability (ascending = least confident first)
    _, sorted_indices = current_probs.sort()

    # Take the N least confident
    to_replace = sorted_indices[:n]

    new_ids = current_ids.clone()
    replaced = []
    for idx in to_replace:
        idx = idx.item()
        old_id = current_ids[idx].item()
        new_id = predicted_ids[idx].item()
        old_conf = current_probs[idx].item()
        new_conf = confidences[idx].item()

        if old_id != new_id:
            new_ids[idx] = predicted_ids[idx]
            replaced.append({
                "position": idx,
                "old_token": old_id,
                "new_token": new_id,
                "old_confidence": old_conf,
                "new_confidence": new_conf,
            })

    return new_ids, replaced


def wir_refine_image_tokens(model, processor, inputs, n_image_tokens,
                              description, modified_description,
                              n_passes=5, n_replace=10,
                              device="cuda", verbose=True):
    """
    WIR refinement loop over image tokens.

    Each pass:
      1. Feed current image tokens + modified description to model
      2. Model predicts what each image token "should" be
      3. n-least-conf replaces the worst tokens
      4. Repeat until converged
    """
    current_input_ids = inputs["input_ids"].clone()
    results = []

    for pass_n in range(n_passes):
        if verbose:
            print(f"\n  === WIR Pass {pass_n + 1}/{n_passes} ===")

        # Build the prompt with modified description
        # Replace the original description prompt with the modified one
        # For Fuyu, we append the modification instruction
        modify_prompt = (
            f"This image should show: {modified_description}\n"
            f"Looking at the image tokens, which ones need to change?"
        )

        # Create modified inputs with the new prompt
        # We keep the image tokens and change the text
        modified_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
        modified_inputs["input_ids"] = current_input_ids

        # Forward pass
        predicted_ids, confidences, current_probs, image_logits = \
            get_image_token_predictions(model, modified_inputs, n_image_tokens, device)

        # n-least-conf replacement
        new_ids, replaced = n_least_confident_replace(
            current_input_ids[0, :n_image_tokens],
            predicted_ids,
            current_probs,
            confidences,
            n=n_replace,
        )

        # Update the input
        current_input_ids[0, :n_image_tokens] = new_ids

        # Stats
        n_changed = len(replaced)
        mean_old_conf = np.mean([r["old_confidence"] for r in replaced]) if replaced else 0
        mean_new_conf = np.mean([r["new_confidence"] for r in replaced]) if replaced else 0

        pass_result = {
            "pass": pass_n + 1,
            "tokens_changed": n_changed,
            "mean_old_confidence": mean_old_conf,
            "mean_new_confidence": mean_new_conf,
            "replacements": replaced,
        }
        results.append(pass_result)

        if verbose:
            print(f"  Changed: {n_changed} tokens")
            print(f"  Mean confidence: {mean_old_conf:.4f} → {mean_new_conf:.4f}")
            for r in replaced[:5]:
                print(f"    pos {r['position']}: token {r['old_token']} → {r['new_token']} "
                      f"(conf {r['old_confidence']:.4f} → {r['new_confidence']:.4f})")

        if n_changed == 0:
            if verbose:
                print(f"  Converged — no more changes needed")
            break

    return current_input_ids, results


def decode_token_ids_to_image(token_ids, model, proj_weight, proj_bias,
                                patch_size=30, image_size=300):
    """
    Decode image token IDs back to an image.

    Two approaches:
    1. Look up token embeddings from the embedding table → pseudoinverse → pixels
    2. Direct: if tokens map to patch indices, look up the projection directly

    Using approach 1 (embedding table → pseudoinverse):
    """
    # Get the embedding table
    embed_table = None
    for name, module in model.named_modules():
        if 'embed_tokens' in name or 'word_embeddings' in name:
            embed_table = module
            break
    if embed_table is None:
        raise ValueError("Could not find embedding table")

    # Look up embeddings for the image token IDs
    with torch.no_grad():
        embeddings = embed_table(token_ids)  # (n_tokens, hidden_dim)

    # Pseudoinverse the projection to get back to patch pixel space
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(0)

    recovered_patches = pseudoinverse(embeddings, proj_weight, proj_bias)

    # Reassemble to image
    return patches_to_image(recovered_patches, patch_size, image_size)


def diagnostic_mode(model, processor, inputs, n_image_tokens, device="cuda"):
    """
    Diagnostic: examine what the model predicts at image token positions.

    This reveals whether the model "understands" image tokens well enough
    to predict meaningful replacements.
    """
    print("\n=== DIAGNOSTIC: Image Token Predictions ===")

    predicted_ids, confidences, current_probs, image_logits = \
        get_image_token_predictions(model, inputs, n_image_tokens, device)

    current_ids = inputs["input_ids"][0, :n_image_tokens]

    print(f"\nImage tokens: {n_image_tokens}")
    print(f"Mean confidence in current tokens: {current_probs.mean().item():.4f}")
    print(f"Mean confidence in predicted tokens: {confidences.mean().item():.4f}")

    # How many tokens does the model want to change?
    different = (predicted_ids != current_ids).sum().item()
    print(f"Tokens model wants to change: {different}/{n_image_tokens} "
          f"({different/n_image_tokens*100:.1f}%)")

    # Distribution of confidences
    print(f"\nCurrent token confidence distribution:")
    for thresh in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
        count = (current_probs > thresh).sum().item()
        print(f"  >{thresh:.2f}: {count}/{n_image_tokens} "
              f"({count/n_image_tokens*100:.1f}%)")

    # Top-5 most and least confident positions
    print(f"\n5 LEAST confident (most likely to change):")
    _, worst_idx = current_probs.sort()
    for i in range(min(5, n_image_tokens)):
        idx = worst_idx[i].item()
        print(f"  pos {idx}: current={current_ids[idx].item()} "
              f"conf={current_probs[idx].item():.4f} "
              f"→ predicted={predicted_ids[idx].item()} "
              f"pred_conf={confidences[idx].item():.4f}")

    print(f"\n5 MOST confident (stable):")
    _, best_idx = current_probs.sort(descending=True)
    for i in range(min(5, n_image_tokens)):
        idx = best_idx[i].item()
        print(f"  pos {idx}: current={current_ids[idx].item()} "
              f"conf={current_probs[idx].item():.4f}")

    # Entropy at image positions (how uncertain is the model?)
    probs = F.softmax(image_logits.float(), dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    max_entropy = np.log(model.config.vocab_size)
    norm_entropy = entropy / max_entropy

    print(f"\nEntropy at image positions:")
    print(f"  Mean normalised entropy: {norm_entropy.mean().item():.4f}")
    print(f"  Min: {norm_entropy.min().item():.4f}")
    print(f"  Max: {norm_entropy.max().item():.4f}")
    print(f"  (0 = certain, 1 = uniform random)")

    return {
        "n_image_tokens": n_image_tokens,
        "mean_current_conf": current_probs.mean().item(),
        "mean_predicted_conf": confidences.mean().item(),
        "n_different": different,
        "mean_entropy": norm_entropy.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fuyu WIR image token refinement")
    parser.add_argument("--model-path", type=str, default="D:/models/Fuyu-8B-HF")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--edit", type=str, default=None,
                        help="Edit instruction for the description")
    parser.add_argument("--output", type=str, default="fuyu_wir_output.png")
    parser.add_argument("--comparison", type=str, default="fuyu_wir_comparison.png")
    parser.add_argument("--passes", type=int, default=5,
                        help="Number of WIR refinement passes")
    parser.add_argument("--n-replace", type=int, default=10,
                        help="Tokens to replace per pass (n-least-conf)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Just examine model predictions at image positions")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    model, processor = load_fuyu_full(args.model_path, args.device)

    # Load image and get model inputs
    print(f"\n=== Processing {args.image} ===")
    image = Image.open(args.image).convert("RGB")
    inputs, n_image_tokens = encode_image_for_fuyu(image, processor, args.device)

    if args.diagnostic:
        diagnostic_mode(model, processor, inputs, n_image_tokens, args.device)
        return

    # Step 1: Describe the image
    print(f"\n=== Step 1: Model describes the image ===")
    describe_prompt = "Describe everything you see in this image in exhaustive detail.\n"
    desc_inputs = processor(text=describe_prompt, images=[image], return_tensors="pt").to(args.device)

    with torch.no_grad():
        output_ids = model.generate(
            **desc_inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
    description = processor.decode(output_ids[0, desc_inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
    print(f"  Description: {description[:300]}...")

    # Step 2: Edit the description
    if args.edit:
        print(f"\n=== Step 2: Editing description ===")
        modified_description = apply_edit(description, args.edit)
        print(f"  Edit: {args.edit}")
    else:
        print("No --edit provided. Running diagnostic mode instead.")
        diagnostic_mode(model, processor, inputs, n_image_tokens, args.device)
        return

    # Step 3: WIR refinement
    print(f"\n=== Step 3: WIR refinement ({args.passes} passes, n={args.n_replace}) ===")
    refined_input_ids, wir_results = wir_refine_image_tokens(
        model, processor, inputs, n_image_tokens,
        description, modified_description,
        n_passes=args.passes, n_replace=args.n_replace,
        device=args.device,
    )

    # Step 4: Decode modified image tokens back to image
    print(f"\n=== Step 4: Decoding modified tokens to image ===")
    proj_weight, proj_bias, config = load_fuyu_patch_projection(args.model_path, "cpu")
    patch_size = config.get("patch_size", 30)
    image_size = config.get("image_size", 300)

    refined_image_token_ids = refined_input_ids[0, :n_image_tokens].cpu()
    refined_image = decode_token_ids_to_image(
        refined_image_token_ids, model,
        proj_weight, proj_bias,
        patch_size=patch_size, image_size=image_size,
    )
    refined_image.save(args.output)
    print(f"  Saved to {args.output}")

    # Also decode the original for comparison
    original_token_ids = inputs["input_ids"][0, :n_image_tokens].cpu()
    original_decoded = decode_token_ids_to_image(
        original_token_ids, model,
        proj_weight, proj_bias,
        patch_size=patch_size, image_size=image_size,
    )

    # Build comparison: original photo | original decoded | WIR modified
    h = 300
    images_list = [
        image.resize((h, h), Image.LANCZOS),
        original_decoded.resize((h, h), Image.LANCZOS),
        refined_image.resize((h, h), Image.LANCZOS),
    ]
    gap = 6
    total_w = len(images_list) * h + (len(images_list) - 1) * gap
    comparison = Image.new('RGB', (total_w, h), (30, 30, 30))
    x = 0
    for img in images_list:
        comparison.paste(img, (x, 0))
        x += h + gap
    comparison.save(args.comparison)
    print(f"  Comparison saved to {args.comparison}")

    # Summary
    total_changed = sum(r["tokens_changed"] for r in wir_results)
    print(f"\n=== Summary ===")
    print(f"  WIR passes: {len(wir_results)}")
    print(f"  Total tokens changed: {total_changed}/{n_image_tokens}")
    print(f"  Description edit: {args.edit}")
    print(f"\n  If the refined image differs from the original in a way")
    print(f"  that matches the edit → Fuyu WIR image editing WORKS")


if __name__ == "__main__":
    main()
