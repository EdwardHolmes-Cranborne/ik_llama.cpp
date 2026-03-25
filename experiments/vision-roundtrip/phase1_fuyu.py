#!/usr/bin/env python3
"""
Phase 1 — Fuyu-8B: The Simplest Possible Vision Token Roundtrip

Fuyu has no ViT. Image patches are linearly projected into token space
via a single matrix multiply. The inverse is a pseudoinverse of that
same matrix. This is the cleanest test of whether vision token
inversion produces recognisable images.

Pipeline:
  Image -> split into 30×30 patches -> linear projection (2700->4096) -> tokens
  Tokens -> pseudoinverse of linear projection (4096->2700) -> pixel patches
  Pixel patches -> reassemble -> image

No gradient descent needed for the basic roundtrip. The pseudoinverse
IS the decoder. We add optional gradient refinement for comparison.

Usage:
  # Basic roundtrip (instant — just matrix multiply)
  python phase1_fuyu.py --model-path D:/models/Fuyu-8B-HF --image test.jpg --output roundtrip.png

  # With gradient refinement (slower but potentially better)
  python phase1_fuyu.py --model-path D:/models/Fuyu-8B-HF --image test.jpg --output refined.png --refine --steps 200

  # Sweep: compare roundtrip quality at different refinement steps
  python phase1_fuyu.py --model-path D:/models/Fuyu-8B-HF --image test.jpg --sweep
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.projector import forward_project, pseudoinverse
from core.inverter import tensor_to_pil, InversionResult


def load_fuyu_patch_projection(model_path: str, device: str = "cpu"):
    """
    Load Fuyu's linear patch projection weight.

    Fuyu's image embedding is a single nn.Linear(2700, 4096):
      patch pixels (30×30×3 = 2700) -> hidden_size (4096)

    Returns: (weight, bias, config)
    """
    import json

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    patch_size = config.get("patch_size", 30)
    num_channels = config.get("num_channels", 3)
    hidden_size = config["hidden_size"]
    image_size = config.get("image_size", 300)

    print(f"Fuyu config: image={image_size}, patch={patch_size}, "
          f"hidden={hidden_size}, channels={num_channels}")
    print(f"  Patch dim: {patch_size}×{patch_size}×{num_channels} = {patch_size*patch_size*num_channels}")
    print(f"  Grid: {image_size//patch_size}×{image_size//patch_size} = "
          f"{(image_size//patch_size)**2} patches")

    # Load the projection weight from safetensors
    # In Fuyu, this is typically "vision_embed_tokens.weight" or similar
    from safetensors.torch import load_file

    weight = None
    bias = None

    # Try each shard
    shard_files = sorted(Path(model_path).glob("*.safetensors"))
    for shard in shard_files:
        tensors = load_file(str(shard), device="cpu")
        for name, tensor in tensors.items():
            if "vision_embed" in name or "image_embed" in name or "patch_embed" in name:
                if "weight" in name:
                    weight = tensor
                    print(f"  Found projection weight: {name} -> {tensor.shape}")
                elif "bias" in name:
                    bias = tensor
                    print(f"  Found projection bias: {name} -> {tensor.shape}")
        if weight is not None:
            break

    if weight is None:
        # Try to find it by shape — looking for (hidden_size, patch_dim)
        patch_dim = patch_size * patch_size * num_channels
        print(f"  Searching by shape ({hidden_size}, {patch_dim})...")
        for shard in shard_files:
            tensors = load_file(str(shard), device="cpu")
            for name, tensor in tensors.items():
                if tensor.shape == torch.Size([hidden_size, patch_dim]):
                    weight = tensor
                    print(f"  Found by shape: {name} -> {tensor.shape}")
                    break
                elif tensor.shape == torch.Size([patch_dim, hidden_size]):
                    weight = tensor.T  # transpose if needed
                    print(f"  Found by shape (transposed): {name} -> {tensor.shape}")
                    break
            if weight is not None:
                break

    if weight is None:
        raise ValueError("Could not find Fuyu's patch projection weight. "
                         "Searched for 'vision_embed', 'image_embed', 'patch_embed' "
                         f"and shape ({hidden_size}, {patch_dim})")

    weight = weight.to(device).float()
    if bias is not None:
        bias = bias.to(device).float()

    return weight, bias, config


def image_to_patches(image: Image.Image, patch_size: int = 30,
                      image_size: int = 300) -> torch.Tensor:
    """
    Convert a PIL image to a tensor of flattened patches.

    Fuyu splits the image into a grid of patch_size × patch_size patches,
    flattens each to a vector of patch_size² × 3 values.

    Returns: (1, n_patches, patch_dim) float tensor normalised to [0, 1]
    """
    # Resize to model's expected size
    image = image.resize((image_size, image_size), Image.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0  # normalise to [0, 1]

    # Split into patches
    n_patches_h = image_size // patch_size
    n_patches_w = image_size // patch_size
    patches = []

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = img_array[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size,
                :
            ]
            patches.append(patch.flatten())

    patches_tensor = torch.tensor(np.array(patches)).unsqueeze(0)  # (1, n_patches, patch_dim)
    return patches_tensor


def patches_to_image(patches: torch.Tensor, patch_size: int = 30,
                      image_size: int = 300) -> Image.Image:
    """
    Reassemble flattened patches back into an image.

    Args:
        patches: (1, n_patches, patch_dim) float tensor
        patch_size: size of each patch
        image_size: output image size

    Returns: PIL Image
    """
    patches = patches.squeeze(0).cpu().float()
    patches = patches.clamp(0, 1)

    n_patches_h = image_size // patch_size
    n_patches_w = image_size // patch_size
    n_channels = 3

    img_array = np.zeros((image_size, image_size, n_channels), dtype=np.float32)

    for idx in range(patches.shape[0]):
        i = idx // n_patches_w
        j = idx % n_patches_w
        patch = patches[idx].numpy().reshape(patch_size, patch_size, n_channels)
        img_array[
            i*patch_size:(i+1)*patch_size,
            j*patch_size:(j+1)*patch_size,
            :
        ] = patch

    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def gradient_refine_patches(target_tokens: torch.Tensor,
                             weight: torch.Tensor,
                             bias: torch.Tensor,
                             patch_size: int = 30,
                             image_size: int = 300,
                             n_steps: int = 200,
                             lr: float = 0.05,
                             device: str = "cpu") -> torch.Tensor:
    """
    Optional: gradient descent to refine the patch reconstruction.

    Instead of just pseudoinversing, optimise the patches so that
    projecting them forward matches the target tokens exactly.
    This can recover detail lost in the pseudoinverse.
    """
    n_patches = target_tokens.shape[1]
    patch_dim = patch_size * patch_size * 3

    # Start from pseudoinverse solution
    initial_patches = pseudoinverse(target_tokens, weight, bias)
    patches = initial_patches.clone().to(device).float().requires_grad_(True)
    target = target_tokens.to(device).float().detach()

    optimizer = torch.optim.Adam([patches], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    print(f"  Gradient refinement: {n_steps} steps, {n_patches} patches")
    best_loss = float('inf')
    best_patches = None

    for step in range(n_steps):
        optimizer.zero_grad()

        projected = forward_project(patches, weight, bias)
        loss = F.mse_loss(projected, target)

        # Regularise: keep pixel values in valid range
        range_penalty = (
            F.relu(-patches).mean() +  # penalise negative values
            F.relu(patches - 1.0).mean()  # penalise values > 1
        )

        total = loss + 0.1 * range_penalty
        total.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_patches = patches.detach().clone()

        if step < 5 or step % 50 == 0 or step == n_steps - 1:
            print(f"    Step {step:4d}: loss={loss.item():.8f} range_pen={range_penalty.item():.6f}")

    best_patches.data.clamp_(0, 1)
    return best_patches


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Fuyu-8B vision token roundtrip")
    parser.add_argument("--model-path", type=str, default="D:/models/Fuyu-8B-HF",
                        help="Path to Fuyu-8B model")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image to encode and decode")
    parser.add_argument("--output", type=str, default="fuyu_roundtrip.png",
                        help="Output path for decoded image")
    parser.add_argument("--comparison", type=str, default="fuyu_comparison.png",
                        help="Side-by-side comparison output")
    parser.add_argument("--refine", action="store_true",
                        help="Use gradient refinement after pseudoinverse")
    parser.add_argument("--steps", type=int, default=200,
                        help="Gradient refinement steps (if --refine)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run sweep: pseudoinverse vs 50/100/200/500 refinement steps")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu is fine — it's just matrix multiplies)")
    args = parser.parse_args()

    # Load the patch projection weight
    print("=== Loading Fuyu patch projection ===")
    weight, bias, config = load_fuyu_patch_projection(args.model_path, args.device)

    patch_size = config.get("patch_size", 30)
    image_size = config.get("image_size", 300)

    # Load and encode image
    print(f"\n=== Encoding {args.image} ===")
    original = Image.open(args.image).convert("RGB")
    print(f"  Original: {original.size}")

    patches = image_to_patches(original, patch_size, image_size)
    print(f"  Patches: {patches.shape} (n_patches={patches.shape[1]}, dim={patches.shape[2]})")

    # Forward project to token space
    patches = patches.to(args.device)
    tokens = forward_project(patches, weight, bias)
    print(f"  Tokens: {tokens.shape}")

    # === BASIC ROUNDTRIP: just pseudoinverse ===
    print(f"\n=== Pseudoinverse roundtrip (instant) ===")
    t0 = time.time()
    recovered_patches = pseudoinverse(tokens, weight, bias)
    t_pinv = time.time() - t0
    print(f"  Pseudoinverse time: {t_pinv*1000:.1f}ms")

    # Measure roundtrip error
    mse = F.mse_loss(recovered_patches.float(), patches.float()).item()
    cos_sim = F.cosine_similarity(
        recovered_patches.reshape(1, -1).float(),
        patches.reshape(1, -1).float()
    ).item()
    print(f"  Roundtrip MSE: {mse:.8f}")
    print(f"  Roundtrip cosine sim: {cos_sim:.6f}")

    # Reassemble to image
    roundtrip_image = patches_to_image(recovered_patches, patch_size, image_size)

    if args.sweep:
        print(f"\n=== Sweep mode ===")
        output_dir = Path(args.output).parent / f"{Path(args.output).stem}_sweep"
        output_dir.mkdir(exist_ok=True)

        # Save pseudoinverse result
        roundtrip_image.save(output_dir / "00_pseudoinverse.png")

        # Run gradient refinement at various step counts
        results = [("Pseudoinverse", roundtrip_image, 0)]
        for n_steps in [50, 100, 200, 500]:
            print(f"\n--- Refining with {n_steps} steps ---")
            refined = gradient_refine_patches(
                tokens, weight, bias,
                patch_size=patch_size, image_size=image_size,
                n_steps=n_steps, lr=0.05, device=args.device,
            )
            refined_image = patches_to_image(refined, patch_size, image_size)
            refined_image.save(output_dir / f"{n_steps:04d}_steps.png")

            ref_mse = F.mse_loss(
                forward_project(refined, weight, bias).float(),
                tokens.float()
            ).item()
            print(f"  Refined MSE: {ref_mse:.8f}")
            results.append((f"{n_steps} steps", refined_image, ref_mse))

        # Build comparison grid
        h = 300
        orig_resized = original.resize((h, h), Image.LANCZOS)
        all_images = [orig_resized] + [img.resize((h, h), Image.LANCZOS) for _, img, _ in results]
        gap = 4
        total_w = len(all_images) * h + (len(all_images) - 1) * gap
        grid = Image.new('RGB', (total_w, h), (30, 30, 30))
        x = 0
        for img in all_images:
            grid.paste(img, (x, 0))
            x += h + gap
        grid.save(output_dir / "sweep_grid.png")
        print(f"\nSweep grid saved to {output_dir / 'sweep_grid.png'}")
        print(f"Order: Original | Pseudoinverse | 50 steps | 100 | 200 | 500")

    elif args.refine:
        print(f"\n=== Gradient refinement ({args.steps} steps) ===")
        refined = gradient_refine_patches(
            tokens, weight, bias,
            patch_size=patch_size, image_size=image_size,
            n_steps=args.steps, lr=0.05, device=args.device,
        )
        roundtrip_image = patches_to_image(refined, patch_size, image_size)

    # Save output
    roundtrip_image.save(args.output)
    print(f"\nOutput saved to {args.output}")

    # Build comparison: original | roundtrip
    h = max(300, image_size)
    orig_resized = original.resize((h, h), Image.LANCZOS)
    rt_resized = roundtrip_image.resize((h, h), Image.LANCZOS)
    gap = 8
    comparison = Image.new('RGB', (h * 2 + gap, h), (30, 30, 30))
    comparison.paste(orig_resized, (0, 0))
    comparison.paste(rt_resized, (h + gap, 0))
    comparison.save(args.comparison)
    print(f"Comparison saved to {args.comparison}")

    print(f"\n=== Summary ===")
    print(f"Original:         {original.size}")
    print(f"Fuyu grid:        {image_size//patch_size}×{image_size//patch_size} = "
          f"{(image_size//patch_size)**2} patches")
    print(f"Patch dim:        {patch_size}×{patch_size}×3 = {patch_size*patch_size*3}")
    print(f"Token dim:        {weight.shape[0]}")
    print(f"Roundtrip MSE:    {mse:.8f}")
    print(f"Roundtrip cosine: {cos_sim:.6f}")
    print(f"\nIf the roundtrip image is recognisable -> the projection is invertible")
    print(f"If cosine sim > 0.99 -> minimal information loss in the projection")


if __name__ == "__main__":
    main()
