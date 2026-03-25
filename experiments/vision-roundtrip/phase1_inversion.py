#!/usr/bin/env python3
"""
Phase 1: Vision Token Roundtrip — Encode → Pseudoinverse → Feature Inversion

Tests whether a VLM's vision token embeddings can be inverted back to a
recognisable image. No training required.

Pipeline:
  Image → ViT encoder → patch embeddings → mmproj → LLM embeddings
  LLM embeddings → pseudoinverse mmproj → recovered patch embeddings
  Recovered patch embeddings → feature inversion (gradient descent through ViT)
  → rough image (blobby but spatially correct)

Usage:
  python phase1_inversion.py \
    --model-path /path/to/qwen-vl/gguf/or/hf \
    --image test.jpg \
    --output inverted.png \
    --steps 15 \
    --lr 0.1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def load_vit_and_mmproj(model_path: str, device: str = "cuda"):
    """
    Load the ViT encoder and mmproj projection from a vision-language model.

    Supports HuggingFace format (Qwen-VL, LLaVA, etc.)
    Returns: (vit_encoder, mmproj_weight, image_processor, config)
    """
    # Try Qwen-VL first
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"Loading Qwen-VL model from {model_path}...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        # Extract ViT and projection
        vit = model.visual
        # The projection from vision space to LLM space
        # varies by model — find the right attribute
        mmproj = None
        for name in ['visual_projection', 'mm_projector', 'multi_modal_projector']:
            if hasattr(model, name):
                mmproj = getattr(model, name)
                break

        if mmproj is None:
            # Qwen2-VL merges projection into the visual model
            # The visual model output is already in LLM space
            print("Note: model uses integrated projection (no separate mmproj)")

        return vit, mmproj, processor, model.config

    except Exception as e:
        print(f"Qwen-VL load failed: {e}")

    # Try generic transformers VLM
    try:
        from transformers import AutoModel, AutoProcessor

        print(f"Loading generic VLM from {model_path}...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        # Try to find vision components
        vit = None
        mmproj = None
        for name in ['vision_model', 'vision_tower', 'visual', 'vit']:
            if hasattr(model, name):
                vit = getattr(model, name)
                break
        for name in ['visual_projection', 'mm_projector', 'multi_modal_projector']:
            if hasattr(model, name):
                mmproj = getattr(model, name)
                break

        if vit is None:
            raise ValueError("Could not find vision encoder in model")

        return vit, mmproj, processor, model.config

    except Exception as e:
        print(f"Generic VLM load failed: {e}")
        raise ValueError(f"Could not load any VLM from {model_path}")


def encode_image(image: Image.Image, vit, mmproj, processor, device="cuda"):
    """
    Encode an image to LLM embedding space.
    Returns: (llm_embeddings, patch_embeddings, vit_output)
    """
    # Preprocess
    if hasattr(processor, 'image_processor'):
        pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"]
    else:
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

    pixel_values = pixel_values.to(device=device, dtype=torch.float16)

    # Forward through ViT
    with torch.no_grad():
        vit_output = vit(pixel_values)

        # Extract patch embeddings (before projection)
        if hasattr(vit_output, 'last_hidden_state'):
            patch_embeddings = vit_output.last_hidden_state
        elif isinstance(vit_output, tuple):
            patch_embeddings = vit_output[0]
        else:
            patch_embeddings = vit_output

        # Project to LLM space if mmproj exists
        if mmproj is not None:
            llm_embeddings = mmproj(patch_embeddings)
        else:
            llm_embeddings = patch_embeddings

    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    print(f"LLM embeddings shape:   {llm_embeddings.shape}")

    return llm_embeddings, patch_embeddings


def pseudoinverse_mmproj(llm_embeddings, mmproj):
    """
    Recover patch embeddings from LLM embeddings via pseudoinverse of mmproj.
    If mmproj is a linear layer, this is exact (to numerical precision).
    If mmproj is an MLP, this is approximate.
    """
    if mmproj is None:
        return llm_embeddings

    if isinstance(mmproj, torch.nn.Linear):
        # Simple linear — exact pseudoinverse
        weight = mmproj.weight.data.float()  # (out_dim, in_dim)
        pinv = torch.linalg.pinv(weight)     # (in_dim, out_dim)

        recovered = llm_embeddings.float() @ pinv.T

        if mmproj.bias is not None:
            # Subtract bias before pseudoinverse
            recovered = (llm_embeddings.float() - mmproj.bias.data.float()) @ pinv.T

        print(f"Pseudoinverse recovery — input: {llm_embeddings.shape}, output: {recovered.shape}")
        return recovered.half()
    else:
        # MLP or complex projection — can't pseudoinverse directly
        # Fall back to optimisation
        print("Warning: mmproj is not a simple linear layer. Pseudoinverse is approximate.")
        print(f"mmproj type: {type(mmproj)}")

        # Try to extract the first linear layer for approximate inversion
        for module in mmproj.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data.float()
                pinv = torch.linalg.pinv(weight)
                recovered = llm_embeddings.float() @ pinv.T
                return recovered.half()

        # Give up — return the embeddings as-is
        print("Could not pseudoinverse mmproj. Using LLM embeddings directly for feature inversion.")
        return llm_embeddings


def feature_inversion(target_embeddings, vit, processor,
                      n_steps=200, lr=0.05, image_size=336, device="cuda",
                      snapshot_steps=None, snapshot_dir=None):
    """
    Recover an image by gradient descent through the frozen ViT.

    Optimise random pixels so that ViT(pixels) ≈ target_embeddings.
    More steps = better quality. 15 steps = blobby layout. 200+ = detailed.

    Args:
        snapshot_steps: list of step numbers to save intermediate images
        snapshot_dir: directory to save snapshots (if snapshot_steps provided)
    """
    # Start from grey with slight noise
    image_tensor = torch.full(
        (1, 3, image_size, image_size), 0.5,
        device=device, dtype=torch.float32, requires_grad=True
    )
    image_tensor.data += torch.randn_like(image_tensor) * 0.05

    # Use Adam with cosine annealing for better convergence at high step counts
    optimizer = torch.optim.Adam([image_tensor], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    target = target_embeddings.float().detach()

    # TV regularisation weight — start higher (encourages structure),
    # decay over time (allows detail to emerge)
    tv_weight_initial = 0.01
    tv_weight_final = 0.0005

    print(f"Feature inversion: {n_steps} steps, lr={lr} (cosine→{lr*0.01:.4f}), target shape={target.shape}")
    if snapshot_steps:
        print(f"  Snapshots at steps: {snapshot_steps}")

    best_loss = float('inf')
    best_image = None
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward through ViT
        vit_output = vit(image_tensor.half())
        if hasattr(vit_output, 'last_hidden_state'):
            current = vit_output.last_hidden_state.float()
        elif isinstance(vit_output, tuple):
            current = vit_output[0].float()
        else:
            current = vit_output.float()

        # Match shape if needed (target might include/exclude CLS token)
        if current.shape[1] != target.shape[1]:
            min_len = min(current.shape[1], target.shape[1])
            current = current[:, :min_len]
            target_trimmed = target[:, :min_len]
        else:
            target_trimmed = target

        # MSE loss in embedding space
        loss = F.mse_loss(current, target_trimmed)

        # Cosine similarity loss — encourages directional alignment, not just magnitude
        cos_loss = 1.0 - F.cosine_similarity(
            current.reshape(1, -1), target_trimmed.reshape(1, -1)
        ).mean()

        # Total variation regularisation — decays over training
        tv_progress = step / max(1, n_steps - 1)
        tv_weight = tv_weight_initial + (tv_weight_final - tv_weight_initial) * tv_progress
        tv_loss = (
            torch.mean(torch.abs(image_tensor[:, :, :, :-1] - image_tensor[:, :, :, 1:])) +
            torch.mean(torch.abs(image_tensor[:, :, :-1, :] - image_tensor[:, :, 1:, :]))
        )

        total_loss = loss + 0.3 * cos_loss + tv_weight * tv_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp to valid pixel range
        image_tensor.data.clamp_(0, 1)

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_image = image_tensor.detach().clone()

        losses.append(loss.item())

        # Logging
        if step < 10 or step % 25 == 0 or step == n_steps - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:4d}: mse={loss.item():.6f} cos={cos_loss.item():.4f} "
                  f"tv={tv_loss.item():.4f} tv_w={tv_weight:.4f} lr={current_lr:.5f}")

        # Save snapshots
        if snapshot_steps and snapshot_dir and (step + 1) in snapshot_steps:
            snap_path = Path(snapshot_dir) / f"step_{step+1:04d}.png"
            snap_img = tensor_to_pil(image_tensor.detach())
            snap_img.save(snap_path)
            print(f"  >>> Snapshot saved: {snap_path} (loss={loss.item():.6f})")

    # Report convergence
    print(f"\n  Final MSE: {losses[-1]:.6f}")
    print(f"  Best MSE:  {best_loss:.6f}")
    if len(losses) > 50:
        early = np.mean(losses[:10])
        late = np.mean(losses[-10:])
        print(f"  Convergence: first 10 avg={early:.6f} → last 10 avg={late:.6f} "
              f"({(1 - late/early)*100:.1f}% improvement)")

    return best_image


def tensor_to_pil(tensor):
    """Convert a (1, 3, H, W) float tensor [0,1] to PIL Image."""
    img = tensor.squeeze(0).cpu().float()
    img = img.clamp(0, 1)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def compare_images(original: Image.Image, inverted: Image.Image, output_path: str):
    """Save a side-by-side comparison."""
    # Resize to same height
    h = min(original.height, inverted.height, 512)
    w_orig = int(original.width * h / original.height)
    w_inv = int(inverted.width * h / inverted.height)

    orig_resized = original.resize((w_orig, h), Image.LANCZOS)
    inv_resized = inverted.resize((w_inv, h), Image.LANCZOS)

    # Side by side with label space
    gap = 20
    combined = Image.new('RGB', (w_orig + w_inv + gap, h), (40, 40, 40))
    combined.paste(orig_resized, (0, 0))
    combined.paste(inv_resized, (w_orig + gap, 0))

    combined.save(output_path)
    print(f"Comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Vision token roundtrip inversion")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to VLM (HuggingFace format)")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image to encode and invert")
    parser.add_argument("--output", type=str, default="inverted.png",
                        help="Output path for inverted image")
    parser.add_argument("--comparison", type=str, default="comparison.png",
                        help="Output path for side-by-side comparison")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of feature inversion gradient steps")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate for feature inversion")
    parser.add_argument("--sweep", action="store_true",
                        help="Run a sweep: save snapshots at 10, 25, 50, 100, 200, 500 steps")
    parser.add_argument("--image-size", type=int, default=336,
                        help="Image size for ViT input")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--save-embeddings", type=str, default=None,
                        help="Optionally save LLM embeddings to .pt file")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    vit, mmproj, processor, config = load_vit_and_mmproj(args.model_path, device)

    # Load and encode image
    print(f"\nEncoding {args.image}...")
    original = Image.open(args.image).convert("RGB")
    llm_embeddings, patch_embeddings = encode_image(original, vit, mmproj, processor, device)

    if args.save_embeddings:
        torch.save({
            'llm_embeddings': llm_embeddings.cpu(),
            'patch_embeddings': patch_embeddings.cpu(),
        }, args.save_embeddings)
        print(f"Embeddings saved to {args.save_embeddings}")

    # Pseudoinverse mmproj
    print(f"\nPseudoinversing mmproj...")
    recovered_patches = pseudoinverse_mmproj(llm_embeddings, mmproj)

    # Setup snapshot directory for sweep mode
    snapshot_dir = None
    snapshot_steps = None
    if args.sweep:
        output_path = Path(args.output)
        snapshot_dir = output_path.parent / f"{output_path.stem}_sweep"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot_steps = [10, 25, 50, 100, 150, 200, 300, 500]
        # Run enough steps to cover the sweep
        args.steps = max(args.steps, max(snapshot_steps))
        print(f"\nSweep mode: running {args.steps} steps, saving snapshots at {snapshot_steps}")
        print(f"Snapshot directory: {snapshot_dir}")

    # Feature inversion
    print(f"\nRunning feature inversion ({args.steps} steps)...")
    inverted_tensor = feature_inversion(
        recovered_patches, vit, processor,
        n_steps=args.steps, lr=args.lr,
        image_size=args.image_size, device=device,
        snapshot_steps=snapshot_steps, snapshot_dir=snapshot_dir,
    )

    # Save results
    inverted_image = tensor_to_pil(inverted_tensor)
    inverted_image.save(args.output)
    print(f"\nInverted image saved to {args.output}")

    # Save comparison
    compare_images(original, inverted_image, args.comparison)

    # If sweep mode, build a grid of all snapshots
    if args.sweep and snapshot_dir:
        print(f"\nBuilding sweep comparison grid...")
        snap_files = sorted(snapshot_dir.glob("step_*.png"))
        if snap_files:
            h = 256
            images_row = [original.resize((h, h), Image.LANCZOS)]
            labels = ["Original"]
            for sf in snap_files:
                step_num = sf.stem.split("_")[1]
                img = Image.open(sf).resize((h, h), Image.LANCZOS)
                images_row.append(img)
                labels.append(f"Step {int(step_num)}")

            gap = 4
            total_w = len(images_row) * h + (len(images_row) - 1) * gap
            grid = Image.new('RGB', (total_w, h), (40, 40, 40))
            x = 0
            for img in images_row:
                grid.paste(img, (x, 0))
                x += h + gap
            grid_path = snapshot_dir / "sweep_grid.png"
            grid.save(grid_path)
            print(f"Sweep grid saved to {grid_path}")

    # Report
    print(f"\n=== Results ===")
    print(f"Original:        {original.size}")
    print(f"Inverted:        {inverted_image.size}")
    print(f"LLM embeddings:  {llm_embeddings.shape}")
    print(f"Patch embeddings:{patch_embeddings.shape}")
    print(f"Steps:           {args.steps}")
    print(f"Learning rate:   {args.lr} (cosine annealed)")
    print(f"\nNext steps:")
    print(f"  - If the inverted image shows recognisable spatial layout → Phase 1 PASS")
    print(f"  - Feed inverted image to diffusion img2img for Phase 2")
    print(f"  - Load embeddings with --save-embeddings for Phase 3 WIR modification")
    if args.sweep:
        print(f"  - Check {snapshot_dir}/sweep_grid.png to see convergence progression")


if __name__ == "__main__":
    main()
