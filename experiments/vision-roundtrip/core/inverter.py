"""Feature inversion — gradient descent through a ViT to recover images.

Optimises pixels so that ViT(pixels) ≈ target_embeddings.
Produces a rough image that captures the spatial structure encoded
in the target embeddings.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from PIL import Image


@dataclass
class InversionResult:
    """Result of a feature inversion run."""
    image: torch.Tensor           # (1, 3, H, W) float tensor in [0, 1]
    loss_history: List[float]     # MSE loss at each step
    best_loss: float              # minimum loss achieved
    snapshots: dict = field(default_factory=dict)  # step -> PIL Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) float tensor [0,1] to PIL Image."""
    img = tensor.squeeze(0).cpu().float()
    img = img.clamp(0, 1)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def feature_inversion(
    target_embeddings: torch.Tensor,
    vit: torch.nn.Module,
    n_steps: int = 200,
    lr: float = 0.05,
    image_size: int = 336,
    device: str = "cuda",
    snapshot_steps: Optional[List[int]] = None,
    snapshot_dir: Optional[str] = None,
    tv_weight_initial: float = 0.01,
    tv_weight_final: float = 0.0005,
    cosine_loss_weight: float = 0.3,
    verbose: bool = True,
) -> InversionResult:
    """
    Recover an image by gradient descent through a frozen ViT.

    Optimises random pixels so that ViT(pixels) ≈ target_embeddings.
    Uses MSE + cosine similarity loss with total variation regularisation.

    Args:
        target_embeddings: (batch, n_patches, embed_dim) target to match
        vit: frozen vision transformer (forward only, no grad on its params)
        n_steps: number of optimisation steps
        lr: initial learning rate (cosine annealed to lr * 0.01)
        image_size: output image size (square)
        device: computation device
        snapshot_steps: list of step numbers to save intermediate images
        snapshot_dir: directory for snapshots (required if snapshot_steps given)
        tv_weight_initial: starting TV regularisation weight (high = smooth)
        tv_weight_final: ending TV regularisation weight (low = detailed)
        cosine_loss_weight: weight for cosine similarity loss component
        verbose: print progress

    Returns:
        InversionResult with the best image found
    """
    # Initialise from grey + slight noise
    image_tensor = torch.full(
        (1, 3, image_size, image_size), 0.5,
        device=device, dtype=torch.float32, requires_grad=True
    )
    image_tensor.data += torch.randn_like(image_tensor) * 0.05
    image_tensor.data.clamp_(0, 1)

    # Optimiser with cosine annealing
    optimizer = torch.optim.Adam([image_tensor], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    target = target_embeddings.to(device).float().detach()

    if verbose:
        print(f"Feature inversion: {n_steps} steps, lr={lr} → {lr*0.01:.4f}, "
              f"target shape={target.shape}")

    best_loss = float('inf')
    best_image = None
    loss_history = []
    snapshots = {}

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward through ViT — handle different output formats
        # Try float32 first, fall back to whatever the model wants
        try:
            vit_output = vit(image_tensor)
        except RuntimeError:
            vit_output = vit(image_tensor.half())

        if hasattr(vit_output, 'last_hidden_state'):
            current = vit_output.last_hidden_state.float()
        elif isinstance(vit_output, tuple):
            current = vit_output[0].float()
        else:
            current = vit_output.float()

        # Match sequence lengths if needed
        if current.shape[1] != target.shape[1]:
            min_len = min(current.shape[1], target.shape[1])
            current = current[:, :min_len]
            target_trimmed = target[:, :min_len]
        else:
            target_trimmed = target

        # MSE loss
        mse_loss = F.mse_loss(current, target_trimmed)

        # Cosine similarity loss
        cos_loss = 1.0 - F.cosine_similarity(
            current.reshape(1, -1), target_trimmed.reshape(1, -1)
        ).mean()

        # Total variation — decays over training (smooth early, detailed late)
        tv_progress = step / max(1, n_steps - 1)
        tv_weight = tv_weight_initial + (tv_weight_final - tv_weight_initial) * tv_progress
        tv_loss = (
            torch.mean(torch.abs(image_tensor[:, :, :, :-1] - image_tensor[:, :, :, 1:])) +
            torch.mean(torch.abs(image_tensor[:, :, :-1, :] - image_tensor[:, :, 1:, :]))
        )

        total_loss = mse_loss + cosine_loss_weight * cos_loss + tv_weight * tv_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp to valid pixel range
        image_tensor.data.clamp_(0, 1)

        # Track
        step_loss = mse_loss.item()
        loss_history.append(step_loss)
        if step_loss < best_loss:
            best_loss = step_loss
            best_image = image_tensor.detach().clone()

        # Logging
        if verbose and (step < 10 or step % 25 == 0 or step == n_steps - 1):
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:4d}: mse={step_loss:.6f} cos={cos_loss.item():.4f} "
                  f"tv={tv_loss.item():.4f} lr={current_lr:.5f}")

        # Snapshots
        if snapshot_steps and (step + 1) in snapshot_steps:
            snap_img = tensor_to_pil(image_tensor.detach())
            snapshots[step + 1] = snap_img
            if snapshot_dir:
                snap_path = Path(snapshot_dir) / f"step_{step+1:04d}.png"
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                snap_img.save(snap_path)
                if verbose:
                    print(f"  >>> Snapshot: {snap_path} (mse={step_loss:.6f})")

    # Final report
    if verbose and len(loss_history) > 20:
        early = np.mean(loss_history[:10])
        late = np.mean(loss_history[-10:])
        print(f"\n  Convergence: first 10 avg={early:.6f} → last 10 avg={late:.6f} "
              f"({(1 - late/early)*100:.1f}% improvement)")

    return InversionResult(
        image=best_image,
        loss_history=loss_history,
        best_loss=best_loss,
        snapshots=snapshots,
    )
