"""mmproj forward projection and pseudoinverse.

Handles the linear (or approximately linear) projection between
ViT patch embedding space and LLM embedding space.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def forward_project(
    patch_embeddings: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Project patch embeddings to LLM embedding space.

    Equivalent to nn.Linear forward: output = input @ weight.T + bias

    Args:
        patch_embeddings: (batch, n_patches, in_dim)
        weight: (out_dim, in_dim) — nn.Linear convention
        bias: (out_dim,) or None

    Returns:
        llm_embeddings: (batch, n_patches, out_dim)
    """
    # Compute in the higher precision of the two inputs
    compute_dtype = torch.float32
    result = torch.matmul(
        patch_embeddings.to(compute_dtype),
        weight.to(compute_dtype).T,
    )
    if bias is not None:
        result = result + bias.to(compute_dtype)
    return result.to(patch_embeddings.dtype)


def pseudoinverse(
    llm_embeddings: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Recover patch embeddings from LLM embeddings via pseudoinverse.

    For a linear projection y = x @ W.T + b:
        x_recovered = (y - b) @ pinv(W.T)
                    = (y - b) @ pinv(W).T   [since pinv(A.T) = pinv(A).T]

    When out_dim > in_dim (which is typical: 4096 > 768), the projection
    is injective and the pseudoinverse recovery is exact to numerical precision.

    Args:
        llm_embeddings: (batch, n_patches, out_dim)
        weight: (out_dim, in_dim) — nn.Linear convention
        bias: (out_dim,) or None

    Returns:
        recovered_patches: (batch, n_patches, in_dim)
    """
    # Work in float32 for numerical stability
    y = llm_embeddings.to(torch.float32)
    w = weight.to(torch.float32)

    if bias is not None:
        y = y - bias.to(torch.float32)

    # Pseudoinverse of weight: pinv(W) has shape (in_dim, out_dim)
    # We need pinv(W.T) which equals pinv(W).T
    # So: x = y @ pinv(W.T) = y @ pinv(W).T
    w_pinv = torch.linalg.pinv(w)  # (in_dim, out_dim)
    recovered = torch.matmul(y, w_pinv.T)  # (batch, n_patches, in_dim)

    return recovered.to(llm_embeddings.dtype)


def roundtrip_error(
    patch_embeddings: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> float:
    """
    Measure the roundtrip error: project → pseudoinverse → compare.

    Returns MSE between original and recovered patch embeddings.
    """
    projected = forward_project(patch_embeddings, weight, bias)
    recovered = pseudoinverse(projected, weight, bias)

    # Compare in float32
    mse = F.mse_loss(
        recovered.to(torch.float32),
        patch_embeddings.to(torch.float32),
    ).item()
    return mse
