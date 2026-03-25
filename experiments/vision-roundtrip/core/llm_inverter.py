"""LLM Inversion — gradient descent through the full language model.

Optimises vision token embeddings so that the model would produce
a target text description when looking at them. This is the model
"running backwards" from text to image.

The key operation:
  "What would I need to SEE to describe the scene THIS way?"
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class LLMInversionResult:
    """Result of an LLM inversion run."""
    modified_embeddings: torch.Tensor   # (1, n_vision, hidden_dim)
    original_embeddings: torch.Tensor   # (1, n_vision, hidden_dim) — for comparison
    loss_history: List[float]           # CE loss at each step
    best_loss: float                    # minimum CE loss achieved
    per_token_deltas: torch.Tensor      # (n_vision,) — magnitude of change per token
    total_delta: float                  # total embedding shift magnitude


def find_embed_table(model) -> torch.nn.Module:
    """Find the text embedding table in a model."""
    for name, module in model.named_modules():
        if name.endswith('embed_tokens') or name.endswith('wte'):
            return module
    raise ValueError("Could not find embedding table in model")


def get_vision_embeddings(model, inputs, device="cuda") -> torch.Tensor:
    """Extract vision token embeddings from the model's visual encoder."""
    with torch.no_grad():
        if hasattr(model, 'visual'):
            pv_key = None
            for key in ['pixel_values', 'pixel_values_videos']:
                if key in inputs:
                    pv_key = key
                    break
            if pv_key is None:
                raise ValueError("No pixel_values found in inputs")

            vis_out = model.visual(inputs[pv_key])
            if isinstance(vis_out, tuple):
                return vis_out[0]
            return vis_out
        else:
            raise ValueError("Model has no visual encoder")


def llm_inversion(
    model,
    processor,
    inputs: dict,
    vision_mask: torch.Tensor,
    target_description: str,
    n_steps: int = 50,
    lr: float = 0.01,
    reg_weight: float = 0.1,
    tv_weight: float = 0.01,
    device: str = "cuda",
    verbose: bool = True,
) -> LLMInversionResult:
    """
    Gradient descent through the full LLM to modify vision embeddings.

    Optimises vision token embeddings so the model would produce the
    target_description when processing them alongside the text context.

    Args:
        model: the full VLM
        processor: the VLM's processor
        inputs: preprocessed model inputs (from processor)
        vision_mask: (seq_len,) boolean mask of vision token positions
        target_description: the text we want the model to produce
        n_steps: number of gradient descent steps
        lr: learning rate (cosine annealed)
        reg_weight: L2 regularisation weight (keeps near original)
        tv_weight: total variation weight on vision embeddings
        device: computation device
        verbose: print progress

    Returns:
        LLMInversionResult with modified embeddings and diagnostics
    """
    embed_table = find_embed_table(model)

    # Tokenise the target description
    target_ids = processor.tokenizer.encode(
        target_description, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    n_target = target_ids.shape[1]

    if verbose:
        print(f"  Target: {n_target} tokens")

    # Get original vision embeddings
    original_vision = get_vision_embeddings(model, inputs, device)
    n_vision = original_vision.shape[1]

    # Create optimisable copy
    vision_embeds = original_vision.clone().float().requires_grad_(True)
    original_frozen = original_vision.clone().float().detach()

    if verbose:
        print(f"  Optimising {n_vision} vision embeddings "
              f"({vision_embeds.shape[-1]}-dim each)")

    # Get text input embeddings
    with torch.no_grad():
        input_embeds = embed_table(inputs.input_ids).float()
        target_embeds = embed_table(target_ids).float()

    # Optimiser
    optimizer = torch.optim.AdamW([vision_embeds], lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    vision_positions = vision_mask.nonzero(as_tuple=True)[0]

    if verbose:
        print(f"  Running {n_steps} LLM inversion steps (lr={lr})...")

    best_loss = float('inf')
    best_embeds = None
    loss_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Build full sequence: text + vision (optimisable) + target
        full_embeds = input_embeds.clone()
        for i, pos in enumerate(vision_positions):
            if i < vision_embeds.shape[1]:
                full_embeds[0, pos] = vision_embeds[0, i]

        # Append target description
        full_embeds = torch.cat([full_embeds, target_embeds], dim=1)

        # Attention mask
        full_mask = torch.ones(
            1, full_embeds.shape[1],
            device=device, dtype=inputs.attention_mask.dtype
        )

        # Forward pass
        outputs = model(
            inputs_embeds=full_embeds.half(),
            attention_mask=full_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        # Cross-entropy on the target description portion
        target_start = inputs.input_ids.shape[1]
        target_logits = outputs.logits[:, target_start-1:target_start-1+n_target, :]

        ce_loss = F.cross_entropy(
            target_logits.reshape(-1, model.config.vocab_size).float(),
            target_ids.reshape(-1),
        )

        # Regularisation: stay close to original
        reg_loss = F.mse_loss(vision_embeds, original_frozen) * reg_weight

        # Total variation on vision embeddings (spatial smoothness)
        if n_vision > 1:
            tv_loss = torch.mean(
                torch.abs(vision_embeds[:, 1:, :] - vision_embeds[:, :-1, :])
            ) * tv_weight
        else:
            tv_loss = torch.tensor(0.0, device=device)

        total_loss = ce_loss + reg_loss + tv_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Track
        ce_val = ce_loss.item()
        loss_history.append(ce_val)
        if ce_val < best_loss:
            best_loss = ce_val
            best_embeds = vision_embeds.detach().clone()

        if verbose and (step < 5 or step % 10 == 0 or step == n_steps - 1):
            delta = (vision_embeds - original_frozen).norm().item()
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Step {step:3d}: ce={ce_val:.4f} reg={reg_loss.item():.4f} "
                  f"tv={tv_loss.item():.4f} delta={delta:.2f} lr={current_lr:.6f}")

    # Per-token change analysis
    per_token_deltas = (best_embeds - original_frozen).norm(dim=-1).squeeze(0)
    total_delta = per_token_deltas.sum().item()

    if verbose:
        print(f"\n  Best CE loss: {best_loss:.4f}")
        print(f"  Total embedding delta: {total_delta:.2f}")
        top_k = min(5, n_vision)
        top_changed = torch.topk(per_token_deltas, top_k)
        print(f"  Top {top_k} changed tokens:")
        for idx, delta in zip(top_changed.indices.tolist(), top_changed.values.tolist()):
            print(f"    Token {idx}: delta={delta:.4f}")

    return LLMInversionResult(
        modified_embeddings=best_embeds.half(),
        original_embeddings=original_frozen.half(),
        loss_history=loss_history,
        best_loss=best_loss,
        per_token_deltas=per_token_deltas,
        total_delta=total_delta,
    )
