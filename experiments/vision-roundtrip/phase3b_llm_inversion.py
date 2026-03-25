#!/usr/bin/env python3
"""
Phase 3b: LLM-Guided Vision Embedding Modification

Complete loop:
  1. Encode image → vision token embeddings
  2. Forward pass: model describes the image in exhaustive detail
  3. Human (or script) modifies the description
  4. Backward pass: gradient descent through the LLM to find vision
     embeddings that would produce the modified description
  5. Decode modified embeddings → image via ViT inversion + diffusion

The model generates its own "what I see" text, you edit it, and the model
reconstructs the vision embeddings that match the edited description.

Usage:
  # Automatic mode — model describes, then apply a text edit
  python phase3b_llm_inversion.py \
    --model-path D:/models/Qwen3.5-9B-HF \
    --image market_scene.jpg \
    --edit "change 'arm at his side' to 'arm raised above his head'" \
    --output modified.png \
    --steps 50

  # Manual mode — model describes, you provide the full modified description
  python phase3b_llm_inversion.py \
    --model-path D:/models/Qwen3.5-9B-HF \
    --image market_scene.jpg \
    --manual-description "A man stands in a market with his arm raised..." \
    --output modified.png \
    --steps 50

  # Just describe (no modification) — see what the model perceives
  python phase3b_llm_inversion.py \
    --model-path D:/models/Qwen3.5-9B-HF \
    --image market_scene.jpg \
    --describe-only
"""

import argparse
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


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


def load_full_model(model_path, device="cuda"):
    """Load the full VLM for both description generation and gradient computation."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load in float16 but we'll need float32 for gradient computation on vision embeddings
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"  Model loaded: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Vocab size: {model.config.vocab_size}")

    return model, processor


def describe_image(model, processor, image, device="cuda", max_new_tokens=1024):
    """
    Step 1-2: Encode image and generate exhaustive description.
    Returns the description text and the vision token positions.
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Find vision token positions before generation
    image_token_id = getattr(model.config, 'image_token_id', 248056)
    vision_mask = (inputs.input_ids == image_token_id).squeeze(0)
    n_vision = vision_mask.sum().item()
    vision_positions = vision_mask.nonzero(as_tuple=True)[0].tolist()

    print(f"  Input sequence length: {inputs.input_ids.shape[1]}")
    print(f"  Vision tokens: {n_vision} (positions {vision_positions[0]}..{vision_positions[-1]})")

    # Generate description
    print(f"  Generating description (max {max_new_tokens} tokens)...")
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # near-greedy for precise description
            do_sample=False,
        )
    elapsed = time.time() - t0

    # Decode — only the new tokens (strip the input)
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    description = processor.decode(new_tokens, skip_special_tokens=True)

    print(f"  Description generated: {len(new_tokens)} tokens, {elapsed:.1f}s")
    print(f"  ---")
    print(f"  {description[:500]}{'...' if len(description) > 500 else ''}")
    print(f"  ---")

    return description, inputs, vision_mask, n_vision


def apply_text_edit(original_description, edit_instruction):
    """
    Step 3: Apply a text edit to the description.
    Supports simple find/replace via "change 'X' to 'Y'" syntax.
    """
    # Parse "change 'X' to 'Y'" format
    match = re.match(r"change\s+'([^']+)'\s+to\s+'([^']+)'", edit_instruction, re.IGNORECASE)
    if match:
        find_text = match.group(1)
        replace_text = match.group(2)
        if find_text.lower() in original_description.lower():
            # Case-insensitive replace
            pattern = re.compile(re.escape(find_text), re.IGNORECASE)
            modified = pattern.sub(replace_text, original_description)
            print(f"  Applied edit: '{find_text}' → '{replace_text}'")
            return modified
        else:
            print(f"  Warning: '{find_text}' not found in description")
            print(f"  Appending edit instruction as modification context")
            return original_description + f"\n\nModification: {edit_instruction}"

    # Parse "replace 'X' with 'Y'" format
    match = re.match(r"replace\s+'([^']+)'\s+with\s+'([^']+)'", edit_instruction, re.IGNORECASE)
    if match:
        find_text = match.group(1)
        replace_text = match.group(2)
        pattern = re.compile(re.escape(find_text), re.IGNORECASE)
        modified = pattern.sub(replace_text, original_description)
        print(f"  Applied edit: '{find_text}' → '{replace_text}'")
        return modified

    # Generic edit — append as modification instruction
    modified = original_description + f"\n\nModification: {edit_instruction}"
    print(f"  Applied generic edit (appended)")
    return modified


def llm_inversion(model, processor, inputs, vision_mask,
                  target_description, original_vision_embeds=None,
                  n_steps=50, lr=0.01, device="cuda"):
    """
    Step 4: Gradient descent through the LLM.

    Optimise vision token embeddings so that the model would produce
    the target_description when looking at them.

    This is the key operation: we're asking "what would I need to SEE
    to describe the scene this way?"
    """
    # Build the target: the text we want the model to produce
    # This is the modified description tokenised
    target_text = DESCRIBE_PROMPT + "\n\n" + target_description
    target_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": target_description,
        }
    ]

    # Tokenise the target description (the text the model should output)
    target_ids = processor.tokenizer.encode(
        target_description, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    n_target_tokens = target_ids.shape[1]
    print(f"  Target description: {n_target_tokens} tokens")

    # Get the model's embedding table
    if hasattr(model, 'model') and hasattr(model.model, 'wte'):
        embed_table = model.model.wte
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_table = model.model.embed_tokens
    else:
        # Try to find it
        for name, module in model.named_modules():
            if 'embed_tokens' in name or 'wte' in name:
                embed_table = module
                break
        else:
            raise ValueError("Could not find embedding table in model")

    # Build the full input sequence embeddings
    # We need: [system/user prompt tokens] [VISION TOKENS] [assistant prefix] [target description]
    with torch.no_grad():
        # Get embeddings for the text parts of the input
        input_embeds = embed_table(inputs.input_ids)  # (1, seq_len, hidden_size)

        # Extract current vision embeddings from the model's visual encoder
        # Run through the visual model to get the actual vision embeddings
        if hasattr(model, 'visual'):
            # Qwen3.5 style
            if 'pixel_values' in inputs:
                vis_out = model.visual(inputs.pixel_values)
            elif 'pixel_values_videos' in inputs:
                vis_out = model.visual(inputs.pixel_values_videos)
            else:
                print("  Warning: no pixel_values found, using embedding table placeholders")
                vis_out = None
        else:
            vis_out = None

    # The vision embeddings we'll optimise
    if vis_out is not None:
        if isinstance(vis_out, tuple):
            vision_embeds = vis_out[0].clone().float().requires_grad_(True)
        else:
            vision_embeds = vis_out.clone().float().requires_grad_(True)
    elif original_vision_embeds is not None:
        vision_embeds = original_vision_embeds.clone().float().requires_grad_(True)
    else:
        # Fall back to the placeholder embeddings
        vision_embeds = input_embeds[:, vision_mask, :].clone().float().requires_grad_(True)

    n_vision = vision_embeds.shape[1]
    print(f"  Optimising {n_vision} vision embeddings ({vision_embeds.shape})")

    # Embed the target description tokens
    with torch.no_grad():
        target_embeds = embed_table(target_ids).float()  # (1, n_target, hidden_size)

    # Build optimiser
    optimizer = torch.optim.AdamW([vision_embeds], lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    # Regularisation: keep modified embeddings close to original
    with torch.no_grad():
        original_vision = vision_embeds.clone()

    print(f"  Running {n_steps} LLM inversion steps (lr={lr})...")
    best_loss = float('inf')
    best_embeds = None
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Construct the full sequence with current vision embeddings:
        # [text tokens before vision] [OPTIMISABLE VISION] [text tokens after vision] [target description]
        full_embeds = input_embeds.clone().float()

        # Inject current vision embeddings at vision positions
        vision_positions = vision_mask.nonzero(as_tuple=True)[0]
        for i, pos in enumerate(vision_positions):
            if i < vision_embeds.shape[1]:
                full_embeds[0, pos] = vision_embeds[0, i]

        # Append target description embeddings
        full_embeds = torch.cat([full_embeds, target_embeds], dim=1)

        # Build attention mask for the full sequence
        full_attn_mask = torch.ones(
            1, full_embeds.shape[1],
            device=device, dtype=inputs.attention_mask.dtype
        )

        # Forward pass through the model
        outputs = model(
            inputs_embeds=full_embeds.half(),
            attention_mask=full_attn_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        # Loss: cross-entropy on the target description portion
        # The model should predict each target token given the vision + prompt context
        logits = outputs.logits  # (1, full_seq_len, vocab_size)

        # We want the logits at positions just before each target token
        # Target tokens start at position (input_seq_len) in the full sequence
        target_start = inputs.input_ids.shape[1]
        target_logits = logits[:, target_start-1:target_start-1+n_target_tokens, :]

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            target_logits.reshape(-1, model.config.vocab_size).float(),
            target_ids.reshape(-1),
        )

        # Regularisation: don't drift too far from original embeddings
        reg_loss = F.mse_loss(vision_embeds, original_vision) * 0.1

        # Total variation on vision embeddings (spatial smoothness)
        # Assumes vision tokens are roughly in spatial order
        if n_vision > 1:
            tv_loss = torch.mean(
                torch.abs(vision_embeds[:, 1:, :] - vision_embeds[:, :-1, :])
            ) * 0.01
        else:
            tv_loss = torch.tensor(0.0, device=device)

        total_loss = ce_loss + reg_loss + tv_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Track
        losses.append(ce_loss.item())
        if ce_loss.item() < best_loss:
            best_loss = ce_loss.item()
            best_embeds = vision_embeds.detach().clone()

        if step < 5 or step % 10 == 0 or step == n_steps - 1:
            current_lr = scheduler.get_last_lr()[0]
            delta = (vision_embeds - original_vision).norm().item()
            print(f"    Step {step:3d}: ce={ce_loss.item():.4f} reg={reg_loss.item():.4f} "
                  f"tv={tv_loss.item():.4f} delta={delta:.2f} lr={current_lr:.6f}")

    # Report
    print(f"\n  Inversion complete.")
    print(f"  Best CE loss: {best_loss:.4f}")
    print(f"  Final embedding delta from original: {(best_embeds - original_vision).norm().item():.2f}")

    # Per-token change analysis
    per_token_delta = (best_embeds - original_vision).norm(dim=-1).squeeze(0)
    top_changed = torch.topk(per_token_delta, min(10, n_vision))
    print(f"  Top changed vision tokens:")
    for idx, delta in zip(top_changed.indices.tolist(), top_changed.values.tolist()):
        print(f"    Token {idx}: delta={delta:.4f}")

    return best_embeds.half()


def main():
    parser = argparse.ArgumentParser(description="Phase 3b: LLM-guided vision embedding modification")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to VLM (HuggingFace format)")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image")
    parser.add_argument("--edit", type=str, default=None,
                        help="Edit instruction: \"change 'X' to 'Y'\" or free text")
    parser.add_argument("--manual-description", type=str, default=None,
                        help="Provide the full modified description directly")
    parser.add_argument("--describe-only", action="store_true",
                        help="Just describe the image and exit (no modification)")
    parser.add_argument("--output", type=str, default="modified.png",
                        help="Output path for modified image")
    parser.add_argument("--comparison", type=str, default="llm_inversion_comparison.png",
                        help="Comparison output path")
    parser.add_argument("--steps", type=int, default=50,
                        help="LLM inversion gradient steps")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for LLM inversion")
    parser.add_argument("--inversion-steps", type=int, default=200,
                        help="ViT feature inversion steps for final decode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=672,
                        help="Image size for ViT processing")
    parser.add_argument("--save-description", type=str, default=None,
                        help="Save the original description to a text file")
    parser.add_argument("--save-embeddings", type=str, default=None,
                        help="Save original and modified embeddings to .pt file")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    model, processor = load_full_model(args.model_path, device)

    # Load image
    original_image = Image.open(args.image).convert("RGB")
    print(f"\nImage: {args.image} ({original_image.size})")

    # Step 1-2: Describe the image
    print(f"\n=== Step 1-2: Generating exhaustive description ===")
    description, inputs, vision_mask, n_vision = describe_image(
        model, processor, original_image, device
    )

    if args.save_description:
        Path(args.save_description).write_text(description)
        print(f"  Description saved to {args.save_description}")

    if args.describe_only:
        print(f"\n=== Description only mode — done ===")
        print(f"\nFull description:\n{description}")
        return

    # Step 3: Modify the description
    print(f"\n=== Step 3: Modifying description ===")
    if args.manual_description:
        modified_description = args.manual_description
        print(f"  Using manually provided description ({len(modified_description)} chars)")
    elif args.edit:
        modified_description = apply_text_edit(description, args.edit)
    else:
        print("Error: provide --edit or --manual-description (or use --describe-only)")
        return

    # Show the diff
    print(f"\n  Original (first 200 chars): {description[:200]}...")
    print(f"  Modified (first 200 chars): {modified_description[:200]}...")

    # Step 4: LLM inversion — gradient descent through the model
    print(f"\n=== Step 4: LLM inversion ({args.steps} steps) ===")
    modified_embeds = llm_inversion(
        model, processor, inputs, vision_mask,
        modified_description,
        n_steps=args.steps, lr=args.lr, device=device,
    )

    if args.save_embeddings:
        # Also get original embeddings for comparison
        with torch.no_grad():
            image_token_id = getattr(model.config, 'image_token_id', 248056)
            orig_mask = (inputs.input_ids == image_token_id).squeeze(0)
            if hasattr(model, 'visual') and 'pixel_values' in inputs:
                orig_vis = model.visual(inputs.pixel_values)
                if isinstance(orig_vis, tuple):
                    orig_embeds = orig_vis[0]
                else:
                    orig_embeds = orig_vis
            else:
                embed_table = None
                for name, module in model.named_modules():
                    if 'embed_tokens' in name or 'wte' in name:
                        embed_table = module
                        break
                orig_embeds = embed_table(inputs.input_ids)[:, orig_mask, :]

        torch.save({
            'original_embeddings': orig_embeds.cpu(),
            'modified_embeddings': modified_embeds.cpu(),
            'original_description': description,
            'modified_description': modified_description,
            'edit': args.edit,
        }, args.save_embeddings)
        print(f"  Embeddings saved to {args.save_embeddings}")

    # Step 5: Decode modified embeddings back to image
    print(f"\n=== Step 5: Decoding modified embeddings to image ===")
    from phase1_inversion import (
        load_vit_and_mmproj,
        pseudoinverse_mmproj,
        feature_inversion,
        tensor_to_pil,
    )

    vit, mmproj, vit_processor, _ = load_vit_and_mmproj(args.model_path, device)

    # Pseudoinverse the modified embeddings back through mmproj
    print(f"  Pseudoinversing mmproj...")
    recovered_patches = pseudoinverse_mmproj(modified_embeds, mmproj)

    # Feature inversion through ViT
    print(f"  Feature inversion ({args.inversion_steps} steps)...")
    inverted_tensor = feature_inversion(
        recovered_patches, vit, vit_processor,
        n_steps=args.inversion_steps, lr=0.05,
        image_size=args.image_size, device=device,
    )

    # Also decode the original for comparison
    print(f"  Decoding original for comparison...")
    with torch.no_grad():
        if hasattr(model, 'visual') and 'pixel_values' in inputs:
            orig_vis = model.visual(inputs.pixel_values)
            orig_embeds = orig_vis[0] if isinstance(orig_vis, tuple) else orig_vis
        else:
            orig_embeds = modified_embeds  # fallback
    orig_patches = pseudoinverse_mmproj(orig_embeds, mmproj)
    orig_inverted = feature_inversion(
        orig_patches, vit, vit_processor,
        n_steps=args.inversion_steps, lr=0.05,
        image_size=args.image_size, device=device,
    )

    # Save results
    modified_image = tensor_to_pil(inverted_tensor)
    modified_image.save(args.output)
    print(f"\nModified image saved to {args.output}")

    # Four-way comparison: original photo | original inversion | modified inversion
    h = 384
    orig_pil = tensor_to_pil(orig_inverted)
    images = [original_image, orig_pil, modified_image]
    labels = ["Photo", "Encoded→Decoded", "LLM Modified"]
    widths = [int(img.width * h / img.height) for img in images]
    gap = 8
    total_w = sum(widths) + gap * (len(images) - 1)

    combined = Image.new('RGB', (total_w, h), (30, 30, 30))
    x = 0
    for img, w in zip(images, widths):
        combined.paste(img.resize((w, h), Image.LANCZOS), (x, 0))
        x += w + gap
    combined.save(args.comparison)
    print(f"Comparison saved to {args.comparison}")

    print(f"\n=== Complete ===")
    print(f"Original description (first 200 chars):")
    print(f"  {description[:200]}...")
    print(f"Modified description (first 200 chars):")
    print(f"  {modified_description[:200]}...")
    print(f"\nIf the modified image shows the edit ('{args.edit}') → Phase 3b PASS")
    print(f"The difference between 'Encoded→Decoded' and 'LLM Modified' shows")
    print(f"what the LLM inversion changed in the vision embeddings.")


if __name__ == "__main__":
    main()
