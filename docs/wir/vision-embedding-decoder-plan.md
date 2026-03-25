# Vision Embedding Decoder — Training Plan

## Inverse Pipeline: LLM Vision Token Embeddings → Images

**Status:** Plan v0.1 — DRAFT
**Author:** Ed Holmes (Cranborne Audio) / Ren
**Date:** March 2026
**Depends on:** WIR Tiered Model Architecture (section 9.7)

---

## 1. The Problem

The forward pipeline exists: Image → ViT encoder → patch embeddings → mmproj → LLM embedding space. Every vision-language model does this. The LLM reasons about images as sequences of embedding vectors in its native space.

The inverse pipeline does not exist off-the-shelf: LLM embedding vectors → image. We need this for:

- Rendering NPC subjective perception (WIR-refined vision tokens → visible image)
- Rendering memories (stored embedding snapshots → recalled scenes)
- Rendering imagination (model-generated embeddings → novel images)
- Debugging (inspecting what the model "sees" in embedding space)

---

## 2. Options Analysis

### 2.1 Option A: Train a Dedicated Embedding Decoder (Recommended)

Train a small network that maps LLM vision token embeddings back to images. This is the inverse of the ViT encoder + mmproj projection.

**Architecture:**
```
LLM embedding vectors     Decoder network       Output
(256 × n_embd)        →   (trained inverse)  →  (256×256 or 512×512 image)
```

The decoder is structurally similar to the ViT encoder but reversed:
- Input: sequence of embedding vectors (same shape as what the LLM produces)
- Unprojection: linear layer mapping from LLM embed dim back to ViT patch dim
- Transformer decoder blocks (4-8 layers, much smaller than the LLM)
- Patch-to-pixel: reshape patch embeddings back to spatial grid, upsample
- Output: RGB image

**Parameters:** ~50-200M depending on output resolution. Small enough to train on a single GPU.

**Training data:** Self-supervised from the existing ViT encoder. No external data collection needed.

**Training procedure:**
```
For each training image:
  1. Encode: image → ViT encoder → patch embeddings → mmproj → LLM embeddings
  2. Decode: LLM embeddings → decoder → reconstructed image
  3. Loss: MSE(original_image, reconstructed_image) + perceptual loss (LPIPS)
  4. Backprop through decoder only (encoder + mmproj frozen)
```

This is a standard autoencoder training loop. The encoder is frozen (it's the existing vision model). Only the decoder learns. The decoder learns to invert whatever the encoder does.

**Advantages:**
- Exact inverse of the actual encoding pipeline — reconstructions are faithful
- Fast inference (~10-50ms per image)
- Small model, trains quickly
- Works with any image the encoder can process

**Disadvantages:**
- Reconstructions will be blurry at high frequencies (standard autoencoder issue)
- Needs a perceptual loss to avoid mean-face averaging
- Produces "good enough" images, not photorealistic

### 2.2 Option B: Diffusion Model with Embedding Conditioning

Use an existing diffusion model (Stable Diffusion, Flux) conditioned on the LLM vision embeddings via an adapter (similar to IP-Adapter).

**Architecture:**
```
LLM embedding vectors     Adapter          Diffusion model     Output
(256 × n_embd)        →   (small MLP)  →  (cross-attention) → (high-res image)
```

**Training procedure:**
```
For each training image:
  1. Encode: image → ViT → mmproj → LLM embeddings (frozen)
  2. Add noise to image (standard diffusion schedule)
  3. Diffusion model denoises, conditioned on LLM embeddings via cross-attention
  4. Loss: standard diffusion loss (predict noise)
  5. Backprop through adapter + diffusion attention layers
     (base diffusion weights can be frozen or LoRA-tuned)
```

This is essentially IP-Adapter training but conditioned on LLM embeddings instead of CLIP embeddings. The adapter is a small projection network (~10-50M params) that maps LLM embeddings into the diffusion model's cross-attention space.

**Advantages:**
- Photorealistic output (diffusion model quality)
- Handles high-frequency detail naturally
- Can combine with text prompts for additional control
- Existing IP-Adapter code can be adapted

**Disadvantages:**
- Slower inference (20-50 diffusion steps, 2-10 seconds per image)
- Larger training compute (diffusion training is expensive)
- May not faithfully reconstruct — diffusion "interprets" rather than "inverts"
- The subjective reinterpretation might actually be a feature for NPC imagination

### 2.3 Option C: Hybrid (Decoder + Diffusion Refinement)

Train the fast decoder (Option A) for spatial layout, then use diffusion img2img to add photorealistic detail.

```
LLM embeddings → Decoder (10ms) → rough image → Diffusion img2img (2s) → final image
                                   (spatial layout)    (photorealistic detail)
```

**Advantages:**
- Fast path available when speed matters (decoder only, ~10ms)
- High quality path available when quality matters (decoder + diffusion)
- The decoder provides a strong init image for diffusion, reducing required steps
- Decoder can run every tick for debugging; diffusion runs on demand

**Disadvantages:**
- Two models to train and maintain
- The decoder's output quality needs to be good enough as a diffusion init

### 2.4 Recommendation

**Start with Option A (decoder only), extend to Option C (hybrid) later.**

The decoder is fast to train, fast to infer, and good enough for most use cases. It validates the entire pipeline (can we go from embeddings back to recognisable images?). Once that works, adding diffusion refinement is straightforward — it's just img2img on the decoder's output.

---

## 2.5 Option D: Zero-Training Inverse Pipeline (Try This First)

Before training anything, validate whether the embedding space can be inverted directly.

**Step 1: Pseudoinverse the mmproj (free, exact)**

The mmproj is a linear projection: `llm_embd = patch_embd @ mmproj_weight`. The pseudoinverse recovers the patch embeddings:

```python
patch_embd = llm_embeddings @ torch.linalg.pinv(mmproj_weight)
```

This is exact to numerical precision. No training, no approximation. Gets you from LLM embedding space back to ViT patch embedding space instantly.

**Step 2: Feature inversion through ViT (10-20 gradient steps)**

Optimise a random image so that when encoded by the ViT, it produces embeddings close to the target patch embeddings:

```python
def feature_inversion(target_patch_embd, vit_encoder, n_steps=15, lr=0.1):
    """Recover a rough image from ViT patch embeddings. No training needed."""
    # Start from grey or noise
    image = torch.randn(1, 3, 336, 336, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr)

    for step in range(n_steps):
        encoded = vit_encoder(image)  # forward through frozen ViT
        loss = F.mse_loss(encoded, target_patch_embd)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        image.data.clamp_(0, 1)  # keep pixel values valid

    return image.detach()
```

15 steps × ~10ms per ViT forward/backward = **~150ms**. The output is blobby — correct spatial layout, correct colours roughly, but no fine detail. That's fine. It's a seed image.

**Step 3: Diffusion img2img (the pretty-maker)**

Hand the blobby feature inversion output to a diffusion model as an init image:

```python
# Denoising strength 0.5-0.7:
#   0.5 = preserve layout strongly, add moderate detail
#   0.7 = preserve layout loosely, add significant detail
output = diffusion.img2img(
    init_image=blobby_inversion,
    prompt="scene description from NPC's narrative context",
    denoising_strength=0.6,
    steps=20
)
```

The diffusion model does what it's good at: adding texture, coherent detail, lighting, and photorealism. The feature inversion just told it where things go.

**Total pipeline: ~150ms (inversion) + ~2s (diffusion) = ~2.2s. Zero training.**

```
LLM embeddings
    → pseudoinverse mmproj (free, instant)
    → ViT patch embeddings
    → feature inversion (15 gradient steps, ~150ms)
    → blobby rough image (spatial layout correct, no detail)
    → diffusion img2img (20 steps at denoising 0.6, ~2s)
    → photorealistic image matching the NPC's subjective vision
```

**This should be Phase 0.** If the blobby inversions are spatially recognisable — objects in the right places, sky at the top, ground at the bottom, figures where figures should be — then the trained decoder (Option A) may not be needed at all. The feature inversion + diffusion pipeline gives you the inverse path with zero training cost.

If feature inversion produces garbage (embedding space too nonlinear, ViT too lossy), fall back to training the decoder. But try the free option first.

---

## 3. Training the Embedding Decoder (If Needed)

### 3.1 Data Pipeline

No external dataset collection needed. The training data is generated from the existing ViT encoder:

```
Data generation (offline, one-time):

  Source images:
    - LAION subset (~100K-500K images, diverse scenes)
    - Game engine screenshots (if targeting game aesthetics)
    - Any image dataset that covers the deployment's visual domain

  For each image:
    1. Resize/crop to ViT input size (e.g., 336×336)
    2. Run through ViT encoder → patch embeddings
    3. Run through mmproj → LLM embedding vectors
    4. Save pair: (LLM_embeddings, original_image)

  Output: dataset of (embedding_tensor, image) pairs
```

The ViT encoder and mmproj are frozen components from the target vision-language model (e.g., Qwen-VL's CLIP encoder + projection). We're training the decoder to invert these specific components.

### 3.2 Decoder Architecture

```python
class EmbeddingDecoder(nn.Module):
    """
    Maps LLM vision token embeddings back to images.
    Inverse of: image → ViT_encoder → mmproj → LLM_embeddings
    This model learns: LLM_embeddings → image
    """

    def __init__(self, n_embd=3584, n_patches=256, patch_size=14,
                 image_size=336, n_decoder_layers=6, n_heads=16):
        super().__init__()

        # Unprojection: LLM embed dim → ViT patch dim
        self.unproject = nn.Linear(n_embd, n_embd // 2)

        # Learnable position embeddings for spatial ordering
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, n_embd // 2))

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(n_embd // 2, n_heads)
            for _ in range(n_decoder_layers)
        ])

        # Patch-to-pixel: project each patch embedding to pixel values
        self.patch_to_pixels = nn.Linear(
            n_embd // 2,
            patch_size * patch_size * 3  # RGB pixels per patch
        )

        # Upsample if needed
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()  # output in [0, 1]
        )

        self.patch_size = patch_size
        self.grid_size = int(n_patches ** 0.5)  # e.g., 16 for 256 patches

    def forward(self, embeddings):
        # embeddings: (batch, n_patches, n_embd)

        x = self.unproject(embeddings)       # (batch, n_patches, n_embd//2)
        x = x + self.pos_embed               # add spatial positions

        for block in self.decoder_blocks:
            x = block(x)                     # self-attention across patches

        # Project each patch to pixels
        pixels = self.patch_to_pixels(x)     # (batch, n_patches, patch_px*3)

        # Reshape to spatial grid
        pixels = pixels.view(-1, self.grid_size, self.grid_size,
                            self.patch_size, self.patch_size, 3)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4).contiguous()
        pixels = pixels.view(-1, 3,
                            self.grid_size * self.patch_size,
                            self.grid_size * self.patch_size)

        # Optional upsample for higher resolution
        output = self.upsample(pixels)

        return output
```

**Parameter count estimate:**
- Unprojection: 3584 × 1792 = ~6.4M
- Position embeddings: 256 × 1792 = ~0.5M
- 6 decoder blocks at ~12M each = ~72M
- Patch-to-pixel: 1792 × (14×14×3) = ~1M
- Upsample convolutions: ~0.5M
- **Total: ~80M parameters**

Small enough to train on a single RTX 5090 in a few hours.

### 3.3 Loss Function

```python
def decoder_loss(original_image, reconstructed_image, lpips_model):
    # Pixel-level reconstruction
    mse = F.mse_loss(reconstructed_image, original_image)

    # Perceptual loss — prevents blurry averaging
    perceptual = lpips_model(reconstructed_image, original_image).mean()

    # Optional: adversarial loss for sharper outputs (adds a discriminator)
    # adv = discriminator_loss(reconstructed_image)

    return mse + 0.5 * perceptual  # + 0.1 * adv
```

The perceptual loss (LPIPS) is critical. Without it, the decoder learns the mean of all possible images for each embedding — producing blurry, averaged outputs. LPIPS compares feature maps from a pretrained network, penalising outputs that are perceptually different from the target even if pixel-level MSE is low.

### 3.4 Training Schedule

```
Phase 1: Warm-up (MSE only)
  Epochs: 5-10
  LR: 1e-4 → 1e-5 (cosine schedule)
  Loss: MSE only
  Purpose: Get spatial layout roughly right

Phase 2: Perceptual refinement
  Epochs: 10-20
  LR: 1e-5 → 1e-6
  Loss: MSE + 0.5 * LPIPS
  Purpose: Sharpen details, fix perceptual quality

Phase 3: Fine-tune on domain (optional)
  Epochs: 5-10
  LR: 1e-6
  Data: Game engine screenshots (if targeting game deployment)
  Loss: MSE + 0.5 * LPIPS
  Purpose: Adapt to deployment's visual style
```

**Estimated training time on RTX 5090 (32GB):**
- 100K images, batch size 32, 30 epochs total
- Forward pass per batch: ~50ms (ViT encode) + ~20ms (decoder)
- ~100K / 32 = 3,125 steps per epoch × 30 epochs = ~94K steps
- ~94K × 70ms = ~6,500 seconds ≈ **~2 hours**

### 3.5 Validation

**Quantitative:**
- PSNR (peak signal-to-noise ratio) — target: >25dB
- SSIM (structural similarity) — target: >0.85
- LPIPS (perceptual similarity) — target: <0.15
- FID (Fréchet inception distance on a test set) — target: <30

**Qualitative:**
- Reconstruct 50 test images through encode → LLM embeddings → decode
- Visual inspection: are spatial layouts preserved? Colours? Object positions?
- Edge case: what happens with unusual/out-of-distribution images?

**Wirstate-specific validation:**
- Encode a scene → WIR-refine the embeddings with emotional colouring → decode
- Does the decoded image reflect the emotional modification?
- Compare: original scene vs fearful-NPC version vs content-NPC version
- The modifications should be visible but the scene should remain recognisable

---

## 4. Training the Diffusion Adapter (Phase 2)

Once the decoder validates that embedding → image is feasible, train a diffusion adapter for higher quality output.

### 4.1 IP-Adapter Style Training

The adapter maps LLM vision embeddings into the diffusion model's cross-attention space:

```
Architecture:
  LLM embeddings (256 × n_embd)
      ↓
  Adapter MLP (2 layers, ~20M params)
      ↓
  Conditioning vectors (256 × cross_attn_dim)
      ↓
  Injected into diffusion model's cross-attention layers
  (alongside or replacing CLIP text conditioning)
```

### 4.2 Training Data

Same dataset as the decoder, but now the target is the diffusion training objective:

```
For each training image:
  1. Encode: image → ViT → mmproj → LLM embeddings (all frozen)
  2. Noise: add noise to image at random timestep t
  3. Forward: diffusion model predicts noise, conditioned on LLM embeddings
     via cross-attention through the adapter
  4. Loss: MSE on predicted vs actual noise (standard diffusion loss)
  5. Backprop: through adapter layers only (diffusion base weights frozen,
     or LoRA-tuned for efficiency)
```

### 4.3 Training Cost

- Adapter: ~20M params (trainable)
- Diffusion LoRA: ~10M params (optional, for style adaptation)
- Training: ~8-12 hours on RTX 5090 (diffusion training is slower)
- Inference: 20-50 steps × ~50ms per step = 1-2.5 seconds per image

### 4.4 Modified Embedding Handling

The key question: does the adapter produce good images from **modified** embeddings (WIR-refined, emotionally coloured) when it was trained on **clean** embeddings (direct ViT encodes)?

Expected: yes, if the modifications are moderate. The adapter learns the structure of the embedding space, not the exact values. Small shifts (emotional colouring, attention focus) produce corresponding small shifts in the output image. Large modifications (complete re-imagination) may produce artifacts.

**Validation:**
- Train on clean embeddings
- Test with WIR-modified embeddings (±5%, ±10%, ±20% perturbation)
- Measure: at what perturbation level does image quality degrade?
- If degradation is too early: fine-tune the adapter on a small set of
  (modified_embedding, intended_image) pairs to teach it the modification space

---

## 5. Integration with Wirstate Pipeline

### 5.1 Encode Path (Perception → Storage)

```
Game engine render → ViT encode (frozen) → mmproj (frozen) → LLM embeddings
  → store as memory visual component (section 10.1)
  → inject as vision tokens in NPC's cascade context

Cost: ~50ms per image encode
When: every perception tick, or on significant scene change
```

### 5.2 Decode Path (Imagination → Render)

```
Fast path (decoder only):
  LLM embeddings (original or WIR-modified) → decoder → image
  Cost: ~10-50ms
  Use: debugging, quick visualisation, every-tick perception monitoring

Quality path (decoder + diffusion):
  LLM embeddings → decoder → rough image → diffusion img2img → final image
  Cost: ~2-5 seconds
  Use: on-demand rendering, memory visualisation, cutscenes, player-facing
```

### 5.3 Re-Imagination Path (Memory Modification)

```
Load stored embeddings from memory
  → inject emotional/narrative context as text tokens
  → WIR soft refinement: embeddings shift in LLM embedding space
  → n-least-conf identifies most uncertain spatial regions
  → multiple passes: each pass shifts uncertain regions toward
    the model's subjective version
  → decode modified embeddings to image

Cost: 3-5 WIR passes × ~80ms each + decode = ~0.5-1s (fast) or ~3-5s (quality)
Use: NPC re-imagining a memory, "what if" visualisation, dream sequences
```

---

## 6. Phased Implementation

### Phase 0: Zero-Training Inverse Validation (1-2 days)
- Load a vision-language model's ViT encoder and mmproj weights
- Encode 20 test images → LLM embeddings
- Pseudoinverse the mmproj → recover ViT patch embeddings
- Feature inversion: 15 gradient steps through the ViT → blobby rough images
- Inspect: are the rough images spatially recognisable? Objects in right places?
- If yes: the embedding space is invertible. Skip to Phase 2.
- If no: fall back to training a decoder (Phase 1).
- **Exit criteria:** Feature inversions show correct spatial layout — sky at top, ground at bottom, objects in approximately correct positions. Detail and texture don't matter (diffusion handles that).

### Phase 1: Train Decoder — ONLY IF Phase 0 fails (1-2 weeks)
- Build data pipeline: images → ViT encode → save (embedding, image) pairs
- Train decoder (80M params, ~2 hours on 5090)
- Validate: reconstruct test images, measure PSNR/SSIM/LPIPS
- **Exit criteria:** Reconstructions are recognisable, spatial layout preserved
- **Skip this phase entirely** if feature inversion produces usable rough images

### Phase 2: Diffusion Refinement Pipeline (3-5 days)
- Take rough images from feature inversion (or decoder if Phase 1 was needed)
- Feed as init image to diffusion img2img at denoising 0.5-0.7
- Text prompt from NPC narrative context for additional conditioning
- Validate: are final images photorealistic and spatially faithful to the embeddings?
- Test with WIR-modified embeddings: perturb, invert, diffuse. Do modifications show?
- **Exit criteria:** Blobby inversions + diffusion produces photorealistic images that faithfully reflect the embedding content, including modifications

### Phase 3: WIR Integration (1-2 weeks)
- Wire the inverse pipeline (inversion + diffusion) into the wirstate system
- Encode game scene → store as vision tokens → WIR soft-refine with NPC context → invert → diffuse
- Test the perception → imagination loop end-to-end
- Test emotional colouring: same scene refined with different emotional context → different images
- **Exit criteria:** NPC's subjective render visibly differs from objective scene in ways that correspond to its emotional/cognitive state

### Phase 3b: LLM Inversion for Geometric Editing (2-3 weeks)

LLM inversion is an alternative to (and complement of) the feature inversion + diffusion pipeline above. Instead of decoding embeddings directly, it uses gradient descent through the full LLM to find vision token embeddings that would make the model produce a modified text description of the scene. See tiered-model-architecture.md section 9.9 for the full specification.

**Why this is a separate phase:** The ViT-only inversion pipeline (Phase 0/2) only needs the small vision encoder in memory (~600MB) for backpropagation. LLM inversion requires the full language model in memory or streaming from NVMe for backprop — the gradient must flow through all transformer layers. This is a fundamentally different resource profile.

**Implementation steps:**
- Implement the exhaustive description prompt and description generation pipeline
- Implement gradient descent loop: optimise vision token embeddings against modified description, with L2 regularisation to original embeddings
- Test on 9B model first (fast iteration, ~10s for 40 inversion steps on RTX 5090)
- Validate geometric edits: change poses, move objects, alter spatial relationships in the description and verify the decoded image reflects the changes
- Validate that regularisation preserves unedited regions of the image
- Compare output quality: LLM inversion vs hidden state blending (section 9.7.2) for the same edits
- Test composition: LLM inversion for geometry, then WIR blending for emotional colouring
- Scale to 27B model for quality comparison
- **Exit criteria:** Geometric changes (arm positions, object placement, spatial rearrangement) are clearly reflected in the decoded image. Unedited regions remain stable. The approach composes with WIR emotional colouring.

**Resource comparison:**

| Approach | Model needed for backprop | VRAM for backprop | Time per image |
|---|---|---|---|
| ViT feature inversion (Phase 0) | ViT encoder only (~300M params) | ~1-2GB | ~150ms |
| LLM inversion (Phase 3b) | Full LLM (9B-397B params) | 16-32GB (or NVMe streaming) | 10-25s |

The two approaches are complementary: ViT inversion handles the spatial decoding (embeddings to rough image), while LLM inversion handles semantic editing (changing what the embeddings represent before decoding). A typical edit pipeline chains them: LLM inversion modifies the embeddings, then ViT inversion + diffusion decodes the modified embeddings to an image.

### Phase 4: Memory Integration (1 week)
- Store embedding snapshots as memory visual components
- Recall: load stored embeddings → invert → diffuse → "remembered" image
- Re-imagine: WIR soft-refine stored embeddings with new context → invert → diffuse
- Multi-pass re-imagination: n-least-conf over embedding vectors, 3-5 passes
- **Exit criteria:** NPC can recall a scene, re-imagine it with different emotional colouring, and produce a visually different but spatially consistent image

### Phase 5: Diffusion Adapter — quality upgrade (2-3 weeks, optional)
- Train IP-Adapter style conditioning on LLM embeddings (if diffusion img2img quality is insufficient)
- Direct embedding → diffusion conditioning without the feature inversion step
- Compare quality: inversion+diffusion vs direct adapter conditioning
- **Exit criteria:** Photorealistic subjective renders that preserve WIR modifications better than the inversion pipeline
- **Exit criteria:** NPC can "remember" and "re-imagine" scenes with emotional colouring

---

## 7. Hardware Requirements

| Component | VRAM | Disk | Notes |
|---|---|---|---|
| ViT encoder (Qwen-VL CLIP) | ~600MB | ~600MB | Frozen, loaded once |
| mmproj | ~50MB | ~50MB | Frozen, loaded once |
| Embedding decoder | ~320MB | ~320MB | 80M params in FP32 |
| Diffusion model (SDXL/Flux) | ~6-12GB | ~6-12GB | Only for quality path |
| Diffusion adapter | ~80MB | ~80MB | 20M params |
| Training: decoder | ~4GB peak | ~20GB dataset | 2 hours on 5090 |
| Training: adapter | ~16GB peak | ~20GB dataset | 8-12 hours on 5090 |

The decoder alone fits comfortably alongside the LLM in VRAM. The diffusion model is only loaded when quality rendering is needed — not resident during normal cascade ticks.

---

## 8. Open Questions

1. **Which ViT encoder to target?** Qwen-VL uses InternViT-300M. LLaVA uses CLIP ViT-L. Different models use different encoders with different embedding spaces. The decoder must be trained per encoder. Start with whichever model Ed is using for WIR inference.

2. **Embedding space linearity.** Does the LLM embedding space support linear interpolation? If lerp(embedding_A, embedding_B, 0.5) decodes to a sensible blend of images A and B, the space is well-structured for WIR's soft refinement. If not, the refinement may produce out-of-distribution embeddings that decode to artifacts.

3. **Temporal consistency.** Sequential frames (tick N, tick N+1) should decode to visually consistent images. If small embedding changes produce large visual changes, the NPC's "perception" will flicker. May need a temporal consistency loss during training.

4. **Resolution vs cost tradeoff.** 336×336 is the standard ViT input. Higher resolution means more patches = more vision tokens = more compute. For NPC imagination, 336×336 is probably fine. For player-facing renders, may need 512+ with diffusion upscaling.
