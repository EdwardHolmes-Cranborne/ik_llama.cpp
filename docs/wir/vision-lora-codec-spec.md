# Vision LoRA Codec

## Teaching an LLM to Read and Write Its Own Image Tokens

**Status:** Specification v0.1 — DRAFT
**Author:** Ed Holmes (Cranborne Audio) / Ren
**Date:** March 2026
**Depends on:** WIR Tiered Model Architecture, Vision Roundtrip Experiments

---

## 1. The Problem

Vision-language models process images as continuous embeddings injected at layer 0. These embeddings bypass the text vocabulary entirely — the model cannot output vision tokens through its LM head. This means:

- WIR cannot refine vision tokens via standard token prediction (n-least-conf operates on vocab logits)
- The model cannot "write" a modified image — it can only write text about the image
- Image modification requires gradient descent through the full model (LLM inversion, section 9.9) — expensive, ~10-25 seconds per edit

## 2. The Solution: Vision Tokens as Vocabulary Tokens

Train a LoRA adapter that teaches the model to represent image patch embeddings as regular vocabulary tokens. The model's existing embedding table (248,320 tokens × 4096 dims) is already a dense codebook of 4096-dimensional vectors. Each vision patch embedding can be mapped to its nearest neighbour in this codebook.

Once the LoRA learns this mapping, vision patches become standard tokens that the model can:
- **Read** — understand what visual content each token represents
- **Write** — output modified tokens through the normal LM head
- **WIR-refine** — standard n-least-conf token replacement at prefill speed

No gradient descent. No backprop through the LLM. Image editing at the speed of text generation.

---

## 3. Architecture

### 3.1 The Embedding Table as Codebook

The model's text embedding table maps 248,320 token IDs to 4096-dim vectors:

```
embed_table: token_id → embedding_vector (4096-dim)

Token 0:      [0.0234, -0.0891, 0.1247, ..., 0.0567]  (4096 values)
Token 1:      [0.0156, -0.1023, 0.0893, ..., -0.0234]
...
Token 248319: [-0.0445, 0.0678, -0.0912, ..., 0.0389]
```

These vectors span the model's internal representation space. A vision patch embedding (also 4096-dim, after projection) can be matched to its nearest token embedding:

```
vision_patch_embedding = [0.0230, -0.0895, 0.1250, ..., 0.0560]

Nearest token: Token 0 (cosine similarity: 0.9987)
→ Encode this patch as Token 0
```

The quantisation error is the distance between the vision embedding and the nearest token embedding. With 248,320 codebook entries in 4096 dimensions, the expected quantisation error is small — there are enough tokens to represent the vision space with reasonable fidelity.

### 3.2 Multi-Token Patches for Higher Fidelity

A single token per patch uses one of 248,320 vectors — good but potentially lossy for fine detail. For higher fidelity, represent each patch with 2-3 tokens:

| Tokens per patch | Codebook size | Precision | Tokens for 100 patches |
|---|---|---|---|
| 1 | 248,320 | Coarse — major features, colour regions | 100 |
| 2 | 248,320² ≈ 61.6 billion | High — most detail preserved | 200 |
| 3 | 248,320³ ≈ absurd | Overkill | 300 |

With 2 tokens per patch: first token encodes the "coarse" embedding (nearest neighbour), second token encodes the residual (difference between actual embedding and first token's vector, also matched to nearest neighbour in the embedding table). This is residual vector quantisation — the same technique used in neural audio codecs (Encodec, DAC).

For initial experiments, start with 1 token per patch. Scale to 2 if quantisation error is too high.

### 3.3 The Quantisation Pipeline

```
ENCODE (image → tokens):
  Image
    → ViT encoder (frozen)
    → patch embeddings [n_patches × 1152]
    → mmproj projection (frozen) [n_patches × 4096]
    → for each patch:
        find nearest token in embed_table (cosine similarity)
        → token_id
    → sequence of token IDs [n_patches]

DECODE (tokens → image):
  Token IDs [n_patches]
    → embed_table lookup → embeddings [n_patches × 4096]
    → pseudoinverse mmproj → ViT patch space [n_patches × 1152]
    → ViT feature inversion or trained decoder → image

WIR MODIFY (tokens → modified tokens):
  Current image tokens [n_patches]
    + text context ("the scene feels threatening")
    → model forward pass (standard, reads tokens from vocab)
    → logits at image token positions → predict replacement tokens
    → n-least-conf: replace least confident image tokens
    → modified token IDs [n_patches]
    → decode to image
```

### 3.4 Sequence Format

The image tokens are placed in the prompt using special delimiters:

```
[system prompt text tokens]
<|vision_start|>
[image token 1] [image token 2] ... [image token 100]
<|vision_end|>
[instruction text tokens: "describe this scene" or "imagine this scene as threatening"]
```

The model sees regular token IDs at the vision positions. The LoRA teaches it that tokens between `<|vision_start|>` and `<|vision_end|>` represent visual content, not text.

---

## 4. LoRA Training

### 4.1 What the LoRA Learns

The base model already knows what every token embedding vector looks like in its representation space. The LoRA teaches it two things:

1. **Recognition:** When tokens between vision delimiters are read, interpret them as visual content — "Token 4723 at vision position means a dark green patch with slight blue tint in the upper-left region"

2. **Generation:** When predicting tokens at vision positions, output token IDs that represent the desired visual content — "To make this region brighter, predict Token 8891 instead of Token 4723"

### 4.2 Training Data Generation

Training data is self-generated from any image dataset:

```
For each training image:

  1. ENCODE the image:
     image → ViT → mmproj → patch embeddings
     → quantise each patch to nearest vocab token
     → image_tokens: [token_id_1, token_id_2, ..., token_id_N]

  2. Build RECOGNITION training examples:
     Input:  <|vision_start|> [image_tokens] <|vision_end|>
             Describe what you see in this image.
     Target: [detailed description of the image]

     The LoRA learns: "these token IDs at vision positions
     represent THIS visual content"

  3. Build GENERATION training examples:
     Input:  <|vision_start|> [image_tokens_v1] <|vision_end|>
             Modify this image: [edit instruction]
     Target: <|vision_start|> [image_tokens_v2] <|vision_end|>

     Where v2 is the quantised encoding of the edited image.
     Edit pairs generated by:
       a) Augmentation: colour shift, crop, rotate → re-encode
       b) Text-guided: use a diffusion model to generate edited version
          → encode both original and edited → training pair
       c) Synthetic: programmatic edits (brightness, contrast, blur)
          → encode before and after

  4. Build RECONSTRUCTION training examples:
     Input:  <|vision_start|> [image_tokens] <|vision_end|>
             Reproduce these exact vision tokens.
     Target: <|vision_start|> [image_tokens] <|vision_end|>

     Identity mapping — teaches the LoRA to copy vision tokens
     through the model without corruption. This is the baseline
     that modification training builds on.
```

### 4.3 Training Configuration

```yaml
lora:
  rank: 16                # rank 16 for vision capability
  alpha: 32               # scaling factor
  target_modules:         # which layers get LoRA adapters
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  dropout: 0.05

training:
  dataset_size: 50000     # images (generates ~150K training examples)
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 2e-4
  warmup_steps: 100
  epochs: 3
  max_seq_length: 2048    # enough for ~100 vision tokens + text
  bf16: true

quantisation_codebook:
  source: model_embed_table    # use the model's own embedding table
  n_entries: 248320            # full vocabulary
  dim: 4096                    # embedding dimension
  distance_metric: cosine      # cosine similarity for nearest neighbour
  multi_token: 1               # tokens per patch (start with 1)
```

### 4.4 Training Cost Estimate

| Component | Cost |
|---|---|
| Dataset generation (50K images → ViT encode → quantise) | ~30 mins on RTX 5090 |
| LoRA training (rank 16, 3 epochs, 150K examples) | ~4-8 hours on RTX 5090 |
| Total trainable params | ~20-40M (LoRA only) |
| Base model | Frozen (9B) |

---

## 5. Quantisation Quality Analysis

### 5.1 Expected Codebook Coverage

The embedding table has 248,320 vectors in 4096-dim space. Vision patch embeddings occupy a subspace of this space (they come from the ViT projection, not from text token lookups). The question is: how well does the text codebook cover the vision subspace?

**Factors in our favour:**
- 248,320 is a LOT of codebook entries. In 4096 dimensions, this provides dense coverage.
- The mmproj projects vision embeddings INTO the LLM embedding space — the same space the token embeddings live in. They're designed to be compatible.
- Vision-language models are trained with vision and text tokens interleaved. The embedding space has been shaped to accommodate both modalities.

**Factors against us:**
- Text token embeddings cluster around text-relevant regions of the space. Vision embeddings might occupy different regions with sparser coverage.
- The embedding table was trained for text reconstruction, not vision — some visual distinctions might map to the same token.

**Empirical test (Phase 0):** Encode 1000 images, quantise all patches to nearest tokens, measure:
- Mean cosine similarity between original and quantised embeddings
- Reconstruction quality (quantised → pseudoinverse → inversion → compare to original)
- Distribution of token usage (are we using many tokens or clustering on a few?)

### 5.2 Residual Quantisation (If Needed)

If single-token quantisation is too lossy, add a second token per patch encoding the residual:

```
patch_embedding = [actual 4096-dim vector]
token_1 = nearest_neighbour(patch_embedding, embed_table)
residual = patch_embedding - embed_table[token_1]
token_2 = nearest_neighbour(residual, embed_table)

Reconstruction: embed_table[token_1] + embed_table[token_2] ≈ patch_embedding
```

This halves the quantisation error. A third token would reduce it further but is likely overkill.

---

## 6. WIR Integration

### 6.1 Vision Tokens as Zone 3 Variables

With the LoRA codec, each vision patch token is a standard Zone 3 mutable variable:

```
Micro-prompt for vision patch 47:

Zone 1 (instruction, cached):
  "You are viewing a scene. The token at this position represents
   visual content in patch row 6, column 7 (of a 10×10 grid).
   Given the context, predict the correct vision token for this patch."

Zone 2 (context, cached per-turn):
  "The scene is a market at dusk. The viewer feels threatened.
   Nearby patches: [token_46=上, token_48=明, token_37=画]"

Zone 3 (answer, WIR-refined):
  碑    ← current vision token (one from the model's vocabulary)
```

WIR n-least-conf evaluates: "is 碑 the right token for this patch given the threatening context?" If the model assigns low probability to 碑 and high probability to 暗 (a darker-meaning token), it replaces it. Standard WIR. Standard prefill speed.

### 6.2 Batch Processing

100 vision patch tokens × ~80 tokens per micro-prompt = ~8000 tokens per WIR pass. At 20K tok/s batched prefill, that's ~400ms per pass. 3 passes to converge = ~1.2 seconds to re-imagine a scene.

Compare to LLM inversion: 50 backward passes × 200-500ms each = 10-25 seconds.

**10-20× faster for the same operation.**

### 6.3 Compatibility with the Cascade

The vision codec tokens slot directly into the NPC cognitive cascade (section 7):

```
Layer 2 (Perception):
  Input: quantised vision tokens from ViT encoding of game scene
  WIR pass: model refines tokens based on NPC's attention/emotion
  Output: subjectively modified vision token sequence

Layer 6 (Prediction):
  Input: current vision tokens
  WIR pass: model predicts what the scene will look like next tick
  Output: predicted future vision tokens

Memory:
  Store: vision token sequence (compact — just token IDs, ~100 integers)
  Recall: load token IDs, inject as Zone 3 for re-imagination
  Re-imagine: WIR modifies recalled tokens with new emotional context
```

Memory storage becomes trivial — a 100-patch image is 100 integers (or 200 for 2-token-per-patch). A few hundred bytes per memory snapshot instead of megabytes of float embeddings.

---

## 7. Decode Pipeline

### 7.1 Tokens Back to Image

```
Modified vision token IDs [n_patches]
  → embed_table lookup → quantised embeddings [n_patches × 4096]
  → pseudoinverse mmproj → ViT patch space [n_patches × 1152]
  → feature inversion (15-50 steps, fast — target is known) → rough image
  → diffusion img2img → photorealistic output
```

Or, for the fast path:
```
Modified token IDs → embed_table lookup → quantised embeddings
  → trained decoder (80M params, 10-50ms) → rough image
  → diffusion → final image
```

### 7.2 Direct Token-to-Image Decoder (Future)

With enough training data, the LoRA could learn to decode vision tokens directly to pixel descriptions — effectively becoming a small image generation model. The model outputs tokens at vision positions, and a lightweight decoder head maps them to pixel patches. This would eliminate the ViT inversion step entirely.

---

## 8. Phased Implementation

### Phase 0: Quantisation Quality (1-2 days)
- Encode 1000 images through ViT + mmproj
- Quantise each patch embedding to nearest token in embed_table
- Measure: cosine similarity, reconstruction quality, token distribution
- **Exit criteria:** Mean cosine similarity > 0.8 between original and quantised embeddings

### Phase 1: LoRA Training — Recognition (1 week)
- Generate recognition training data (image → tokens → description pairs)
- Train LoRA (rank 16) on recognition task only
- Validate: model correctly describes images from their quantised token representation
- **Exit criteria:** Descriptions from quantised tokens are comparable quality to descriptions from real vision embeddings

### Phase 2: LoRA Training — Generation (1-2 weeks)
- Generate edit training data (original tokens → edit instruction → modified tokens)
- Train LoRA on generation task (predicting modified vision tokens)
- Validate: model outputs sensible vision tokens when asked to modify a scene
- **Exit criteria:** Modified tokens decode to visually different images that reflect the requested edit

### Phase 3: WIR Integration (1 week)
- Wire quantised vision tokens into the wirstate as Zone 3 variables
- Run WIR n-least-conf over vision token positions
- Validate: WIR converges, modified tokens decode to coherent images
- **Exit criteria:** WIR-based scene modification at <2 seconds, visually comparable to LLM inversion results

### Phase 4: Cascade Integration (1 week)
- Vision tokens in the NPC perception and prediction layers
- Memory storage as token ID sequences
- Re-imagination via WIR with emotional context
- **Exit criteria:** NPCs can perceive, remember, and re-imagine scenes using the token codec

---

## 9. Comparison: Three Approaches to Image Modification

| Approach | Speed | Quality | Training needed | Complexity |
|---|---|---|---|---|
| **Hidden state blending** (section 9.7.2) | Instant (~100ms) | Mood/colour only | None | Low |
| **LLM inversion** (section 9.9) | 10-25s | Geometric + mood | None | High (backprop) |
| **LoRA codec + WIR** (this spec) | ~1.2s | Geometric + mood | LoRA training (~8hrs) | Medium |

The LoRA codec is the sweet spot: 10× faster than LLM inversion, handles geometric edits (not just mood), and uses standard WIR infrastructure. The upfront cost is training the LoRA, but once trained it's a permanent capability.

All three approaches compose. For a specific edit:
1. LoRA codec WIR for the main edit (fast, geometric)
2. Hidden state blending for emotional colouring (instant, mood)
3. LLM inversion as fallback for edits the codec can't handle (slow, any edit)

---

## 10. Open Questions

1. **Codebook coverage:** Does the text embedding table adequately cover the vision embedding subspace? Phase 0 answers this empirically.

2. **Semantic leakage:** When a vision token uses Token 4723 (which in text means "the"), does the model's text understanding of "the" interfere with its visual interpretation? The LoRA should learn to suppress text semantics at vision positions, but this needs validation.

3. **Quantisation artifacts:** Do nearest-neighbour boundaries in the codebook create visible artifacts in decoded images? Nearby patches that should look similar might quantise to very different tokens if they fall on opposite sides of a Voronoi boundary.

4. **Multi-token ordering:** For 2-token-per-patch encoding, does the order matter? (Token pair [A, B] vs [B, A]). Residual quantisation has a natural order (coarse then residual), but other orderings might work better.

5. **LoRA rank:** Is rank 16 enough for vision capability, or does it need rank 32-64? Higher rank = more parameters = longer training but potentially better quality.

6. **Vocab subset:** Would it be better to designate a specific subset of the vocabulary for vision tokens (e.g., reserve tokens 200000-248320 for vision) rather than using the full vocab? This would prevent semantic leakage at the cost of a smaller codebook.
