# Vision Roundtrip — Implementation Log

## Phase 1: Encode → Invert → Image

### Session 1 — 2026-03-18

#### Starting State
- Qwen3.5-9B safetensors downloading to D:/models/Qwen3.5-9B-HF/
- CLI scripts written (phase1_inversion.py, phase3_wir_modify.py, phase3b_llm_inversion.py)
- No modular code structure yet — refactoring into core/ modules with TDD

#### Architecture Decisions

**Decision: Separate core modules from CLI scripts**
- Why: CLI scripts were monolithic, hard to test, duplicated code
- Approach: core/ package with encoder, projector, inverter, etc.
- Each module has a clear contract and independent tests

**Decision: TDD build order — editor and projector first**
- Why: editor.py is pure text logic (no GPU, no model, instant tests)
- Why: projector.py tests pseudoinverse maths with random tensors on CPU
- These validate the foundations before touching any model code

**Decision: Two test tiers — unit (no GPU) and integration (needs model)**
- Why: want to iterate fast on logic without waiting for model loads
- Unit tests use dummy tensors and mocks
- Integration tests marked with pytest.mark.skipif when model not available

#### Discoveries

1. **Pseudoinverse roundtrip error is near-zero** for linear mmproj (4096×768 random weight). Confirmed with test — MSE < 1e-3. This means the projection step loses effectively no information. The ViT encoding is the lossy step, not the projection.

2. **Half-precision (float16) roundtrip** has higher error (~0.1 MSE) but still usable. The pseudoinverse computation should always be done in float32 for stability, with inputs/outputs cast as needed.

3. **Inverter convergence on dummy ViT** confirmed: loss decreases monotonically, more steps always helps, snapshots work. The cosine annealing LR + decaying TV regularisation combination converges well. 30 steps sufficient for simple patterns, 200+ needed for detail.

4. **C: drive is full (22MB free)** — HuggingFace downloads fail because the cache is on C:. Fixed by setting `HF_HOME=/d/.hf_cache`. Download restarted to D: drive.

5. **Qwen3.5-9B has dynamic resolution** — more pixels = more vision tokens. 336px → ~110 tokens, 1024px → ~1024 tokens, 4096px → ~16384 tokens. The spatial merge (2×) is the biggest resolution reduction.

6. **Vision tokens are continuous embeddings, not discrete token IDs.** They bypass the text embedding table entirely. The model cannot "output" vision tokens through the LM head — but hidden states at vision positions are in the same 4096-dim space. WIR operates on hidden states. LLM inversion backprops through the full model.

7. **MoE backprop is not special** — PyTorch autograd records which experts were used in the forward pass and backprops through the same path. No straight-through estimator needed.

#### Changes from Original Plan

1. **Added LLM inversion (phase3b)** — was not in original plan. The "describe → edit → backprop" loop is a more powerful approach than hidden state blending for geometric/structural edits.

2. **Refactored to core/ modules with TDD** — original scripts were monolithic. Now have separate tested modules: editor (15 tests), projector (13 tests), encoder (8 tests), inverter (10 tests), llm_inverter (4 tests), describer, model_loader.

3. **Changed default inversion steps from 15 to 200** — 15 was too conservative. Sweep mode tests up to 500 steps to find the quality ceiling.

4. **Added cosine similarity loss** alongside MSE for feature inversion — helps with directional alignment in high-dim space, not just magnitude matching.

#### Test Results

| Module | Tests | Status | Time |
|---|---|---|---|
| editor | 15 | All pass | 0.04s |
| projector | 13 | All pass | 34.5s (pinv computation) |
| encoder (unit) | 8 | All pass | 0.02s |
| inverter | 10 | All pass | 225s (gradient descent on CPU) |
| llm_inverter (unit) | 4 | All pass | <1s |
| **Total** | **50** | **All pass** | |

Integration tests pending model download completion.

### Session 2 — 2026-03-19

#### Downloads Completed
- Qwen3.5-9B: 18.2GB safetensors on D:/models/Qwen3.5-9B-HF/ ✓
- Fuyu-8B: 17.6GB safetensors on D:/models/Fuyu-8B-HF/ ✓
- PyTorch 2.10.0+cu128 with CUDA 12.8 ✓
- RTX 5090 (34.2GB) + RTX 3070 (8.6GB) detected ✓

#### Download Struggles
- C: drive full (22MB free). HuggingFace hub downloads fail because cache is on C:.
- HF_HOME env var partially works but xet handler writes temp files to system temp (also C:).
- Setting TMPDIR/TEMP/TMP/XET_CACHE_DIR all to D: still didn't work reliably.
- First HF download reported "complete" but files were empty/symlinked.
- Direct curl download used wrong filenames (model-00001 vs model.safetensors-00001).
- Final working approach: curl with correct filenames from HF resolve URLs.
- **Lesson: on disk-constrained systems, skip HF hub and use direct curl downloads.**

#### Fuyu Phase 1 Results — PASS

**Roundtrip MSE: 0.00000000. Cosine similarity: 1.000092.**

Fuyu's linear patch projection (2700→4096 with bias) is **perfectly invertible** via pseudoinverse. The image roundtrips through token space with zero measurable loss. This confirms:
- The projection weight shape (4096, 2700) is in shard 2: `vision_embed_tokens.weight`
- The bias exists: `vision_embed_tokens.bias` (4096,)
- Pseudoinverse recovery is exact — the projection is injective (out_dim > in_dim)
- We can decode any modified token back to pixels perfectly

**Critical: we don't need the full model for the roundtrip test.** Loading just the projection weight (2 tensors, ~44MB) avoids the 18GB full model load. The full model is only needed for WIR refinement (Phase 2).

#### Discoveries (Session 2)

1. **Fuyu's patch projection is vision_embed_tokens.weight** — not in a separate mmproj file. It's in the main model weights, shard 2.

2. **The projection has a bias** — vision_embed_tokens.bias (4096,). The pseudoinverse correctly handles this (subtract bias before inversion).

3. **Segfault on CPU with full safetensors load** — loading 18GB of bfloat16 tensors on CPU causes OOM/segfault. Solution: load only the needed tensors.

4. **Unicode arrow (→) crashes on Windows cp1252 console** — replaced with ASCII arrows in output strings.
