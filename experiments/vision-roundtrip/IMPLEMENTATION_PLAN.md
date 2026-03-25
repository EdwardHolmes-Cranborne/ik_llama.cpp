# Vision Roundtrip — Implementation Plan (TDD)

## Architecture

```
vision_roundtrip/
├── core/
│   ├── __init__.py
│   ├── encoder.py          # Image → vision token embeddings via VLM's ViT
│   ├── projector.py        # mmproj forward + pseudoinverse
│   ├── inverter.py         # Feature inversion (gradient through ViT)
│   ├── llm_inverter.py     # LLM inversion (gradient through full model)
│   ├── describer.py        # Exhaustive image description via VLM
│   ├── editor.py           # Text description editing (find/replace, structured)
│   └── model_loader.py     # Load VLM components (ViT, mmproj, full model)
├── tests/
│   ├── __init__.py
│   ├── test_projector.py       # Pseudoinverse roundtrip tests
│   ├── test_encoder.py         # ViT encoding shape/dtype tests
│   ├── test_inverter.py        # Feature inversion convergence tests
│   ├── test_llm_inverter.py    # LLM inversion gradient flow tests
│   ├── test_describer.py       # Description generation tests
│   ├── test_editor.py          # Text edit parsing tests
│   ├── test_roundtrip.py       # End-to-end encode → decode tests
│   └── conftest.py             # Shared fixtures (dummy models, test images)
├── phase1_inversion.py     # CLI: encode → invert → image
├── phase3_wir_modify.py    # CLI: hidden state blending
├── phase3b_llm_inversion.py # CLI: full LLM inversion loop
├── requirements.txt
└── README.md
```

## Module Contracts

### 1. model_loader.py
```
load_vit(model_path) → (vit_encoder, image_processor, config)
load_mmproj(model_path) → (mmproj_weight, bias_or_none, in_dim, out_dim)
load_full_model(model_path) → (model, processor)
```
Tests: correct shapes, dtypes, device placement.

### 2. encoder.py
```
encode_image(image, vit, processor) → patch_embeddings (1, n_patches, vit_dim)
encode_image_to_llm(image, vit, mmproj, processor) → llm_embeddings (1, n_tokens, llm_dim)
get_vision_token_count(image_size, patch_size, merge_size) → int
```
Tests: output shapes match expected patch/token counts for known resolutions.
       dtype is float16. No NaN/Inf values.

### 3. projector.py
```
forward(patch_embeddings, mmproj_weight, bias) → llm_embeddings
pseudoinverse(llm_embeddings, mmproj_weight, bias) → recovered_patch_embeddings
roundtrip_error(patch_embeddings, mmproj_weight, bias) → mse_float
```
Tests: roundtrip error is near-zero for linear mmproj.
       Output shapes match input shapes transposed.
       Works with and without bias.

### 4. inverter.py
```
feature_inversion(
    target_embeddings, vit, processor,
    n_steps, lr, image_size,
    snapshot_steps=None, snapshot_dir=None,
    tv_weight_initial=0.01, tv_weight_final=0.0005,
) → (image_tensor, loss_history)
```
Tests: loss decreases monotonically (mostly).
       Output image is valid (0-1 range, correct shape).
       Snapshot files created at specified steps.
       Higher step count → lower final loss.

### 5. describer.py
```
DESCRIBE_PROMPT: str  # The fixed exhaustive description prompt
describe_image(model, processor, image) → (description_text, vision_mask, n_vision_tokens)
```
Tests: description is non-empty.
       vision_mask correctly identifies image token positions.
       n_vision_tokens > 0 for image input.

### 6. editor.py
```
parse_edit(edit_string) → (find_text, replace_text) or None
apply_edit(original_text, edit_string) → modified_text
apply_structured_edit(original_text, edits_dict) → modified_text
```
Tests: "change 'X' to 'Y'" correctly parsed.
       "replace 'X' with 'Y'" correctly parsed.
       Case-insensitive matching.
       Missing find_text returns original + appended instruction.
       Multiple edits applied in sequence.

### 7. llm_inverter.py
```
llm_inversion(
    model, processor, inputs, vision_mask,
    target_description,
    n_steps, lr,
    reg_weight=0.1, tv_weight=0.01,
) → (modified_embeddings, loss_history, per_token_deltas)
```
Tests: loss decreases over steps.
       Modified embeddings differ from original.
       Per-token deltas are non-zero.
       Regularisation prevents excessive drift.
       Output shapes match input vision embedding shapes.

## Test Strategy

### Unit Tests (no GPU, no model required)
- editor.py: pure text manipulation, no dependencies
- projector.py: can test with random tensors on CPU
- encoder.py shape calculations: pure maths

### Integration Tests (need model on disk)
- encoder.py: actual ViT encoding
- inverter.py: actual gradient descent
- describer.py: actual model generation
- llm_inverter.py: actual backward pass

### Fixtures (conftest.py)
- `dummy_mmproj`: random linear weight matrix for projector tests
- `dummy_embeddings`: random tensors with correct shapes
- `test_image`: a simple synthetic test image (colour gradient or checkerboard)
- `model_path`: path to Qwen3.5-9B (skip if not available)

## Build Order (TDD)

1. **editor.py + test_editor.py** — pure logic, no deps, instant tests
2. **projector.py + test_projector.py** — CPU tensors only, fast tests
3. **conftest.py** — shared fixtures
4. **encoder.py + test_encoder.py** — needs model for integration, mock shapes for unit
5. **model_loader.py** — thin wrappers, test with actual model
6. **inverter.py + test_inverter.py** — needs ViT, test convergence properties
7. **describer.py + test_describer.py** — needs full model
8. **llm_inverter.py + test_llm_inverter.py** — needs full model, most complex
9. **Integration: test_roundtrip.py** — end-to-end
10. **Update CLI scripts** to use core modules
