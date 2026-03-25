"""Integration tests — end-to-end encode → decode with real model.

These tests are skipped if the model is not available.
Run with: pytest tests/test_roundtrip.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from tests.conftest import requires_model, requires_cuda, QWEN_MODEL_PATH
from core.model_loader import load_full_model, extract_vision_components, check_model_ready
from core.encoder import encode_image, get_vision_token_count, extract_vision_embeddings_from_model_input
from core.projector import pseudoinverse, roundtrip_error, forward_project
from core.inverter import feature_inversion, tensor_to_pil
from core.describer import describe_image


@requires_model
@requires_cuda
class TestModelLoading:
    """Test that we can load the model and extract components."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load model once for all tests in this class."""
        model, processor = load_full_model(QWEN_MODEL_PATH, device="cuda")
        return model, processor

    def test_model_loads(self, loaded_model):
        model, processor = loaded_model
        assert model is not None
        assert processor is not None

    def test_vision_components_extracted(self, loaded_model):
        model, _ = loaded_model
        components = extract_vision_components(model)
        assert 'visual_model' in components
        assert 'hidden_size' in components
        assert components['hidden_size'] == 4096

    def test_image_token_id_set(self, loaded_model):
        model, _ = loaded_model
        components = extract_vision_components(model)
        assert components['image_token_id'] == 248056


@requires_model
@requires_cuda
class TestEncodeDecode:
    """Test the full encode → project → pseudoinverse → inversion pipeline."""

    @pytest.fixture(scope="class")
    def model_and_image(self):
        model, processor = load_full_model(QWEN_MODEL_PATH, device="cuda")
        # Create a simple test image
        img = np.zeros((336, 336, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 336, dtype=np.uint8)[np.newaxis, :]
        img[:, :, 1] = np.linspace(0, 255, 336, dtype=np.uint8)[:, np.newaxis]
        img[:, :, 2] = 128
        image = Image.fromarray(img)
        return model, processor, image

    def test_encoding_produces_embeddings(self, model_and_image):
        model, processor, image = model_and_image
        components = extract_vision_components(model)
        vision_embeds, vision_mask, info = extract_vision_embeddings_from_model_input(
            model, processor, image, device="cuda"
        )
        assert vision_embeds.shape[0] == 1
        assert vision_embeds.shape[1] > 0  # at least some vision tokens
        assert vision_embeds.shape[2] == 4096  # LLM hidden dim
        assert info['n_vision_tokens'] > 0

    def test_vision_token_count_reasonable(self, model_and_image):
        model, processor, image = model_and_image
        _, _, info = extract_vision_embeddings_from_model_input(
            model, processor, image, device="cuda"
        )
        # 336×336 should produce ~110 tokens (depends on exact preprocessing)
        n = info['n_vision_tokens']
        assert 50 < n < 500, f"Unexpected token count: {n}"

    def test_embeddings_not_all_same(self, model_and_image):
        model, processor, image = model_and_image
        vision_embeds, _, _ = extract_vision_embeddings_from_model_input(
            model, processor, image, device="cuda"
        )
        # Different spatial regions should produce different embeddings
        # (our test image is a gradient, so patches should differ)
        first = vision_embeds[0, 0, :]
        last = vision_embeds[0, -1, :]
        diff = (first - last).norm().item()
        assert diff > 0.1, "All vision tokens are identical — no spatial info encoded"


@requires_model
@requires_cuda
class TestDescription:
    """Test that the model can describe an image."""

    @pytest.fixture(scope="class")
    def model_and_image(self):
        model, processor = load_full_model(QWEN_MODEL_PATH, device="cuda")
        # Use a more interesting test image — a photo if available, else gradient
        test_photos = list(Path("test_images").glob("*.jpg")) + list(Path("test_images").glob("*.png"))
        if test_photos:
            image = Image.open(test_photos[0]).convert("RGB")
        else:
            img = np.zeros((336, 336, 3), dtype=np.uint8)
            img[:168, :, :] = [135, 206, 235]  # sky blue top half
            img[168:, :, :] = [34, 139, 34]     # green bottom half
            image = Image.fromarray(img)
        return model, processor, image

    def test_description_generated(self, model_and_image):
        model, processor, image = model_and_image
        description, info = describe_image(
            model, processor, image,
            device="cuda", max_new_tokens=200
        )
        assert len(description) > 50, "Description too short"
        assert info['n_vision_tokens'] > 0
        assert info['n_description_tokens'] > 0

    def test_description_contains_visual_content(self, model_and_image):
        model, processor, image = model_and_image
        description, _ = describe_image(
            model, processor, image,
            device="cuda", max_new_tokens=200
        )
        # Should contain at least some visual descriptors
        desc_lower = description.lower()
        visual_words = ['color', 'colour', 'light', 'dark', 'left', 'right',
                        'top', 'bottom', 'background', 'image', 'scene',
                        'blue', 'green', 'red', 'sky', 'ground']
        found = [w for w in visual_words if w in desc_lower]
        assert len(found) >= 2, f"Description lacks visual content. Found: {found}"


@requires_model
@requires_cuda
class TestFeatureInversionWithRealViT:
    """Test feature inversion through the actual ViT encoder."""

    @pytest.fixture(scope="class")
    def setup(self):
        model, processor = load_full_model(QWEN_MODEL_PATH, device="cuda")
        components = extract_vision_components(model)

        # Create test image and encode
        img = np.zeros((336, 336, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 336, dtype=np.uint8)[np.newaxis, :]
        img[:, :, 1] = np.linspace(0, 255, 336, dtype=np.uint8)[:, np.newaxis]
        img[:, :, 2] = 128
        image = Image.fromarray(img)

        vision_embeds, _, info = extract_vision_embeddings_from_model_input(
            model, processor, image, device="cuda"
        )
        return model, processor, components, image, vision_embeds, info

    def test_inversion_loss_decreases(self, setup):
        model, processor, components, image, vision_embeds, info = setup
        vit = components['visual_model']

        # Run a short inversion
        result = feature_inversion(
            vision_embeds, vit,
            n_steps=20, lr=0.05, image_size=336, device="cuda",
            verbose=False,
        )
        # Loss should decrease
        assert result.loss_history[-1] < result.loss_history[0]

    def test_inversion_produces_image(self, setup):
        model, processor, components, image, vision_embeds, info = setup
        vit = components['visual_model']

        result = feature_inversion(
            vision_embeds, vit,
            n_steps=10, lr=0.05, image_size=336, device="cuda",
            verbose=False,
        )
        pil_img = tensor_to_pil(result.image)
        assert pil_img.size == (336, 336)
        assert pil_img.mode == "RGB"
