"""Tests for feature inversion — unit tests with dummy targets on CPU."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from core.inverter import feature_inversion, tensor_to_pil, InversionResult


class DummyViT(torch.nn.Module):
    """A trivial 'ViT' that just averages patches — for testing the
    optimisation loop without loading a real model."""

    def __init__(self, image_size=64, patch_size=16, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        # Simple linear projection from patch pixels to embedding
        self.proj = torch.nn.Linear(patch_size * patch_size * 3, embed_dim, bias=False)
        # Freeze — we're testing inversion, not training the ViT
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: (batch, 3, H, W)
        b, c, h, w = x.shape
        ph = pw = self.patch_size
        # Reshape to patches
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (b, 3, n_h, n_w, ph, pw)
        x = x.contiguous().view(b, c, -1, ph, pw)  # (b, 3, n_patches, ph, pw)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (b, n_patches, 3, ph, pw)
        x = x.view(b, x.shape[1], -1)  # (b, n_patches, 3*ph*pw)
        return self.proj(x)  # (b, n_patches, embed_dim)


@pytest.fixture
def dummy_vit():
    torch.manual_seed(42)
    return DummyViT(image_size=64, patch_size=16, embed_dim=128)


@pytest.fixture
def target_from_gradient_image(dummy_vit):
    """Create target embeddings by encoding a gradient image through the dummy ViT."""
    img = torch.zeros(1, 3, 64, 64)
    # Red gradient left to right
    img[0, 0, :, :] = torch.linspace(0, 1, 64).unsqueeze(0)
    # Green gradient top to bottom
    img[0, 1, :, :] = torch.linspace(0, 1, 64).unsqueeze(1)
    # Blue constant
    img[0, 2, :, :] = 0.5

    with torch.no_grad():
        target = dummy_vit(img)
    return target


class TestFeatureInversion:
    def test_returns_inversion_result(self, dummy_vit, target_from_gradient_image):
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=5, lr=0.1, image_size=64, device="cpu"
        )
        assert isinstance(result, InversionResult)
        assert result.image is not None
        assert result.loss_history is not None
        assert result.best_loss is not None

    def test_output_image_shape(self, dummy_vit, target_from_gradient_image):
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=5, lr=0.1, image_size=64, device="cpu"
        )
        assert result.image.shape == (1, 3, 64, 64)

    def test_output_in_valid_range(self, dummy_vit, target_from_gradient_image):
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=10, lr=0.1, image_size=64, device="cpu"
        )
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_loss_decreases(self, dummy_vit, target_from_gradient_image):
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=30, lr=0.1, image_size=64, device="cpu"
        )
        # First loss should be higher than last loss
        assert result.loss_history[-1] < result.loss_history[0]

    def test_more_steps_lower_loss(self, dummy_vit, target_from_gradient_image):
        result_10 = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=10, lr=0.1, image_size=64, device="cpu"
        )
        result_50 = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=50, lr=0.1, image_size=64, device="cpu"
        )
        assert result_50.best_loss < result_10.best_loss

    def test_no_nans(self, dummy_vit, target_from_gradient_image):
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=20, lr=0.1, image_size=64, device="cpu"
        )
        assert not torch.isnan(result.image).any()
        assert all(not np.isnan(l) for l in result.loss_history)

    def test_snapshots_created(self, dummy_vit, target_from_gradient_image, tmp_path):
        snapshot_steps = [5, 10]
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=10, lr=0.1, image_size=64, device="cpu",
            snapshot_steps=snapshot_steps, snapshot_dir=str(tmp_path),
        )
        for step in snapshot_steps:
            assert (tmp_path / f"step_{step:04d}.png").exists()

    def test_best_image_tracked(self, dummy_vit, target_from_gradient_image):
        result = feature_inversion(
            target_from_gradient_image, dummy_vit,
            n_steps=20, lr=0.1, image_size=64, device="cpu"
        )
        # best_loss should be the minimum of the history
        assert result.best_loss <= min(result.loss_history) + 1e-6


class TestTensorToPil:
    def test_valid_output(self):
        tensor = torch.rand(1, 3, 64, 64)
        img = tensor_to_pil(tensor)
        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)
        assert img.mode == "RGB"

    def test_clamping(self):
        tensor = torch.tensor([[[[2.0, -1.0], [0.5, 0.5]]]]).expand(1, 3, 2, 2)
        img = tensor_to_pil(tensor)
        pixels = np.array(img)
        assert pixels.max() <= 255
        assert pixels.min() >= 0
