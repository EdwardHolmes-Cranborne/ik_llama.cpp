"""Tests for mmproj pseudoinverse — CPU tensors only, no model needed."""

import pytest
import torch
from core.projector import forward_project, pseudoinverse, roundtrip_error


@pytest.fixture
def linear_mmproj():
    """A simple linear projection matrix (no bias). 768 → 4096."""
    torch.manual_seed(42)
    weight = torch.randn(4096, 768)  # (out_dim, in_dim) — nn.Linear convention
    return weight, None


@pytest.fixture
def linear_mmproj_with_bias():
    """Linear projection with bias."""
    torch.manual_seed(42)
    weight = torch.randn(4096, 768)
    bias = torch.randn(4096)
    return weight, bias


@pytest.fixture
def patch_embeddings():
    """Fake patch embeddings: batch=1, n_patches=100, dim=768."""
    torch.manual_seed(123)
    return torch.randn(1, 100, 768)


class TestForwardProject:
    def test_output_shape(self, patch_embeddings, linear_mmproj):
        weight, bias = linear_mmproj
        result = forward_project(patch_embeddings, weight, bias)
        assert result.shape == (1, 100, 4096)

    def test_output_shape_with_bias(self, patch_embeddings, linear_mmproj_with_bias):
        weight, bias = linear_mmproj_with_bias
        result = forward_project(patch_embeddings, weight, bias)
        assert result.shape == (1, 100, 4096)

    def test_no_nans(self, patch_embeddings, linear_mmproj):
        weight, bias = linear_mmproj
        result = forward_project(patch_embeddings, weight, bias)
        assert not torch.isnan(result).any()

    def test_no_infs(self, patch_embeddings, linear_mmproj):
        weight, bias = linear_mmproj
        result = forward_project(patch_embeddings, weight, bias)
        assert not torch.isinf(result).any()


class TestPseudoinverse:
    def test_output_shape(self, patch_embeddings, linear_mmproj):
        weight, bias = linear_mmproj
        projected = forward_project(patch_embeddings, weight, bias)
        recovered = pseudoinverse(projected, weight, bias)
        assert recovered.shape == (1, 100, 768)

    def test_roundtrip_near_zero_error(self, patch_embeddings, linear_mmproj):
        """For a linear projection with out_dim > in_dim, pseudoinverse should
        recover the original almost exactly (the projection is injective)."""
        weight, bias = linear_mmproj
        error = roundtrip_error(patch_embeddings, weight, bias)
        assert error < 1e-3, f"Roundtrip error too high: {error}"

    def test_roundtrip_with_bias(self, patch_embeddings, linear_mmproj_with_bias):
        weight, bias = linear_mmproj_with_bias
        error = roundtrip_error(patch_embeddings, weight, bias)
        assert error < 1e-3, f"Roundtrip error with bias too high: {error}"

    def test_no_nans(self, patch_embeddings, linear_mmproj):
        weight, bias = linear_mmproj
        projected = forward_project(patch_embeddings, weight, bias)
        recovered = pseudoinverse(projected, weight, bias)
        assert not torch.isnan(recovered).any()

    def test_different_batch_sizes(self, linear_mmproj):
        weight, bias = linear_mmproj
        for batch_size in [1, 4, 16]:
            patches = torch.randn(batch_size, 50, 768)
            projected = forward_project(patches, weight, bias)
            recovered = pseudoinverse(projected, weight, bias)
            assert recovered.shape == patches.shape

    def test_single_patch(self, linear_mmproj):
        """Edge case: single patch."""
        weight, bias = linear_mmproj
        patches = torch.randn(1, 1, 768)
        error = roundtrip_error(patches, weight, bias)
        assert error < 1e-3


class TestRoundtripError:
    def test_returns_scalar(self, patch_embeddings, linear_mmproj):
        weight, bias = linear_mmproj
        error = roundtrip_error(patch_embeddings, weight, bias)
        assert isinstance(error, float)

    def test_zero_input(self, linear_mmproj):
        """Zero embeddings should roundtrip to zero (no bias) or near-zero."""
        weight, bias = linear_mmproj
        patches = torch.zeros(1, 10, 768)
        error = roundtrip_error(patches, weight, bias)
        assert error < 1e-6

    def test_half_precision_roundtrip(self, linear_mmproj):
        """Test that roundtrip works in float16 (with slightly higher error)."""
        weight, bias = linear_mmproj
        patches = torch.randn(1, 50, 768).half()
        weight_h = weight.half()
        error = roundtrip_error(patches, weight_h, None)
        # float16 has less precision, allow more error
        assert error < 0.1, f"Half-precision roundtrip error too high: {error}"
