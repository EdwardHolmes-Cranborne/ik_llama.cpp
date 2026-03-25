"""Shared test fixtures for vision roundtrip experiments."""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Model path — skip integration tests if not available
QWEN_MODEL_PATH = "D:/models/Qwen3.5-9B-HF"


def model_available():
    """Check if the Qwen3.5-9B model is downloaded and ready."""
    path = Path(QWEN_MODEL_PATH)
    if not path.exists():
        return False
    safetensors = list(path.glob("*.safetensors"))
    return len(safetensors) >= 4  # Qwen3.5-9B has 4 shards


requires_model = pytest.mark.skipif(
    not model_available(),
    reason=f"Qwen3.5-9B not found at {QWEN_MODEL_PATH}"
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture
def test_image_gradient():
    """A 336x336 RGB gradient image — red increases left-to-right,
    green increases top-to-bottom, blue is constant.
    Useful because spatial structure is trivially verifiable."""
    size = 336
    r = np.linspace(0, 255, size, dtype=np.uint8)
    g = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = r[np.newaxis, :]   # red varies with x
    img[:, :, 1] = g[:, np.newaxis]   # green varies with y
    img[:, :, 2] = 128                 # blue constant
    return Image.fromarray(img)


@pytest.fixture
def test_image_checkerboard():
    """A 336x336 checkerboard pattern — high spatial frequency test."""
    size = 336
    block = 42  # 8x8 blocks
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = [255, 255, 255]
            else:
                img[i, j] = [0, 0, 0]
    return Image.fromarray(img)


@pytest.fixture
def test_image_solid_red():
    """A solid red 336x336 image — baseline for colour tests."""
    img = np.full((336, 336, 3), [255, 0, 0], dtype=np.uint8)
    return Image.fromarray(img)


@pytest.fixture
def dummy_mmproj_weight():
    """A random 4096x1152 linear projection weight (Qwen3.5 dimensions)."""
    torch.manual_seed(42)
    return torch.randn(4096, 1152)


@pytest.fixture
def dummy_patch_embeddings():
    """Random patch embeddings: batch=1, n_patches=100, dim=1152."""
    torch.manual_seed(123)
    return torch.randn(1, 100, 1152)


@pytest.fixture
def model_path():
    """Path to Qwen3.5-9B — for integration tests only."""
    return QWEN_MODEL_PATH
