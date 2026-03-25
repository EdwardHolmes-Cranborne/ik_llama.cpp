"""Tests for image encoder — unit tests (no model) + integration tests."""

import pytest
import torch
from core.encoder import get_vision_token_count


class TestGetVisionTokenCount:
    """Pure maths — no GPU, no model needed."""

    def test_336_standard(self):
        """336×336 with patch=16, merge=2 → 21×21 patches → ~110 tokens."""
        count = get_vision_token_count(336, 336, patch_size=16, merge_size=2)
        # 336/16 = 21, 21*21 = 441, 441/4 = 110.25 → 110
        assert count == 110

    def test_672_double(self):
        """672×672 → 42×42 patches → 441 tokens."""
        count = get_vision_token_count(672, 672, patch_size=16, merge_size=2)
        assert count == 441

    def test_1024_high_res(self):
        """1024×1024 → 64×64 patches → 1024 tokens."""
        count = get_vision_token_count(1024, 1024, patch_size=16, merge_size=2)
        assert count == 1024

    def test_rectangular(self):
        """672×336 → 42×21 patches → 882/4 = 220 tokens."""
        count = get_vision_token_count(672, 336, patch_size=16, merge_size=2)
        assert count == 220

    def test_no_merge(self):
        """Without merge: 336×336 → 441 tokens."""
        count = get_vision_token_count(336, 336, patch_size=16, merge_size=1)
        assert count == 441

    def test_4096_max_res(self):
        """4096×4096 → 256×256 patches → 16384 tokens."""
        count = get_vision_token_count(4096, 4096, patch_size=16, merge_size=2)
        assert count == 16384

    def test_scales_linearly_with_area(self):
        """Doubling both dimensions → 4× tokens."""
        small = get_vision_token_count(336, 336)
        large = get_vision_token_count(672, 672)
        # Should be approximately 4× (exact depends on integer division)
        ratio = large / small
        assert 3.5 < ratio < 4.5

    def test_minimum_image(self):
        """Smallest viable image: 32×32 → 2×2 patches → 1 token."""
        count = get_vision_token_count(32, 32, patch_size=16, merge_size=2)
        assert count == 1
