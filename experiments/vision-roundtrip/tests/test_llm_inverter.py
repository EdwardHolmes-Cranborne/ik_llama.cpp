"""Tests for LLM inverter — unit tests with mock components on CPU."""

import pytest
import torch
from core.llm_inverter import find_embed_table, LLMInversionResult


class MockEmbedding(torch.nn.Module):
    """Mock embedding table."""
    def __init__(self, vocab_size=1000, dim=64):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.embed_tokens(x)


class TestFindEmbedTable:
    def test_finds_embed_tokens(self):
        model = MockEmbedding()
        table = find_embed_table(model)
        assert table is not None
        assert isinstance(table, torch.nn.Embedding)

    def test_raises_on_missing(self):
        model = torch.nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Could not find"):
            find_embed_table(model)


class TestLLMInversionResult:
    def test_dataclass_fields(self):
        result = LLMInversionResult(
            modified_embeddings=torch.randn(1, 10, 64),
            original_embeddings=torch.randn(1, 10, 64),
            loss_history=[1.0, 0.8, 0.6],
            best_loss=0.6,
            per_token_deltas=torch.randn(10),
            total_delta=5.0,
        )
        assert result.modified_embeddings.shape == (1, 10, 64)
        assert len(result.loss_history) == 3
        assert result.best_loss == 0.6
        assert result.total_delta == 5.0

    def test_delta_between_modified_and_original(self):
        orig = torch.randn(1, 10, 64)
        modified = orig.clone()
        modified[0, 3, :] += 1.0  # change one token

        deltas = (modified - orig).norm(dim=-1).squeeze(0)
        assert deltas[3] > 0.5  # the changed token has a large delta
        assert deltas[0] < 1e-6  # unchanged tokens have zero delta
