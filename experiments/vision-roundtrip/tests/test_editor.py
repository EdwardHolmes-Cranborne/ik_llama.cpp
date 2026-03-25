"""Tests for text description editor — pure logic, no GPU needed."""

import pytest
from core.editor import parse_edit, apply_edit, apply_structured_edit


class TestParseEdit:
    def test_change_to_format(self):
        find, replace = parse_edit("change 'blue sky' to 'dark storm clouds'")
        assert find == "blue sky"
        assert replace == "dark storm clouds"

    def test_replace_with_format(self):
        find, replace = parse_edit("replace 'arm at his side' with 'arm raised above his head'")
        assert find == "arm at his side"
        assert replace == "arm raised above his head"

    def test_case_insensitive_keyword(self):
        find, replace = parse_edit("Change 'old' to 'new'")
        assert find == "old"
        assert replace == "new"

    def test_unparseable_returns_none(self):
        result = parse_edit("make the sky darker")
        assert result is None

    def test_empty_string(self):
        result = parse_edit("")
        assert result is None

    def test_double_quoted_format(self):
        find, replace = parse_edit('change "sitting down" to "standing up"')
        assert find == "sitting down"
        assert replace == "standing up"


class TestApplyEdit:
    def test_simple_replacement(self):
        original = "A man with his arm at his side stands near a fountain."
        result = apply_edit(original, "change 'arm at his side' to 'arm raised high'")
        assert "arm raised high" in result
        assert "arm at his side" not in result

    def test_case_insensitive_match(self):
        original = "The Blue Sky is clear."
        result = apply_edit(original, "change 'blue sky' to 'grey sky'")
        assert "grey sky" in result

    def test_no_match_appends(self):
        original = "A sunny day in the park."
        result = apply_edit(original, "change 'rainy night' to 'snowy evening'")
        assert "sunny day" in result  # original preserved
        assert "rainy night" in result or "snowy evening" in result  # edit info appended

    def test_multiple_occurrences_all_replaced(self):
        original = "red hat and red shoes on a red carpet"
        result = apply_edit(original, "change 'red' to 'blue'")
        assert result.count("blue") == 3
        assert "red" not in result

    def test_generic_edit_appended(self):
        original = "A man stands in a market."
        result = apply_edit(original, "make the scene more threatening")
        assert "A man stands in a market." in result
        assert "more threatening" in result

    def test_preserves_surrounding_text(self):
        original = "Before. The blue sky shines. After."
        result = apply_edit(original, "change 'blue sky' to 'grey sky'")
        assert result.startswith("Before.")
        assert result.endswith("After.")
        assert "grey sky" in result


class TestApplyStructuredEdit:
    def test_multiple_edits(self):
        original = "A man in a blue shirt with a red hat."
        edits = {
            "blue shirt": "green jacket",
            "red hat": "black cap",
        }
        result = apply_structured_edit(original, edits)
        assert "green jacket" in result
        assert "black cap" in result
        assert "blue shirt" not in result
        assert "red hat" not in result

    def test_empty_edits(self):
        original = "Nothing changes."
        result = apply_structured_edit(original, {})
        assert result == original

    def test_overlapping_edits_applied_sequentially(self):
        original = "A big red ball."
        edits = {
            "big red": "small blue",
            "ball": "cube",
        }
        result = apply_structured_edit(original, edits)
        assert "small blue" in result
        assert "cube" in result
