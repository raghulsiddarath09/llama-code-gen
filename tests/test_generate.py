"""Tests for prompt construction."""
from src.generate import build_prompt


def test_matches_training_format():
    # Must be byte-identical to src.data.format_prompt's no-input branch.
    from src.data import format_prompt
    assert build_prompt("Reverse a string") == format_prompt(
        {"instruction": "Reverse a string", "input": ""})


def test_ends_at_response_marker():
    assert build_prompt("t").endswith("### Response:\n")


def test_contains_instruction():
    assert "Reverse a string" in build_prompt("Reverse a string")
