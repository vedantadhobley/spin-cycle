"""Tests for Pydantic schemas."""

import pytest
from src.data.schemas import ClaimSubmit


def test_claim_submit_valid():
    claim = ClaimSubmit(text="The earth is round")
    assert claim.text == "The earth is round"
    assert claim.source is None


def test_claim_submit_with_source():
    claim = ClaimSubmit(text="Water is wet", source="https://example.com")
    assert claim.source == "https://example.com"


def test_claim_submit_empty_text():
    with pytest.raises(Exception):
        ClaimSubmit(text="")
