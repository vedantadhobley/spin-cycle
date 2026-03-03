"""Pydantic schemas for structured data validation.

This package contains:
- api.py: Request/response schemas for the REST API
- llm_outputs.py: Schemas for validating LLM outputs

All LLM outputs are validated against Pydantic models BEFORE being used
by the rest of the system. This provides a clear contract and catches
malformed outputs early.
"""

from src.schemas.llm_outputs import (
    # Normalize
    NormalizeOutput,
    # Decompose
    InterestedParties,
    AtomicFact,
    DecomposeOutput,
    # Judge
    Verdict,
    JudgeOutput,
    # Synthesize
    SynthesizeOutput,
)

from src.schemas.api import (
    ClaimSubmit,
    ClaimBatchSubmit,
    ClaimBatchResponse,
    ClaimResponse,
    SubClaimResponse,
    VerdictResponse,
    ClaimListResponse,
)

__all__ = [
    # LLM outputs
    "NormalizeOutput",
    "InterestedParties",
    "AtomicFact",
    "DecomposeOutput",
    "Verdict",
    "JudgeOutput",
    "SynthesizeOutput",
    # API
    "ClaimSubmit",
    "ClaimBatchSubmit",
    "ClaimBatchResponse",
    "ClaimResponse",
    "SubClaimResponse",
    "VerdictResponse",
    "ClaimListResponse",
]
