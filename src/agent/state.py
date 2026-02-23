"""LangGraph verification agent state schema."""

from typing import TypedDict, Optional, Literal
from dataclasses import dataclass, field


class SubClaim(TypedDict):
    """A single verifiable sub-claim decomposed from the original."""
    text: str
    verdict: Optional[Literal["true", "false", "partially_true", "unverifiable"]]
    confidence: Optional[float]
    reasoning: Optional[str]
    evidence: list[dict]


class VerificationState(TypedDict):
    """State that flows through the LangGraph verification graph.

    This is the central state object — every node reads from and writes to it.
    LangGraph persists this between steps, enabling cycles and crash recovery.
    """
    # Input
    claim_text: str
    source_url: Optional[str]
    source_name: Optional[str]

    # Decomposition
    sub_claims: list[SubClaim]

    # Research
    current_sub_claim_index: int
    research_iterations: int
    max_research_iterations: int

    # Evidence
    evidence: list[dict]

    # Verdict
    verdict: Optional[Literal["true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"]]
    confidence: Optional[float]
    reasoning_chain: list[str]

    # Control flow
    needs_more_research: bool
    error: Optional[str]
