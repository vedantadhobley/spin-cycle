"""Pydantic schemas for LLM outputs.

These schemas define the EXACT structure expected from each LLM call.
All LLM responses are validated against these schemas before being used.

Benefits:
1. Clear contract: Documents what each LLM call should return
2. Fail-fast: Bad output raises ValidationError immediately
3. Type safety: Downstream code can trust the shape
4. Testability: Can unit test validation separately from LLM calls
"""

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# VERDICT ENUM
# =============================================================================

Verdict = Literal[
    "true",
    "mostly_true",
    "mixed",
    "mostly_false",
    "false",
    "unverifiable",
]


# =============================================================================
# DECOMPOSE OUTPUT
# =============================================================================

class InterestedParties(BaseModel):
    """Entities with potential conflicts of interest.
    
    Used to flag evidence from self-interested sources.
    """
    direct: list[str] = Field(
        default_factory=list,
        description="Organizations immediately involved (e.g., the company being discussed)"
    )
    institutional: list[str] = Field(
        default_factory=list,
        description="Parent/governing bodies (e.g., parent company, government department)"
    )
    affiliated_media: list[str] = Field(
        default_factory=list,
        description="News outlets with ownership ties to interested parties"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why these parties have stake in the claim"
    )

    @classmethod
    def from_legacy(cls, data: Union[list, dict, None]) -> "InterestedParties":
        """Convert legacy formats to structured InterestedParties.
        
        Handles:
        - None → empty
        - list of strings → all go to 'direct'
        - dict with direct/institutional/affiliated_media → proper structure
        """
        if data is None:
            return cls()
        if isinstance(data, list):
            # Legacy: bare list of entity names
            return cls(direct=[str(x) for x in data if x])
        if isinstance(data, dict):
            return cls(
                direct=data.get("direct", []),
                institutional=data.get("institutional", []),
                affiliated_media=data.get("affiliated_media", []),
                reasoning=data.get("reasoning", ""),
            )
        return cls()


class DecomposeOutput(BaseModel):
    """Output from the decompose_claim activity.
    
    Contains a flat list of atomic verifiable facts plus metadata for synthesis.
    This simplified format matches standard fact-checking approaches (SAFE, FActScore).
    """
    thesis: Optional[str] = Field(
        default=None,
        description="One sentence: what is the speaker fundamentally arguing?"
    )
    key_test: Optional[str] = Field(
        default=None,
        description="What must be true for the thesis to hold?"
    )
    structure: str = Field(
        default="simple",
        description="Claim structure type"
    )
    interested_parties: InterestedParties = Field(
        default_factory=InterestedParties,
        description="Entities with potential conflicts of interest"
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Flat list of atomic verifiable facts"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_interested_parties(cls, data: dict) -> dict:
        """Convert legacy interested_parties formats."""
        if "interested_parties" in data:
            raw = data["interested_parties"]
            if not isinstance(raw, InterestedParties):
                data["interested_parties"] = InterestedParties.from_legacy(raw)
        return data

    def has_content(self) -> bool:
        """Check if decomposition produced any verifiable content."""
        return bool(self.facts)


# =============================================================================
# JUDGE OUTPUT
# =============================================================================

class JudgeOutput(BaseModel):
    """Output from the judge_subclaim activity.
    
    Contains the verdict for a single sub-claim based on gathered evidence.
    """
    verdict: Verdict = Field(
        ...,
        description="The verdict for this sub-claim"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0.0-1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how the evidence supports the verdict, including any important context"
    )

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        return max(0.0, min(1.0, v))
    
    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: str) -> str:
        """Handle common verdict variations."""
        if isinstance(v, str):
            v = v.lower().strip()
            # Handle underscores vs spaces
            v = v.replace(" ", "_")
            # Handle common typos/variations
            variations = {
                "partially_true": "mostly_true",
                "partially_false": "mostly_false",
                "uncertain": "unverifiable",
                "unknown": "unverifiable",
                "inconclusive": "unverifiable",
            }
            return variations.get(v, v)
        return v


# =============================================================================
# SYNTHESIZE OUTPUT
# =============================================================================

class SynthesizeOutput(BaseModel):
    """Output from the synthesize_verdict activity.
    
    Contains the combined verdict for multiple sub-claims.
    """
    verdict: Verdict = Field(
        ...,
        description="The combined verdict"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the combined verdict (0.0-1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how the sub-verdicts combine, including any important context"
    )

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range."""
        return max(0.0, min(1.0, v))
    
    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: str) -> str:
        """Handle common verdict variations."""
        if isinstance(v, str):
            v = v.lower().strip()
            v = v.replace(" ", "_")
            variations = {
                "partially_true": "mostly_true",
                "partially_false": "mostly_false",
                "uncertain": "unverifiable",
                "unknown": "unverifiable",
                "inconclusive": "unverifiable",
            }
            return variations.get(v, v)
        return v
