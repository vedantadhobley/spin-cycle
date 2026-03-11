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
# NORMALIZE OUTPUT
# =============================================================================

class NormalizeOutput(BaseModel):
    """Normalized claim with loaded language neutralized and opinions separated."""
    normalized_claim: str = Field(
        ...,
        description="Claim rewritten with neutral, researchable language"
    )
    changes: list[str] = Field(
        default_factory=list,
        description="What was changed and why"
    )


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


class AtomicFact(BaseModel):
    """A single atomic verifiable fact with evidence-need categories and seed queries.

    Categories tell the research agent what KIND of evidence to seek:
    - QUANTITATIVE: data portals, official statistics
    - ATTRIBUTION: exact quotes, transcripts, original statements
    - LEGISLATIVE: bill text, roll call votes, legislative records
    - CAUSAL: mechanism evidence, alternative explanations
    - COMPARATIVE: each comparison target separately, rankings
    - CURRENT_EVENTS: news sources, recent reporting
    - SCIENTIFIC: peer-reviewed sources, meta-analyses
    - GENERAL: standard web search (default)

    Seed queries are LLM-written search queries tailored to find the
    specific evidence this fact needs. They're fired in the seed phase
    before the research agent starts, so the agent begins with relevant
    evidence already in hand.
    """
    text: str = Field(..., description="The atomic fact text")
    categories: list[str] = Field(
        default_factory=lambda: ["GENERAL"],
        description="Evidence-need categories for this fact",
    )
    seed_queries: list[str] = Field(
        default_factory=list,
        description="2-4 targeted search queries to find evidence for this fact",
    )

    @field_validator("categories", mode="before")
    @classmethod
    def ensure_categories(cls, v):
        """Default to GENERAL if empty or missing."""
        if not v:
            return ["GENERAL"]
        return v


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
    facts: list[AtomicFact] = Field(
        default_factory=list,
        description="Flat list of atomic verifiable facts with categories"
    )

    @field_validator("facts", mode="before")
    @classmethod
    def normalize_facts(cls, v):
        """Handle both plain strings and fact objects from LLM output."""
        if not v:
            return v
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append({"text": item, "categories": ["GENERAL"]})
            elif isinstance(item, dict) and "text" not in item:
                # Shouldn't happen, but guard against it
                normalized.append({"text": str(item), "categories": ["GENERAL"]})
            else:
                normalized.append(item)
        return normalized

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
# SUBCLAIM QUALITY CHECK (post-decompose validator)
# =============================================================================

class SubclaimQualityCheck(BaseModel):
    """Output from the post-decompose semantic quality validator.

    Detects two structural issues the LLM can't self-enforce during generation:
    1. Semantic duplicates (logically equivalent sub-claims)
    2. Group enumeration (individual members instead of group-level claim)
    """
    has_duplicates: bool = Field(
        ..., description="Whether any sub-claims are logically equivalent"
    )
    duplicate_pairs: list[list[int]] = Field(
        default_factory=list,
        description="Pairs of indices that are semantic duplicates, e.g. [[0, 2], [1, 3]]",
    )
    has_enumeration: bool = Field(
        ..., description="Whether sub-claims enumerate members of a named group"
    )
    enumerated_indices: list[int] = Field(
        default_factory=list,
        description="Indices of sub-claims that should be consolidated into a group claim",
    )
    reasoning: str = Field(
        default="", description="Explanation of findings"
    )


# =============================================================================
# JUDGE OUTPUT (rubric-based with explicit reasoning chain)
# =============================================================================

# Direction of evidence assessment (Step 3)
EvidenceDirection = Literal[
    "clearly_supports",
    "leans_supports",
    "genuinely_mixed",
    "leans_contradicts",
    "clearly_contradicts",
    "insufficient",
]


class EvidenceAssessment(BaseModel):
    """Assessment of a key evidence item (Step 2 of judge rubric)."""
    source_index: int = Field(
        ..., description="Evidence item number from the provided list"
    )
    assessment: Literal["supports", "contradicts", "neutral"] = Field(
        ..., description="Does this evidence support or contradict the claim?"
    )
    is_independent: bool = Field(
        ..., description="Is this source independent from the claim subject? "
        "False if source IS the claim subject, quotes the claim subject, "
        "or has ownership ties to the claim subject."
    )
    key_point: str = Field(
        ..., description="1-2 sentences: what does this evidence say?"
    )

    @field_validator("assessment", mode="before")
    @classmethod
    def normalize_assessment(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.lower().strip()
            # Strip parenthetical qualifiers: "supports (historical)" → "supports"
            if "(" in v:
                v = v[:v.index("(")].strip()
            # Handle common variations
            mapping = {
                "support": "supports",
                "contradict": "contradicts",
                "neither": "neutral",
                "irrelevant": "neutral",
                "n/a": "neutral",
            }
            return mapping.get(v, v)
        return v


class JudgeOutput(BaseModel):
    """Rubric-based judge output with explicit reasoning chain.

    Five-step rubric:
      1. Interpret the claim (charitable restatement)
      2. Triage key evidence (3-5 items with independence check)
      3. Assess direction (based on independent evidence only)
      4. Assess precision (attribution, quantifiers, arithmetic)
      5. Render verdict (derived from steps 3+4)
    """

    # Step 1 — Interpret the claim
    claim_interpretation: str = Field(
        ..., description="Charitable restatement of what the claim is asking"
    )

    # Step 2 — Triage key evidence
    key_evidence: list[EvidenceAssessment] = Field(
        ..., description="Assessment of the 3-5 most relevant evidence items"
    )

    # Step 3 — Assess direction
    evidence_direction: EvidenceDirection = Field(
        ..., description="Overall direction of independent evidence"
    )
    direction_reasoning: str = Field(
        ..., description="2-3 sentences explaining the direction assessment"
    )

    # Step 4 — Assess precision
    precision_assessment: str = Field(
        ..., description="How precise is the claim? Where do specifics "
        "match or diverge from evidence?"
    )

    # Step 5 — Render verdict
    verdict: Verdict = Field(
        ..., description="The verdict for this sub-claim"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the verdict (0.0-1.0)"
    )
    reasoning: str = Field(
        ..., description="Public-facing explanation of the verdict"
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

    @field_validator("evidence_direction", mode="before")
    @classmethod
    def normalize_direction(cls, v: str) -> str:
        """Handle common direction variations."""
        if isinstance(v, str):
            v = v.lower().strip().replace(" ", "_").replace("-", "_")
            mapping = {
                "supports": "clearly_supports",
                "contradicts": "clearly_contradicts",
                "mixed": "genuinely_mixed",
                "unclear": "insufficient",
                "unknown": "insufficient",
                "not_enough": "insufficient",
            }
            return mapping.get(v, v)
        return v


# =============================================================================
# SYNTHESIZE OUTPUT (rubric-based with thesis evaluation)
# =============================================================================

# Subclaim role classification (Step 2)
SubclaimRole = Literal["core_assertion", "supporting_detail", "background_context"]


class SubclaimWeight(BaseModel):
    """Classification of a subclaim's importance (Step 2 of synthesize rubric)."""
    subclaim_index: int = Field(
        ..., description="Which sub-verdict this refers to (1-indexed)"
    )
    role: SubclaimRole = Field(
        ..., description="Role of this subclaim in the overall argument"
    )
    brief_reason: str = Field(
        ..., description="Why this classification (1 sentence)"
    )

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.lower().strip().replace(" ", "_").replace("-", "_")
            mapping = {
                "core": "core_assertion",
                "main": "core_assertion",
                "primary": "core_assertion",
                "supporting": "supporting_detail",
                "detail": "supporting_detail",
                "secondary": "supporting_detail",
                "background": "background_context",
                "context": "background_context",
            }
            return mapping.get(v, v)
        return v


class SynthesizeOutput(BaseModel):
    """Rubric-based synthesis with explicit thesis evaluation.

    Four-step rubric:
      1. Identify the thesis (one-sentence restatement)
      2. Classify each subclaim (core/supporting/background)
      3. Does the thesis survive? (based on core assertion verdicts)
      4. Render verdict (derived from steps 2+3)
    """

    # Step 1 — Identify the thesis
    thesis_restatement: str = Field(
        ..., description="One sentence: what is the speaker arguing?"
    )

    # Step 2 — Classify each subclaim
    subclaim_weights: list[SubclaimWeight] = Field(
        ..., description="Role classification for each subclaim"
    )

    # Step 3 — Does the thesis survive?
    thesis_survives: bool = Field(
        ..., description="Does the thesis hold given core assertion verdicts?"
    )

    # Step 4 — Render verdict
    verdict: Verdict = Field(
        ..., description="The combined verdict"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the combined verdict (0.0-1.0)"
    )
    reasoning: str = Field(
        ..., description="Public-facing explanation (never reference "
        "sub-claim numbers or internal process)"
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
