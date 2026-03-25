"""Pydantic schemas for API requests/responses."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ClaimSubmit(BaseModel):
    """Request body for submitting a claim."""
    text: str = Field(..., description="The claim text to verify")
    source: Optional[str] = Field(None, description="URL where the claim was found")
    source_name: Optional[str] = Field(None, description="Name of the source (e.g., 'BBC News')")
    speaker: Optional[str] = Field(None, description="Person or entity making the claim")
    speaker_description: Optional[str] = Field(None, description="Speaker's title/role (e.g., '45th president of the United States'). Looked up via Wikidata if not provided.")
    claim_date: Optional[str] = Field(None, description="When the claim was made (ISO date or free text)")
    transcript_title: Optional[str] = Field(None, description="Title of the source transcript for topic context")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text must not be empty")
        return v.strip()


class ClaimBatchSubmit(BaseModel):
    """Request body for batch claim submission."""
    claims: list[ClaimSubmit] = Field(..., min_length=1, description="List of claims to verify")


class ClaimBatchResponse(BaseModel):
    """Response after batch claim submission."""
    claims: list["ClaimResponse"]


class ClaimResponse(BaseModel):
    """Response after submitting a claim."""
    id: str
    text: str
    status: str
    created_at: datetime


class EvidenceResponse(BaseModel):
    """A single evidence item with source quality metadata."""
    judge_index: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None
    source_type: str = "web"
    bias: Optional[str] = None
    factual: Optional[str] = None
    tier: Optional[str] = None
    assessment: Optional[str] = None
    is_independent: Optional[bool] = None
    key_point: Optional[str] = None


class CitationResponse(BaseModel):
    """A citation reference linking reasoning text to evidence."""
    index: int
    url: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None


class SubClaimResponse(BaseModel):
    """A verified sub-claim (leaf) or group node in the response."""
    text: str  # leaf: verifiable assertion, group: label
    is_leaf: bool = True
    verdict: Optional[Literal[
        "true", "mostly_true", "mixed", "mostly_false",
        "false", "partially_true", "unverifiable"
    ]] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    evidence: list[EvidenceResponse] = []
    children: list["SubClaimResponse"] = []


# Pydantic v2 needs model_rebuild() for recursive (self-referential) models
SubClaimResponse.model_rebuild()


class VerdictResponse(BaseModel):
    """Full verdict response for a claim."""
    id: str
    text: str
    status: Literal["queued", "pending", "processing", "verified", "flagged"]
    source: Optional[str] = None
    source_name: Optional[str] = None
    speaker: Optional[str] = None
    verdict: Optional[Literal["true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"]] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    citations: list[CitationResponse] = []
    sub_claims: list[SubClaimResponse] = []
    created_at: datetime
    updated_at: datetime


class ClaimListResponse(BaseModel):
    """Paginated list of claims."""
    claims: list[VerdictResponse]
    total: int
    limit: int
    offset: int
