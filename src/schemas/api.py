"""Pydantic schemas for API requests/responses."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ClaimSubmit(BaseModel):
    """Request body for submitting a claim."""
    text: str = Field(..., description="The claim text to verify")
    source: Optional[str] = Field(None, description="URL where the claim was found")
    source_name: Optional[str] = Field(None, description="Name of the source (e.g., 'BBC News')")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Claim text must not be empty")
        return v.strip()


class ClaimResponse(BaseModel):
    """Response after submitting a claim."""
    id: str
    text: str
    status: str
    created_at: datetime


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
    evidence_count: int = 0
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
    verdict: Optional[Literal["true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"]] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    sub_claims: list[SubClaimResponse] = []
    created_at: datetime
    updated_at: datetime


class ClaimListResponse(BaseModel):
    """Paginated list of claims."""
    claims: list[VerdictResponse]
    total: int
    limit: int
    offset: int
