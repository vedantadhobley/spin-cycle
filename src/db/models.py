"""SQLAlchemy models for claims, verdicts, and evidence."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Integer, Text, Enum, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Claim(Base):
    __tablename__ = "claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=False)
    source_url = Column(String(2048), nullable=True)
    source_name = Column(String(256), nullable=True)
    speaker = Column(String(256), nullable=True)
    speaker_description = Column(String(512), nullable=True)  # Wikidata role/title (e.g. "45th president")
    claim_date = Column(String(64), nullable=True)  # when the claim was made (from transcript, article, etc.)
    transcript_title = Column(String(512), nullable=True)  # source transcript title for topic context
    status = Column(
        Enum("queued", "pending", "processing", "verified", "flagged", name="claim_status"),
        default="pending",
        nullable=False,
    )
    # Decompose rubric fields
    normalized_claim = Column(Text, nullable=True)
    normalization_changes = Column(JSONB, nullable=True)
    thesis = Column(Text, nullable=True)
    key_test = Column(Text, nullable=True)
    claim_structure = Column(String(64), nullable=True)
    claim_analysis = Column(Text, nullable=True)
    structure_justification = Column(Text, nullable=True)
    interested_parties_reasoning = Column(Text, nullable=True)
    wikidata_context = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    sub_claims = relationship("SubClaim", back_populates="claim", cascade="all, delete-orphan")
    verdict = relationship("Verdict", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    interested_parties = relationship("InterestedParty", back_populates="claim", cascade="all, delete-orphan")


class SubClaim(Base):
    __tablename__ = "sub_claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("sub_claims.id"), nullable=True)
    is_leaf = Column(Boolean, default=True, nullable=False)
    text = Column(Text, nullable=False)  # leaf: verifiable assertion, group: decomposed text
    verdict = Column(
        Enum("true", "false", "partially_true", "unverifiable",
             "mostly_true", "mixed", "mostly_false",
             name="sub_claim_verdict"),
        nullable=True,
    )
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Decompose metadata
    categories = Column(JSONB, nullable=True)
    seed_queries = Column(JSONB, nullable=True)
    category_rationale = Column(Text, nullable=True)
    # Judge rubric
    judge_rubric = Column(JSONB, nullable=True)

    claim = relationship("Claim", back_populates="sub_claims")
    parent = relationship("SubClaim", remote_side="SubClaim.id", backref="children")
    evidence = relationship("Evidence", back_populates="sub_claim", cascade="all, delete-orphan")


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sub_claim_id = Column(UUID(as_uuid=True), ForeignKey("sub_claims.id"), nullable=False)
    source_type = Column(
        Enum("web", "wikipedia", "news_api", name="evidence_source_type"),
        nullable=False,
    )
    source_url = Column(String(2048), nullable=True)
    content = Column(Text, nullable=True)
    title = Column(String(512), nullable=True)
    domain = Column(String(256), nullable=True)
    bias = Column(String(64), nullable=True)
    factual = Column(String(64), nullable=True)
    tier = Column(String(64), nullable=True)
    judge_index = Column(Integer, nullable=True)
    assessment = Column(String(32), nullable=True)
    is_independent = Column(Boolean, nullable=True)
    key_point = Column(Text, nullable=True)
    retrieved_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    sub_claim = relationship("SubClaim", back_populates="evidence")


class Verdict(Base):
    __tablename__ = "verdicts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False, unique=True)
    verdict = Column(
        Enum("true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable", name="verdict_type"),
        nullable=False,
    )
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    reasoning_chain = Column(JSONB, nullable=True)
    citations = Column(JSONB, nullable=True)
    synthesis_rubric = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    claim = relationship("Claim", back_populates="verdict")


class TranscriptRecord(Base):
    """A stored transcript with cleaned display text."""
    __tablename__ = "transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String(2048), nullable=False, unique=True)
    title = Column(String(512), nullable=False)
    date = Column(String(64), nullable=True)
    speakers = Column(JSONB, nullable=False)  # list of speaker names
    word_count = Column(Integer, nullable=False)
    segment_count = Column(Integer, nullable=False)
    display_text = Column(Text, nullable=False)  # cleaned, merged same-speaker segments
    status = Column(String(32), default="queued", nullable=False)  # queued → extracting → verifying → complete → failed
    # Thesis extraction v2 fields
    segments_data = Column(JSONB, nullable=True)  # list of NumberedSegment dicts
    source_format = Column(String(32), default="revcom", nullable=True)  # "revcom", "raw_text"
    speaker_aliases = Column(JSONB, nullable=True)  # canonical → variants
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    transcript_claims = relationship("TranscriptClaim", back_populates="transcript", cascade="all, delete-orphan")


class TranscriptClaim(Base):
    """A claim extracted from a transcript, linking extraction to verification."""
    __tablename__ = "transcript_claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("transcripts.id"), nullable=False)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=True)  # set when sent to verification
    claim_text = Column(Text, nullable=False)  # decontextualized — pronouns resolved
    original_quote = Column(Text, nullable=False)  # speaker's exact words — used for inline highlighting
    speaker = Column(String(256), nullable=False)
    claim_type = Column(String(64), nullable=True)
    # Extraction rubric fields
    worth_checking = Column(Boolean, nullable=False, default=True)
    skip_reason = Column(String(64), nullable=True)
    checkable = Column(Boolean, nullable=True)
    checkability_rationale = Column(Text, nullable=True)
    is_restatement = Column(Boolean, nullable=True, default=False)
    segment_gist = Column(Text, nullable=True)
    # Thesis extraction v2 fields
    supporting_references = Column(JSONB, nullable=True)  # list of {segment_index, excerpt}
    topic = Column(String(64), nullable=True)
    thesis_version = Column(Integer, default=1, nullable=True)  # 1=old batch, 2=thesis
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    transcript = relationship("TranscriptRecord", back_populates="transcript_claims")
    claim = relationship("Claim")


class InterestedParty(Base):
    """An entity with potential conflict of interest related to a claim."""
    __tablename__ = "interested_parties"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False)
    entity_name = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False)  # direct | institutional | affiliated_media | wikidata_expanded
    source = Column(String(32), nullable=False)  # llm | ner | speaker | wikidata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    claim = relationship("Claim", back_populates="interested_parties")

    __table_args__ = (
        Index("ix_interested_parties_entity_claim", "entity_name", "claim_id"),
    )


class SourceRating(Base):
    """Cached media bias/factual ratings from MBFC and similar services."""
    __tablename__ = "source_ratings"

    domain = Column(String(256), primary_key=True)  # e.g., "reuters.com"
    bias = Column(
        Enum(
            "extreme-left", "left", "left-center", "center",
            "right-center", "right", "extreme-right", "satire", "conspiracy-pseudoscience",
            name="bias_rating"
        ),
        nullable=True,
    )
    bias_score = Column(Float, nullable=True)  # Numeric bias: -10 (far left) to +10 (far right)
    factual_reporting = Column(
        Enum("very-high", "high", "mostly-factual", "mixed", "low", "very-low", name="factual_rating"),
        nullable=True,
    )
    credibility = Column(
        Enum("high", "medium", "low", name="credibility_rating"),
        nullable=True,
    )
    country = Column(String(128), nullable=True)  # e.g., "United Kingdom", "Russia"
    media_type = Column(String(128), nullable=True)  # e.g., "News Wire", "TV Station", "Newspaper"
    ownership = Column(String(256), nullable=True)  # e.g., "State-Funded", "Thomson Reuters Corp"
    traffic = Column(String(64), nullable=True)  # e.g., "High Traffic", "Medium Traffic"
    mbfc_url = Column(String(512), nullable=True)  # Link to MBFC page for reference
    raw_data = Column(JSONB, nullable=True)  # Store any extra scraped fields
    scraped_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class WikidataCache(Base):
    """Cached Wikidata entity relationships for conflict-of-interest detection.
    
    Stores ownership chains, media holdings, political affiliations, etc.
    TTL: 7 days (entities change less frequently than news bias ratings).
    """
    __tablename__ = "wikidata_cache"

    entity_name = Column(String(256), primary_key=True)  # Search term, e.g., "Acme Corp"
    qid = Column(String(32), nullable=True)  # Wikidata QID, e.g., "Q312", None if not found
    relationships = Column(JSONB, nullable=True)  # Full get_ownership_chain() result
    scraped_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

