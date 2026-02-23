"""SQLAlchemy models for claims, verdicts, and evidence."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Text, Enum
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
    status = Column(
        Enum("pending", "processing", "verified", "flagged", name="claim_status"),
        default="pending",
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    sub_claims = relationship("SubClaim", back_populates="claim", cascade="all, delete-orphan")
    verdict = relationship("Verdict", back_populates="claim", uselist=False, cascade="all, delete-orphan")


class SubClaim(Base):
    __tablename__ = "sub_claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey("claims.id"), nullable=False)
    text = Column(Text, nullable=False)
    verdict = Column(
        Enum("true", "false", "partially_true", "unverifiable", name="sub_claim_verdict"),
        nullable=True,
    )
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)

    claim = relationship("Claim", back_populates="sub_claims")
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
    supports_claim = Column(Boolean, nullable=True)
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
    reasoning_chain = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    claim = relationship("Claim", back_populates="verdict")
