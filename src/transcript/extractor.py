"""Extract verifiable claims from parsed transcripts.

Takes a Transcript (from fetcher.py) and uses the LLM to identify factual
assertions with contextual bracketing.  The LLM processes the transcript
segment-by-segment using a structured per-segment output schema — the
`assertion_count` field forces the model to scan each segment before listing
claims, preventing it from skipping sections.

Processing strategy:
- Single call for small transcripts (≤30 segments)
- Segment batching with overlap context for larger transcripts:
  each batch gets its target segments + a few overlap segments before/after
  for bracket resolution and restatement detection
- Each batch is a separate Temporal activity for full UI visibility
- Programmatic consistency enforcement on worth_checking
- Validate assertion_count matches actual claims per segment
- Cross-batch deduplication catches restatements across boundaries
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.llm import invoke_llm, validate_extraction
from src.prompts.extraction import EXTRACTION_SYSTEM, EXTRACTION_USER
from src.transcript.fetcher import Transcript, TranscriptSegment
from src.utils.logging import log, get_logger

MODULE = "transcript.extractor"
logger = get_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_TARGET_WORDS = 3000  # target words per batch (output time scales with input words)
BATCH_MAX_SEGMENTS = 40    # hard cap on segments per batch (structured output limit)
OVERLAP_SEGMENTS = 3       # context segments before/after each batch boundary


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM output
# ---------------------------------------------------------------------------

SkipReason = Literal[
    "not_argumentative",
    "not_checkable",
    "low_consequence",
    "common_knowledge",
    "restatement",
    "future_prediction",
]

Consequence = Literal["high", "low", "none"]


class ExtractedClaim(BaseModel):
    claim_text: str = Field(..., description="Contextualized claim with [brackets]")
    original_quote: str = Field(..., description="Speaker's exact words")
    speaker: str = Field(..., description="Speaker name from transcript")
    timestamp: str = Field(..., description="MM:SS timestamp")
    claim_type: str = Field("other", description="quantitative|historical|causal|comparative|attribution|other")
    argument_summary: Optional[str] = Field(None, description="What argument does citing this fact support?")
    supports_argument: bool = Field(..., description="Is this fact deployed to persuade?")
    checkable: bool = Field(..., description="Could independent data confirm or deny?")
    checkability_rationale: str = Field(default="", description="Why checkable or not (1 sentence)")
    consequence_if_wrong: Consequence = Field(..., description="If wrong, does it matter?")
    consequence_rationale: str = Field(default="", description="Why high/low/none consequence (1 sentence)")
    worth_checking: bool = Field(..., description="Final call: checkable AND consequence=high")
    skip_reason: Optional[SkipReason] = Field(None, description="Why not worth checking")


class SegmentExtraction(BaseModel):
    timestamp: str = Field(..., description="MM:SS timestamp of this segment")
    speaker: str = Field(..., description="Speaker name")
    segment_gist: str = Field(default="", description="One sentence: what is the speaker arguing in this segment?")
    assertion_count: int = Field(..., description="Total factual assertions found in this segment")
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    segments: list[SegmentExtraction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataclass for final output
# ---------------------------------------------------------------------------

@dataclass
class TranscriptClaim:
    """A claim extracted from a transcript, ready for the verification pipeline."""
    claim_text: str           # contextualized with [brackets]
    original_quote: str       # speaker's exact words
    speaker: str
    timestamp: str
    timestamp_secs: float
    claim_type: str
    source_url: str           # transcript URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_segment_manifest(segments: list[TranscriptSegment]) -> str:
    """Build a numbered manifest of segments for the LLM prompt."""
    lines = []
    for i, seg in enumerate(segments, 1):
        word_count = len(seg.text.split())
        lines.append(f"{i}. {seg.speaker} ({seg.timestamp}) — {word_count} words")
    return "\n".join(lines)


def _format_transcript(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments into text for the LLM."""
    parts = []
    for seg in segments:
        parts.append(f"{seg.speaker} ({seg.timestamp}):\n{seg.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------

def build_batches(segment_word_counts: list[int]) -> list[dict]:
    """Build batch specs targeting ~BATCH_TARGET_WORDS per batch.

    Splits at segment boundaries so no segment is cut mid-text. Each batch
    accumulates segments until adding the next would exceed the word target,
    then starts a new batch. Hard cap at BATCH_MAX_SEGMENTS per batch.

    Each batch spec has:
        target_start, target_end: segment indices for the manifest
        text_start, text_end: segment indices for transcript text (includes overlap)

    Args:
        segment_word_counts: word count for each segment, in order.
    """
    num_segments = len(segment_word_counts)
    total_words = sum(segment_word_counts)

    # Small transcript — single batch, no overlap
    if total_words <= BATCH_TARGET_WORDS and num_segments <= BATCH_MAX_SEGMENTS:
        return [{
            "target_start": 0,
            "target_end": num_segments,
            "text_start": 0,
            "text_end": num_segments,
        }]

    # Build batches by word count, cutting at segment boundaries.
    # Cut AFTER the current segment once the batch has reached the target.
    batches = []
    batch_start = 0
    batch_words = 0

    for i, wc in enumerate(segment_word_counts):
        batch_words += wc
        at_end = (i == num_segments - 1)
        segments_in_batch = i - batch_start + 1

        should_cut = (
            (batch_words >= BATCH_TARGET_WORDS or segments_in_batch >= BATCH_MAX_SEGMENTS)
            and not at_end
        )

        if should_cut:
            target_end = i + 1
            text_start = max(0, batch_start - OVERLAP_SEGMENTS)
            text_end = min(num_segments, target_end + OVERLAP_SEGMENTS)
            batches.append({
                "target_start": batch_start,
                "target_end": target_end,
                "text_start": text_start,
                "text_end": text_end,
            })
            batch_start = target_end
            batch_words = 0

    # Flush remaining segments
    if batch_start < num_segments:
        text_start = max(0, batch_start - OVERLAP_SEGMENTS)
        batches.append({
            "target_start": batch_start,
            "target_end": num_segments,
            "text_start": text_start,
            "text_end": num_segments,
        })

    return batches


# ---------------------------------------------------------------------------
# Consistency enforcement
# ---------------------------------------------------------------------------

# Patterns that indicate future predictions/promises (case-insensitive)
_FUTURE_PATTERNS = re.compile(
    r"\b(?:"
    r"will (?:go|be|have|increase|decrease|rise|fall|grow|create|bring|save|get)"
    r"|going to (?:be|have|do|create|bring|see)"
    r"|anticipat(?:e|ing|ed) .{0,30}(?:growth|increase|improvement)"
    r"|expect(?:s|ing)? .{0,30}(?:growth|increase|improvement)"
    r")\b",
    re.IGNORECASE,
)


def _is_future_prediction(claim: ExtractedClaim) -> bool:
    """Detect future predictions/promises that can't be verified yet."""
    text = claim.claim_text
    # Check both claim_text and original_quote
    if _FUTURE_PATTERNS.search(text) or _FUTURE_PATTERNS.search(claim.original_quote):
        return True
    return False


def _enforce_consistency(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    """Override worth_checking based on the structured reasoning fields.

    The prompt says worth_checking should be true only when checkable AND
    consequence_if_wrong=high.  The model doesn't always follow that rule,
    so we enforce it programmatically.

    Additionally: if argument_summary is null/empty, supports_argument must
    be false — the model couldn't articulate an argument.

    Future predictions are caught by regex as a safety net — the prompt asks
    the model to skip these, but it doesn't always comply.
    """
    for claim in claims:
        if not claim.argument_summary or not claim.argument_summary.strip():
            claim.supports_argument = False

        # Catch future predictions the model marked as worth checking
        if claim.worth_checking and _is_future_prediction(claim):
            claim.worth_checking = False
            claim.checkable = False
            claim.skip_reason = "future_prediction"
            continue

        should_check = (
            claim.checkable
            and claim.consequence_if_wrong == "high"
        )

        if claim.worth_checking and not should_check:
            claim.worth_checking = False
            if not claim.checkable:
                claim.skip_reason = "not_checkable"
            else:
                claim.skip_reason = "low_consequence"

        elif not claim.worth_checking and should_check:
            claim.worth_checking = True
            claim.skip_reason = None

    return claims


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_segment_coverage(
    output: ExtractionOutput,
    expected_segments: list[TranscriptSegment],
) -> None:
    """Log warnings if the model skipped segments or assertion counts mismatch."""
    output_timestamps = {seg.timestamp for seg in output.segments}
    expected_timestamps = {seg.timestamp for seg in expected_segments}

    missing = expected_timestamps - output_timestamps
    if missing:
        log.warning(logger, MODULE, "segments_missing",
                    f"Model skipped {len(missing)} segments",
                    missing_timestamps=sorted(missing))

    for seg in output.segments:
        actual = len(seg.claims)
        if seg.assertion_count != actual:
            log.warning(logger, MODULE, "assertion_count_mismatch",
                        f"Segment {seg.speaker} ({seg.timestamp}): "
                        f"assertion_count={seg.assertion_count} but {actual} claims listed")


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_claims(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    """Remove duplicate claims.

    Two claims are duplicates if they share the same speaker and their
    original quotes overlap significantly (>60% word overlap).
    """
    unique: list[ExtractedClaim] = []

    for claim in claims:
        claim_words = set(claim.original_quote.lower().split())
        is_dup = False

        for existing in unique:
            if claim.speaker != existing.speaker:
                continue
            existing_words = set(existing.original_quote.lower().split())
            if not claim_words or not existing_words:
                continue
            overlap = len(claim_words & existing_words)
            smaller = min(len(claim_words), len(existing_words))
            if smaller > 0 and overlap / smaller > 0.6:
                is_dup = True
                break

        if not is_dup:
            unique.append(claim)

    return unique


# ---------------------------------------------------------------------------
# Post-hoc consistency check (permissive — warnings only)
# ---------------------------------------------------------------------------

def _validate_extraction_consistency(output: ExtractionOutput) -> None:
    """Log warnings for rubric inconsistencies. Permissive — never rejects."""
    for seg in output.segments:
        actual = len(seg.claims)

        # assertion_count doesn't match len(claims)
        if seg.assertion_count != actual:
            log.warning(logger, MODULE, "assertion_count_mismatch",
                        f"Segment {seg.speaker} ({seg.timestamp}): "
                        f"assertion_count={seg.assertion_count} but {actual} claims",
                        speaker=seg.speaker, timestamp=seg.timestamp)

        # Segment has claims but empty segment_gist
        if seg.claims and not seg.segment_gist.strip():
            log.warning(logger, MODULE, "missing_segment_gist",
                        f"Segment {seg.speaker} ({seg.timestamp}) has "
                        f"{actual} claims but empty segment_gist",
                        speaker=seg.speaker, timestamp=seg.timestamp)

        for claim in seg.claims:
            # checkable=True but empty checkability_rationale
            if claim.checkable and not claim.checkability_rationale.strip():
                log.warning(logger, MODULE, "missing_checkability_rationale",
                            f"Claim marked checkable but no rationale: "
                            f"'{claim.claim_text[:60]}...'")

            # consequence=high but empty consequence_rationale
            if claim.consequence_if_wrong == "high" and not claim.consequence_rationale.strip():
                log.warning(logger, MODULE, "missing_consequence_rationale",
                            f"Claim has high consequence but no rationale: "
                            f"'{claim.claim_text[:60]}...'")

            # worth_checking doesn't match checkable AND consequence=high
            expected_worth = claim.checkable and claim.consequence_if_wrong == "high"
            if claim.worth_checking != expected_worth:
                log.warning(logger, MODULE, "worth_checking_mismatch",
                            f"worth_checking={claim.worth_checking} but "
                            f"checkable={claim.checkable}, "
                            f"consequence={claim.consequence_if_wrong}: "
                            f"'{claim.claim_text[:60]}...'")


# ---------------------------------------------------------------------------
# Core extraction — single batch
# ---------------------------------------------------------------------------

async def extract_batch(
    text_segments: list[TranscriptSegment],
    target_segments: list[TranscriptSegment],
    batch_label: str | None = None,
) -> list[ExtractedClaim]:
    """Extract claims from a single batch of segments.

    Args:
        text_segments: All segments included in the transcript text
            (target segments + overlap context before/after).
        target_segments: Only the segments that appear in the manifest.
            The model outputs entries only for these.
        batch_label: Optional "batch N of M" string for logging.
    """
    transcript_text = _format_transcript(text_segments)
    segment_manifest = _build_segment_manifest(target_segments)
    speakers = ", ".join(dict.fromkeys(s.speaker for s in target_segments))
    target_range = f"{target_segments[0].timestamp}–{target_segments[-1].timestamp}"
    text_range = f"{text_segments[0].timestamp}–{text_segments[-1].timestamp}"
    word_count = sum(len(s.text.split()) for s in text_segments)

    has_overlap = len(text_segments) != len(target_segments)
    context_note = (
        f"Transcript text covers {text_range} ({word_count} words, "
        f"{len(text_segments)} segments). "
        f"Manifest lists {len(target_segments)} segments to process "
        f"({target_range}). Speakers: {speakers}."
    )
    if has_overlap:
        context_note += (
            f" Additional segments outside the manifest are included for "
            f"context only — do NOT output entries for them."
        )
    if batch_label:
        context_note += f" {batch_label}."

    log.info(logger, MODULE, "extracting",
             f"Extracting from {len(target_segments)} segments "
             f"({target_range}, {word_count} words text"
             f"{', +' + str(len(text_segments) - len(target_segments)) + ' overlap' if has_overlap else ''})"
             f"{' ' + batch_label if batch_label else ''}")

    async def _call(temperature: float = 0):
        return await invoke_llm(
            system_prompt=EXTRACTION_SYSTEM.format(
                current_date=date.today().isoformat(),
            ),
            user_prompt=EXTRACTION_USER.format(
                transcript_text=transcript_text,
                segment_manifest=segment_manifest,
                context_note=context_note,
            ),
            schema=ExtractionOutput,
            semantic_validator=validate_extraction,
            temperature=temperature,
            max_tokens=16384,
            activity_name="extract_claims",
        )

    # Call LLM, retry once if segment coverage is below 50%
    output = await _call()
    covered = len(output.segments)
    expected = len(target_segments)
    if expected > 1 and covered < expected * 0.5:
        log.warning(logger, MODULE, "low_coverage_retry",
                    f"Model returned {covered}/{expected} segments, "
                    f"retrying with temperature=0.3")
        output = await _call(temperature=0.3)

    # Post-hoc consistency check (permissive warnings)
    _validate_extraction_consistency(output)

    # Rubric summary logging
    segment_count = len(output.segments)
    gist_count = sum(1 for s in output.segments if s.segment_gist.strip())
    total_claims = sum(len(s.claims) for s in output.segments)
    log.info(logger, MODULE, "rubric_summary",
             f"Extraction rubric: {segment_count} segments, "
             f"{gist_count}/{segment_count} with gist, "
             f"{total_claims} total claims",
             segment_count=segment_count,
             gist_coverage=f"{gist_count}/{segment_count}",
             total_claims=total_claims)

    # Validate segment coverage
    _validate_segment_coverage(output, target_segments)

    # Flatten segments → claims, propagating segment_gist
    all_claims: list[ExtractedClaim] = []
    segments_with_claims = 0
    segments_empty = 0
    for seg in output.segments:
        if seg.claims:
            segments_with_claims += 1
        else:
            segments_empty += 1
        for claim in seg.claims:
            claim._segment_gist = seg.segment_gist  # transient — used in serialization
        all_claims.extend(seg.claims)

    # Enforce consistency
    _enforce_consistency(all_claims)

    total = len(all_claims)
    worth = sum(1 for c in all_claims if c.worth_checking)

    skip_counts: dict[str, int] = {}
    for c in all_claims:
        if not c.worth_checking and c.skip_reason:
            skip_counts[c.skip_reason] = skip_counts.get(c.skip_reason, 0) + 1

    log.info(logger, MODULE, "extracted",
             f"{total} assertions from {len(output.segments)} segments "
             f"({segments_with_claims} with claims, {segments_empty} empty), "
             f"{worth} worth checking",
             skip_reasons=skip_counts if skip_counts else None)

    return all_claims


# ---------------------------------------------------------------------------
# Finalize — filter + dedup across all batches
# ---------------------------------------------------------------------------

def finalize_claims(
    all_claims: list[ExtractedClaim],
    transcript: Transcript,
) -> list[TranscriptClaim]:
    """Filter to worth_checking, deduplicate, and convert to TranscriptClaim.

    Called after all batches are collected to produce the final claim list.
    """
    total_raw = len(all_claims)
    worth_claims = [c for c in all_claims if c.worth_checking]
    skipped_claims = [c for c in all_claims if not c.worth_checking]

    # Log aggregate skip stats
    agg_skips: dict[str, int] = {}
    for c in skipped_claims:
        reason = c.skip_reason or "unknown"
        agg_skips[reason] = agg_skips.get(reason, 0) + 1
    if agg_skips:
        log.info(logger, MODULE, "filter",
                 f"Filtered {total_raw} → {len(worth_claims)} worth checking",
                 skip_reasons=agg_skips)

    # Deduplicate across batches
    before = len(worth_claims)
    unique_claims = _deduplicate_claims(worth_claims)
    if before != len(unique_claims):
        log.info(logger, MODULE, "dedup",
                 f"Deduplicated {before} → {len(unique_claims)} claims")

    # Build timestamp lookup for seconds
    ts_lookup: dict[str, float] = {}
    for seg in transcript.segments:
        ts_lookup[seg.timestamp] = seg.timestamp_secs

    # Convert to TranscriptClaim
    results = []
    for claim in unique_claims:
        results.append(TranscriptClaim(
            claim_text=claim.claim_text,
            original_quote=claim.original_quote,
            speaker=claim.speaker,
            timestamp=claim.timestamp,
            timestamp_secs=ts_lookup.get(claim.timestamp, 0.0),
            claim_type=claim.claim_type,
            source_url=transcript.url,
        ))

    log.info(logger, MODULE, "complete",
             f"Finalized {len(results)} claims from transcript "
             f"({transcript.word_count} words)")

    return results


# ---------------------------------------------------------------------------
# Convenience — single-call extraction (for small transcripts or direct use)
# ---------------------------------------------------------------------------

async def extract_claims(transcript: Transcript) -> list[TranscriptClaim]:
    """Extract verifiable claims from a transcript in a single call.

    For small transcripts (≤SEGMENT_BATCH_SIZE segments).  Larger transcripts
    should use the Temporal workflow which calls extract_batch per batch.
    """
    all_claims = await extract_batch(
        text_segments=transcript.segments,
        target_segments=transcript.segments,
    )
    return finalize_claims(all_claims, transcript)
