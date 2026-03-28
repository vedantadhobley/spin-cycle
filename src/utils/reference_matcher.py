"""Verify thesis supporting references against actual transcript segments.

Each ExtractedThesis has supporting_references with a segment_index and an
excerpt (the LLM's attempt at copying the first ~15-20 words).  This module
fuzzy-matches those excerpts against the actual segment text to:
  1. Confirm the reference is real (not hallucinated)
  2. Compute character offsets for frontend highlighting

Uses difflib.SequenceMatcher with a 0.65 threshold — excerpts may have minor
LLM variation (capitalization, missing articles, truncation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.schemas.llm_outputs import ExtractedThesis, SupportingReference
from src.transcript.parsers import NumberedSegment
from src.utils.logging import log, get_logger

MODULE = "reference_matcher"
logger = get_logger()

# Minimum similarity ratio for a match
MATCH_THRESHOLD = 0.65


@dataclass
class VerifiedReference:
    """A supporting reference with verification status."""
    segment_index: int
    excerpt: str
    verified: bool
    match_ratio: float = 0.0
    char_start: int | None = None
    char_end: int | None = None
    matched_text: str = ""


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, collapse whitespace."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def verify_reference(
    ref: SupportingReference,
    segments: list[NumberedSegment],
) -> VerifiedReference:
    """Verify a single supporting reference against transcript segments.

    Looks up the segment by index, then fuzzy-matches the excerpt within
    the segment text.

    Returns VerifiedReference with match status and character offsets.
    """
    # Find the segment
    segment = None
    for seg in segments:
        if seg.index == ref.segment_index:
            segment = seg
            break

    if segment is None:
        return VerifiedReference(
            segment_index=ref.segment_index,
            excerpt=ref.excerpt,
            verified=False,
        )

    seg_text = segment.text
    excerpt = ref.excerpt.strip()

    if not excerpt:
        return VerifiedReference(
            segment_index=ref.segment_index,
            excerpt=ref.excerpt,
            verified=False,
        )

    # Try exact substring match first (case-insensitive)
    norm_seg = _normalize_for_matching(seg_text)
    norm_excerpt = _normalize_for_matching(excerpt)

    idx = norm_seg.find(norm_excerpt)
    if idx >= 0:
        # Map back to original text positions (approximate)
        return VerifiedReference(
            segment_index=ref.segment_index,
            excerpt=ref.excerpt,
            verified=True,
            match_ratio=1.0,
            char_start=idx,
            char_end=idx + len(norm_excerpt),
            matched_text=seg_text[idx:idx + len(norm_excerpt)],
        )

    # Sliding window fuzzy match — find the best matching window in segment
    excerpt_words = norm_excerpt.split()
    seg_words = norm_seg.split()

    if not excerpt_words or not seg_words:
        return VerifiedReference(
            segment_index=ref.segment_index,
            excerpt=ref.excerpt,
            verified=False,
        )

    window_size = len(excerpt_words)
    best_ratio = 0.0
    best_start = 0

    for i in range(max(1, len(seg_words) - window_size + 1)):
        window = " ".join(seg_words[i:i + window_size])
        ratio = SequenceMatcher(None, norm_excerpt, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio >= MATCH_THRESHOLD:
        # Compute approximate character offsets
        matched_window = " ".join(seg_words[best_start:best_start + window_size])
        # Find position of this window in normalized text
        char_start = norm_seg.find(matched_window)
        char_end = char_start + len(matched_window) if char_start >= 0 else None

        return VerifiedReference(
            segment_index=ref.segment_index,
            excerpt=ref.excerpt,
            verified=True,
            match_ratio=best_ratio,
            char_start=char_start if char_start >= 0 else None,
            char_end=char_end,
            matched_text=matched_window,
        )

    return VerifiedReference(
        segment_index=ref.segment_index,
        excerpt=ref.excerpt,
        verified=False,
        match_ratio=best_ratio,
    )


def resolve_all_references(
    theses: list[ExtractedThesis],
    segments: list[NumberedSegment],
) -> tuple[list[ExtractedThesis], dict]:
    """Verify all references across all theses.

    Returns:
        (theses, stats) — theses are unchanged but verification data is logged.
        stats dict has counts of verified/unverified references.
    """
    total = 0
    verified_count = 0
    unverified_count = 0

    for thesis in theses:
        for ref in thesis.supporting_references:
            total += 1
            result = verify_reference(ref, segments)
            if result.verified:
                verified_count += 1
            else:
                unverified_count += 1
                log.warning(logger, MODULE, "unverified_ref",
                            f"Reference not verified: segment [{ref.segment_index}] "
                            f"excerpt '{ref.excerpt[:50]}...' "
                            f"(ratio={result.match_ratio:.2f})",
                            thesis=thesis.thesis_statement[:60])

    stats = {
        "total": total,
        "verified": verified_count,
        "unverified": unverified_count,
    }

    if total > 0:
        pct = verified_count / total * 100
        log.info(logger, MODULE, "verification_done",
                 f"References: {verified_count}/{total} verified ({pct:.0f}%)",
                 **stats)

    return theses, stats
