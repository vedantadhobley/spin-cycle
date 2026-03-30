"""Claim extraction from parsed transcripts (Phase 1: chunked extraction).

Takes a TranscriptData (from parser registry) and extracts every verifiable
factual claim with supporting segment references via overlapping chunks.

The old batch-based atomic claim extractor is archived at
src/transcript/_archive_extractor.py.

Post-processing:
- Reference verification: check segment_index values are in range
- Deduplication: handled by Phase 2 (claim review), NOT here
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.llm import invoke_llm, validate_thesis_extraction
from src.prompts.extraction import (
    THESIS_EXTRACTION_SYSTEM, THESIS_EXTRACTION_USER,
    CHUNK_BOUNDARY_INSTRUCTION,
)
from src.schemas.llm_outputs import (
    ExtractedThesis, ThesisExtractionOutput, SupportingReference,
)
from src.transcript.parsers import TranscriptData, NumberedSegment
from src.utils.logging import log, get_logger

MODULE = "thesis_extractor"
logger = get_logger()

from src.transcript.speakers import _enrich_speakers  # noqa: F401


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

TARGET_WORDS_PER_CHUNK = 2500
MAX_SEGMENTS_PER_CHUNK = 25
OVERLAP_SEGMENTS = 3


@dataclass
class ChunkSpec:
    """Defines a chunk of transcript segments for extraction."""
    target_start: int   # first segment index to extract from
    target_end: int     # exclusive — extract claims from [target_start, target_end)
    context_start: int  # first segment in the LLM input (includes leading overlap)
    context_end: int    # exclusive end of LLM input (includes trailing overlap)


def build_chunks(segments: list[NumberedSegment]) -> list[ChunkSpec]:
    """Split transcript into overlapping chunks for extraction.

    Targets ~2500 words per chunk with a hard cap of 25 segments.
    3-segment overlap between adjacent chunks for context continuity.
    Small transcripts (under 2500 words total) → single chunk, no overlap.
    """
    total_words = sum(len(seg.text.split()) for seg in segments)
    n = len(segments)

    # Small transcript: single chunk
    if total_words <= TARGET_WORDS_PER_CHUNK or n <= MAX_SEGMENTS_PER_CHUNK:
        return [ChunkSpec(
            target_start=0, target_end=n,
            context_start=0, context_end=n,
        )]

    chunks: list[ChunkSpec] = []
    pos = 0

    while pos < n:
        # Find how many segments fit in this chunk
        chunk_words = 0
        chunk_end = pos
        while chunk_end < n and chunk_end - pos < MAX_SEGMENTS_PER_CHUNK:
            seg_words = len(segments[chunk_end].text.split())
            if chunk_words + seg_words > TARGET_WORDS_PER_CHUNK and chunk_end > pos:
                break
            chunk_words += seg_words
            chunk_end += 1

        # If we're near the end and the remaining is small, absorb it
        remaining = n - chunk_end
        if 0 < remaining <= OVERLAP_SEGMENTS:
            chunk_end = n

        target_start = pos
        target_end = chunk_end

        # Add overlap context
        context_start = max(0, target_start - OVERLAP_SEGMENTS)
        context_end = min(n, target_end + OVERLAP_SEGMENTS)

        chunks.append(ChunkSpec(
            target_start=target_start,
            target_end=target_end,
            context_start=context_start,
            context_end=context_end,
        ))

        if chunk_end >= n:
            break
        pos = chunk_end

    log.info(logger, MODULE, "chunks_built",
             f"Built {len(chunks)} chunks from {n} segments ({total_words} words)",
             chunk_count=len(chunks),
             chunk_sizes=[c.target_end - c.target_start for c in chunks])

    return chunks


# ---------------------------------------------------------------------------
# Reference verification (kept — programmatic bounds check)
# ---------------------------------------------------------------------------

def _verify_references(
    theses: list[ExtractedThesis],
    segments: list[NumberedSegment],
) -> list[ExtractedThesis]:
    """Check that segment_index values are in range and deduplicate."""
    max_idx = len(segments) - 1
    for thesis in theses:
        valid_refs = []
        seen_indices: set[int] = set()
        for ref in thesis.supporting_references:
            if ref.segment_index in seen_indices:
                continue
            if 0 <= ref.segment_index <= max_idx:
                valid_refs.append(ref)
                seen_indices.add(ref.segment_index)
            else:
                log.warning(logger, MODULE, "ref_out_of_range",
                            f"Segment index {ref.segment_index} out of range "
                            f"(max={max_idx}), dropping reference",
                            thesis=thesis.thesis_statement[:60])
        thesis.supporting_references = valid_refs
    return theses


# ---------------------------------------------------------------------------
# Chunk-level extraction (Phase 1)
# ---------------------------------------------------------------------------

def _build_speaker_desc(enriched_speakers: list[dict]) -> str:
    """Build speaker descriptions string from enriched speakers."""
    speaker_lines = []
    for s in enriched_speakers:
        if s.get("description"):
            speaker_lines.append(f"- {s['name']}: {s['description']}")
        else:
            speaker_lines.append(f"- {s['name']}")
    return "\n".join(speaker_lines) if speaker_lines else "(no speaker info)"


async def extract_chunk(
    transcript: TranscriptData,
    chunk: ChunkSpec,
    enriched_speakers: list[dict],
) -> list[ExtractedThesis]:
    """Extract claims from a single chunk of a transcript.

    Args:
        transcript: Full parsed transcript (for metadata).
        chunk: ChunkSpec defining target and context ranges.
        enriched_speakers: Pre-resolved speaker descriptions.

    Returns:
        List of ExtractedThesis from this chunk (post-processed).
    """
    # Slice segments for LLM input
    context_segments = transcript.segments[chunk.context_start:chunk.context_end]

    # Build numbered text for this chunk
    parts = []
    for seg in context_segments:
        header = ""
        if seg.section_header:
            header = f"[Section: {seg.section_header}]\n"
        parts.append(f"{header}[{seg.index}] {seg.speaker}: {seg.text}")
    numbered_text = "\n\n".join(parts)

    # Build chunk boundary instruction if this is not the full transcript
    is_chunk = (chunk.context_start != 0 or chunk.context_end != len(transcript.segments))
    if is_chunk:
        chunk_boundary = CHUNK_BOUNDARY_INSTRUCTION.format(
            target_start=chunk.target_start,
            target_end=chunk.target_end - 1,  # inclusive end for human readability
        )
    else:
        chunk_boundary = ""

    context_note = (
        f"Title: {transcript.title}. "
        f"Date: {transcript.date or 'unknown'}. "
        f"{transcript.word_count} words across {transcript.segment_count} segments. "
        f"Speakers: {', '.join(transcript.speakers)}."
    )

    chunk_label = (
        f"segments [{chunk.target_start}]-[{chunk.target_end - 1}]"
        if is_chunk else "full transcript"
    )
    log.info(logger, MODULE, "chunk_extracting",
             f"Extracting from {chunk_label}",
             target_start=chunk.target_start,
             target_end=chunk.target_end,
             context_start=chunk.context_start,
             context_end=chunk.context_end)

    output = await invoke_llm(
        system_prompt=THESIS_EXTRACTION_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=THESIS_EXTRACTION_USER.format(
            chunk_boundary=chunk_boundary,
            numbered_transcript=numbered_text,
            context_note=context_note,
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
        ),
        schema=ThesisExtractionOutput,
        semantic_validator=validate_thesis_extraction,
        temperature=0,
        max_tokens=16384,
        activity_name=f"extract_chunk_{chunk.target_start}_{chunk.target_end}",
    )

    theses = output.theses

    # Post-processing: verify references against full transcript segments
    theses = _verify_references(theses, transcript.segments)

    # Drop theses with no valid references
    before = len(theses)
    theses = [t for t in theses if len(t.supporting_references) >= 1]
    if len(theses) < before:
        log.warning(logger, MODULE, "no_refs_dropped",
                    f"Dropped {before - len(theses)} theses with no valid references")

    # If processing a chunk, filter out claims whose references are all outside target range
    if is_chunk:
        target_indices = set(range(chunk.target_start, chunk.target_end))
        filtered = []
        for t in theses:
            has_target_ref = any(
                r.segment_index in target_indices
                for r in t.supporting_references
            )
            if has_target_ref:
                filtered.append(t)
            else:
                log.info(logger, MODULE, "context_only_claim",
                         "Dropped claim with refs only in context segments",
                         thesis=t.thesis_statement[:60])
        theses = filtered

    log.info(logger, MODULE, "chunk_extracted",
             f"Extracted {len(theses)} theses from {chunk_label}",
             count=len(theses),
             topics=[t.topic for t in theses])

    return theses
