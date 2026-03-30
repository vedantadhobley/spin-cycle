"""Claim extraction from parsed transcripts (Phase 1: chunked extraction).

Takes a TranscriptData (from parser registry) and extracts every verifiable
factual claim with original_quote attribution via overlapping word-based chunks.

Chunking operates directly on SpeakerTurn lists — no intermediate segment
indexing. Overlap is word-based (~500 words each side), consistent regardless
of source format.

Post-processing:
- Quote validation: original_quote must appear in chunk target text
- Deduplication: handled by Phase 2 (claim review), NOT here
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from src.llm import invoke_llm, validate_thesis_extraction
from src.prompts.extraction import (
    THESIS_EXTRACTION_SYSTEM, THESIS_EXTRACTION_USER,
)
from src.schemas.llm_outputs import (
    ExtractedThesis, ThesisExtractionOutput,
)
from src.transcript.parsers import TranscriptData, SpeakerTurn
from src.utils.logging import log, get_logger

MODULE = "thesis_extractor"
logger = get_logger()

from src.transcript.speakers import _enrich_speakers  # noqa: F401


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

from src.config import TARGET_WORDS_PER_CHUNK, OVERLAP_WORDS


@dataclass
class Chunk:
    """A word-based chunk of transcript for extraction."""
    target_text: str         # screenplay-formatted extraction target
    context_before: str      # leading overlap context
    context_after: str       # trailing overlap context
    full_text: str           # all three concatenated with markers
    chunk_index: int
    total_chunks: int


def _format_turns_screenplay(turns: list[SpeakerTurn]) -> str:
    """Format turns as screenplay text: 'Speaker: text' separated by blank lines."""
    parts = []
    for turn in turns:
        header = ""
        if turn.section_header:
            header = f"[Section: {turn.section_header}]\n"
        parts.append(f"{header}{turn.speaker}: {turn.text}")
    return "\n\n".join(parts)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_turn_at_boundary(turn: SpeakerTurn, max_words: int) -> list[SpeakerTurn]:
    """Split a long monologue into multiple turns at natural boundaries.

    Split priority:
    1. Paragraph break (\\n\\n)
    2. Sentence boundary ('. ' followed by uppercase)

    All resulting turns keep the same speaker. Only the first keeps section_header.
    """
    if _word_count(turn.text) <= max_words:
        return [turn]

    # Try paragraph splits first
    paragraphs = re.split(r"\n\s*\n", turn.text)
    if len(paragraphs) > 1:
        chunks = _merge_text_pieces(paragraphs, max_words, "\n\n")
        if len(chunks) > 1:
            # Recursively split any chunks still over max_words (sentence fallback)
            result = []
            for i, c in enumerate(chunks):
                if not c.strip():
                    continue
                sub = SpeakerTurn(
                    speaker=turn.speaker,
                    text=c.strip(),
                    section_header=turn.section_header if i == 0 else None,
                )
                result.extend(_split_turn_at_boundary(sub, max_words))
            return result

    # Fall back to sentence splitting
    sentences = re.split(r"(?<=\.\s)(?=[A-Z])", turn.text)
    if len(sentences) > 1:
        chunks = _merge_text_pieces(sentences, max_words, "")
        if len(chunks) > 1:
            return [
                SpeakerTurn(
                    speaker=turn.speaker,
                    text=c.strip(),
                    section_header=turn.section_header if i == 0 else None,
                )
                for i, c in enumerate(chunks) if c.strip()
            ]

    # Can't split further — return as-is
    return [turn]


def _merge_text_pieces(pieces: list[str], max_words: int, joiner: str) -> list[str]:
    """Merge text pieces into chunks targeting max_words each."""
    chunks = []
    current: list[str] = []
    current_words = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        pw = _word_count(piece)

        if current_words + pw > max_words and current:
            chunks.append(joiner.join(current))
            current = [piece]
            current_words = pw
        else:
            current.append(piece)
            current_words += pw

    if current:
        chunks.append(joiner.join(current))

    return chunks


def _collect_turns_for_words(
    turns: list[SpeakerTurn], start: int, target_words: int,
) -> int:
    """Return the end index (exclusive) collecting ~target_words from turns[start:].

    Always includes at least one turn. Stops before adding a turn that would
    push the total significantly over target (>1.5x), unless it's the first turn.
    """
    collected = 0
    end = start
    while end < len(turns):
        next_words = _word_count(turns[end].text)
        # If adding this turn would push us well over target and we already have content, stop
        if collected > 0 and collected + next_words > target_words:
            break
        collected += next_words
        end += 1
    return max(end, start + 1)  # always at least one turn


def _collect_overlap_before(
    turns: list[SpeakerTurn], end: int, overlap_words: int,
) -> list[SpeakerTurn]:
    """Collect turns before `end` totaling ~overlap_words."""
    if end <= 0:
        return []
    collected = 0
    start = end
    while start > 0 and collected < overlap_words:
        start -= 1
        collected += _word_count(turns[start].text)
    return turns[start:end]


def _collect_overlap_after(
    turns: list[SpeakerTurn], start: int, overlap_words: int,
) -> list[SpeakerTurn]:
    """Collect turns after `start` totaling ~overlap_words."""
    if start >= len(turns):
        return []
    collected = 0
    end = start
    while end < len(turns) and collected < overlap_words:
        collected += _word_count(turns[end].text)
        end += 1
    return turns[start:end]


def build_chunks(turns: list[SpeakerTurn]) -> list[Chunk]:
    """Split transcript turns into overlapping word-based chunks.

    Targets ~2500 words per chunk with ~500 words overlap on each side.
    Long monologues are split at paragraph/sentence boundaries first.
    Small transcripts → single chunk, no markers.
    """
    total_words = sum(_word_count(t.text) for t in turns)

    # Explode long monologues so we have fine-grained split points
    split_turns: list[SpeakerTurn] = []
    for turn in turns:
        split_turns.extend(_split_turn_at_boundary(turn, TARGET_WORDS_PER_CHUNK))

    # Small transcript: single chunk, no section markers
    if total_words <= TARGET_WORDS_PER_CHUNK:
        text = _format_turns_screenplay(split_turns)
        return [Chunk(
            target_text=text,
            context_before="",
            context_after="",
            full_text=text,
            chunk_index=0,
            total_chunks=1,
        )]

    # Build chunks with word-based boundaries
    chunks: list[Chunk] = []
    pos = 0
    n = len(split_turns)

    while pos < n:
        # Collect target turns for this chunk
        target_end = _collect_turns_for_words(split_turns, pos, TARGET_WORDS_PER_CHUNK)

        # If remainder is small, absorb it
        remaining_words = sum(
            _word_count(split_turns[i].text) for i in range(target_end, n)
        )
        if 0 < remaining_words <= OVERLAP_WORDS:
            target_end = n

        target_turns = split_turns[pos:target_end]

        # Collect overlap context
        before_turns = _collect_overlap_before(split_turns, pos, OVERLAP_WORDS)
        after_turns = _collect_overlap_after(split_turns, target_end, OVERLAP_WORDS)

        target_text = _format_turns_screenplay(target_turns)
        context_before = _format_turns_screenplay(before_turns)
        context_after = _format_turns_screenplay(after_turns)

        chunks.append(Chunk(
            target_text=target_text,
            context_before=context_before,
            context_after=context_after,
            full_text="",  # built below
            chunk_index=len(chunks),
            total_chunks=0,  # set below
        ))

        if target_end >= n:
            break
        pos = target_end

    # Set total_chunks and build full_text with section markers
    for chunk in chunks:
        chunk.total_chunks = len(chunks)
        if len(chunks) == 1:
            chunk.full_text = chunk.target_text
        else:
            parts = []
            if chunk.context_before:
                parts.append(
                    "## Context (do not extract claims from this section)\n"
                    + chunk.context_before
                )
            parts.append(
                "## Extract claims from this section\n"
                + chunk.target_text
            )
            if chunk.context_after:
                parts.append(
                    "## Context (do not extract claims from this section)\n"
                    + chunk.context_after
                )
            chunk.full_text = "\n\n".join(parts)

    log.info(logger, MODULE, "chunks_built",
             f"Built {len(chunks)} chunks from {len(turns)} turns ({total_words} words)",
             chunk_count=len(chunks),
             chunk_words=[_word_count(c.target_text) for c in chunks])

    return chunks



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
    chunk: Chunk,
    enriched_speakers: list[dict],
) -> list[ExtractedThesis]:
    """Extract claims from a single chunk of a transcript.

    Args:
        transcript: Full parsed transcript (for metadata).
        chunk: Chunk with target text and context.
        enriched_speakers: Pre-resolved speaker descriptions.

    Returns:
        List of ExtractedThesis from this chunk (post-processed).
    """
    is_multi_chunk = chunk.total_chunks > 1

    context_note = (
        f"Title: {transcript.title}. "
        f"Date: {transcript.date or 'unknown'}. "
        f"{transcript.word_count} words across {transcript.turn_count} speaker turns. "
        f"Speakers: {', '.join(transcript.speakers)}."
    )

    chunk_label = (
        f"chunk {chunk.chunk_index + 1}/{chunk.total_chunks}"
        if is_multi_chunk else "full transcript"
    )
    log.info(logger, MODULE, "chunk_extracting",
             f"Extracting from {chunk_label}",
             chunk_index=chunk.chunk_index,
             target_words=_word_count(chunk.target_text))

    output = await invoke_llm(
        system_prompt=THESIS_EXTRACTION_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=THESIS_EXTRACTION_USER.format(
            transcript_text=chunk.full_text,
            context_note=context_note,
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
        ),
        schema=ThesisExtractionOutput,
        semantic_validator=validate_thesis_extraction,
        temperature=0,
        max_tokens=16384,
        activity_name=f"extract_chunk_{chunk.chunk_index}",
    )

    theses = output.theses

    log.info(logger, MODULE, "chunk_extracted",
             f"Extracted {len(theses)} theses from {chunk_label}",
             count=len(theses),
             topics=[t.topic for t in theses])

    return theses
