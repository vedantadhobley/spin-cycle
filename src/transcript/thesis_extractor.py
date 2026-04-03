"""Sentence-level claim extraction from parsed transcripts (Phase 1).

Takes a TranscriptData (from parser registry) and extracts every verifiable
factual claim via sentence-level forced accountability.

Flow:
  SpeakerTurn[] → sentencize → NumberedSentence[] → build_sentence_chunks
  → SentenceChunk[] → [per chunk] extract_chunk (LLM + coverage validator)
  → SentenceExtractionOutput → convert_to_theses → ExtractedThesis[]

SpaCy splits each turn into sentences with global indices. The LLM must
account for every sentence in the target range — either extract a claim
or mark it not_claims. A programmatic validator enforces full coverage.
original_quote is derived from sentence text, not LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import partial

from src.llm import invoke_llm, validate_sentence_extraction
from src.prompts.extraction import (
    SENTENCE_EXTRACTION_SYSTEM, SENTENCE_EXTRACTION_USER,
)
from src.schemas.llm_outputs import (
    ExtractedThesis, SentenceExtractionOutput,
)
from src.transcript.parsers import TranscriptData, SpeakerTurn
from src.utils.logging import log, get_logger

MODULE = "thesis_extractor"
_default_logger = get_logger()

from src.transcript.speakers import _enrich_speakers  # noqa: F401


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NumberedSentence:
    """A single sentence with a global index across the whole transcript."""
    global_index: int
    speaker: str
    text: str
    section_header: str | None = None


@dataclass
class SentenceChunk:
    """A chunk of numbered sentences for extraction."""
    target_sentences: list[NumberedSentence]
    context_before: list[NumberedSentence]
    context_after: list[NumberedSentence]
    full_text: str                    # formatted with markers + sentence numbers
    target_range: tuple[int, int]     # (start_inclusive, end_exclusive)
    chunk_index: int
    total_chunks: int


# ---------------------------------------------------------------------------
# Sentencization
# ---------------------------------------------------------------------------

def _get_nlp():
    """Get SpaCy model from shared NER module."""
    from src.utils.ner import _get_nlp as ner_get_nlp
    return ner_get_nlp()


def sentencize_transcript(turns: list[SpeakerTurn]) -> list[NumberedSentence]:
    """Split speaker turns into globally-numbered sentences via SpaCy.

    Each turn's text is sentencized independently (speaker boundaries are
    natural sentence boundaries). Global index runs continuously across
    all turns.
    """
    nlp = _get_nlp()
    sentences: list[NumberedSentence] = []
    idx = 0

    for turn in turns:
        doc = nlp(turn.text)
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            sentences.append(NumberedSentence(
                global_index=idx,
                speaker=turn.speaker,
                text=text,
                section_header=turn.section_header if not sentences or sentences[-1].section_header != turn.section_header else None,
            ))
            idx += 1

    return sentences


# ---------------------------------------------------------------------------
# Chunking (sentence-count-based)
# ---------------------------------------------------------------------------

from src.config import (
    TARGET_SENTENCES_PER_CHUNK, OVERLAP_SENTENCES, SPEAKER_CUTOFF_WINDOW,
)


def build_sentence_chunks(
    sentences: list[NumberedSentence],
    logger=None,
) -> list[SentenceChunk]:
    """Build overlapping chunks from numbered sentences.

    Accumulates by sentence count (~45 target). Prefers to cut at speaker
    boundaries — if the target cutoff lands mid-speaker-turn, extends up
    to SPEAKER_CUTOFF_WINDOW sentences to hit the next speaker change.
    Absorbs small remainders (< OVERLAP_SENTENCES) into the last chunk.
    """
    logger = logger or _default_logger

    if not sentences:
        return []

    n = len(sentences)

    # Small transcript: single chunk, no context markers
    if n <= TARGET_SENTENCES_PER_CHUNK:
        text = _format_sentences_numbered(sentences)
        return [SentenceChunk(
            target_sentences=sentences,
            context_before=[],
            context_after=[],
            full_text=text,
            target_range=(sentences[0].global_index, sentences[-1].global_index + 1),
            chunk_index=0,
            total_chunks=1,
        )]

    # Build chunks with sentence-count accumulation
    chunks: list[SentenceChunk] = []
    pos = 0

    while pos < n:
        target_end = min(pos + TARGET_SENTENCES_PER_CHUNK, n)

        # Speaker-turn-aware cutoff: if we landed mid-speaker, extend up to
        # SPEAKER_CUTOFF_WINDOW sentences to reach the next speaker boundary
        if target_end < n:
            current_speaker = sentences[target_end - 1].speaker
            for i in range(target_end, min(target_end + SPEAKER_CUTOFF_WINDOW, n)):
                if sentences[i].speaker != current_speaker:
                    target_end = i
                    break

        # Absorb small remainder into this chunk
        remaining = n - target_end
        if 0 < remaining <= OVERLAP_SENTENCES:
            target_end = n

        target = sentences[pos:target_end]

        # Overlap context: whole sentences before and after
        ctx_start = max(0, pos - OVERLAP_SENTENCES)
        ctx_before = sentences[ctx_start:pos]

        ctx_end = min(n, target_end + OVERLAP_SENTENCES)
        ctx_after = sentences[target_end:ctx_end]

        chunks.append(SentenceChunk(
            target_sentences=target,
            context_before=ctx_before,
            context_after=ctx_after,
            full_text="",  # built below
            target_range=(target[0].global_index, target[-1].global_index + 1),
            chunk_index=len(chunks),
            total_chunks=0,  # set below
        ))

        if target_end >= n:
            break
        pos = target_end

    # Set total_chunks and build full_text
    for chunk in chunks:
        chunk.total_chunks = len(chunks)
        if len(chunks) == 1:
            chunk.full_text = _format_sentences_numbered(chunk.target_sentences)
        else:
            parts = []
            if chunk.context_before:
                parts.append(
                    "## Context (do not extract from this section)\n"
                    + _format_sentences_numbered(chunk.context_before)
                )
            parts.append(
                f"## Extract claims from sentences S{chunk.target_range[0]} through S{chunk.target_range[1] - 1}\n"
                + _format_sentences_numbered(chunk.target_sentences)
            )
            if chunk.context_after:
                parts.append(
                    "## Context (do not extract from this section)\n"
                    + _format_sentences_numbered(chunk.context_after)
                )
            chunk.full_text = "\n\n".join(parts)

    total_words = sum(len(s.text.split()) for s in sentences)
    log.info(logger, MODULE, "chunks_built",
             "Sentence chunks built",
             chunk_count=len(chunks),
             sentence_count=n,
             total_words=total_words,
             chunk_sentences=[len(c.target_sentences) for c in chunks])

    return chunks


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_sentences_numbered(sentences: list[NumberedSentence]) -> str:
    """Format sentences as [Sn] Speaker: text, one per line."""
    lines = []
    for s in sentences:
        header = ""
        if s.section_header:
            header = f"[Section: {s.section_header}]\n"
        lines.append(f"{header}[S{s.global_index}] {s.speaker}: {s.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bridge: SentenceExtractionOutput → ExtractedThesis[]
# ---------------------------------------------------------------------------

def _convert_to_theses(
    output: SentenceExtractionOutput,
    sentences_lookup: dict[int, NumberedSentence],
) -> list[ExtractedThesis]:
    """Convert one-per-sentence extraction output to ExtractedThesis format.

    Groups dispositions by claim_group, looks up the corresponding
    ClaimGroupThesis, and derives original_quote from sentence texts.
    """
    # Index groups by claim_group for fast lookup
    group_lookup = {g.claim_group: g for g in output.groups}

    # Collect sentence indices per claim_group
    group_indices: dict[int, list[int]] = {}
    for d in output.dispositions:
        if d.claim_group > 0:
            group_indices.setdefault(d.claim_group, []).append(d.index)

    theses = []
    for gid, indices in sorted(group_indices.items()):
        group = group_lookup.get(gid)
        if not group:
            continue

        # Build original_quote from sentence texts
        quote_parts = []
        for idx in sorted(indices):
            sent = sentences_lookup.get(idx)
            if sent:
                quote_parts.append(sent.text)
        original_quote = " ".join(quote_parts)

        theses.append(ExtractedThesis(
            thesis_statement=group.thesis_statement,
            speakers=group.speakers,
            original_quote=original_quote,
        ))
    return theses


# ---------------------------------------------------------------------------
# Chunk-level extraction
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
    chunk: SentenceChunk,
    enriched_speakers: list[dict],
    sentences_lookup: dict[int, NumberedSentence],
    logger=None,
) -> list[ExtractedThesis]:
    """Extract claims from a single sentence chunk.

    Uses sentence-level forced extraction: the LLM must account for every
    sentence in the target range. A coverage validator enforces this.

    Returns list of ExtractedThesis (bridged from sentence output).
    """
    logger = logger or _default_logger

    desc_line = f" Description: {transcript.description}." if transcript.description else ""
    context_note = (
        f"Title: {transcript.title}. "
        f"Date: {transcript.date or 'unknown'}."
        f"{desc_line} "
        f"{transcript.word_count} words across {transcript.turn_count} speaker turns. "
        f"Speakers: {', '.join(transcript.speakers)}."
    )

    log.info(logger, MODULE, "chunk_extracting",
             "Extracting from sentence chunk",
             chunk_index=chunk.chunk_index,
             total_chunks=chunk.total_chunks,
             target_range=chunk.target_range,
             target_sentences=len(chunk.target_sentences))

    # Coverage validator closure — captures this chunk's target range
    coverage_validator = partial(
        validate_sentence_extraction,
        target_range=chunk.target_range,
    )

    start = chunk.target_range[0]
    output = await invoke_llm(
        system_prompt=SENTENCE_EXTRACTION_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=SENTENCE_EXTRACTION_USER.format(
            numbered_sentences=chunk.full_text,
            target_range_start=start,
            target_range_end=chunk.target_range[1] - 1,
            context_note=context_note,
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
            next_index=start + 1,
            next_next_index=start + 2,
        ),
        schema=SentenceExtractionOutput,
        semantic_validator=coverage_validator,
        max_tokens=16384,
        activity_name=f"extract_chunk_{chunk.chunk_index}",
    )

    theses = _convert_to_theses(output, sentences_lookup)

    not_claim_count = sum(1 for d in output.dispositions if d.disposition == "not_claim")
    log.info(logger, MODULE, "chunk_extracted",
             "Claims extracted from sentence chunk",
             count=len(theses),
             chunk_index=chunk.chunk_index,
             not_claims=not_claim_count)

    return theses
