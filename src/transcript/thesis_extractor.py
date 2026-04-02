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
# Chunking (sentence-based)
# ---------------------------------------------------------------------------

from src.config import TARGET_WORDS_PER_CHUNK, OVERLAP_WORDS


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_words(sentence: NumberedSentence) -> int:
    return _word_count(sentence.text)


def build_sentence_chunks(
    sentences: list[NumberedSentence],
    logger=None,
) -> list[SentenceChunk]:
    """Build overlapping chunks from numbered sentences.

    Accumulates sentences by word budget. Overlap is whole sentences
    (not partial words). Small transcripts → single chunk.
    """
    logger = logger or _default_logger
    total_words = sum(_sentence_words(s) for s in sentences)

    if not sentences:
        return []

    # Small transcript: single chunk, no context markers
    if total_words <= TARGET_WORDS_PER_CHUNK:
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

    # Build chunks with word-budget accumulation
    chunks: list[SentenceChunk] = []
    pos = 0
    n = len(sentences)

    while pos < n:
        # Accumulate target sentences up to word budget
        target_end = pos
        words_acc = 0
        while target_end < n:
            next_words = _sentence_words(sentences[target_end])
            if words_acc > 0 and words_acc + next_words > TARGET_WORDS_PER_CHUNK:
                break
            words_acc += next_words
            target_end += 1

        # Absorb small remainder
        remaining_words = sum(
            _sentence_words(sentences[i]) for i in range(target_end, n)
        )
        if 0 < remaining_words <= OVERLAP_WORDS:
            target_end = n

        target = sentences[pos:target_end]

        # Collect overlap context (whole sentences)
        ctx_before: list[NumberedSentence] = []
        ctx_words = 0
        scan = pos - 1
        while scan >= 0 and ctx_words < OVERLAP_WORDS:
            ctx_words += _sentence_words(sentences[scan])
            ctx_before.insert(0, sentences[scan])
            scan -= 1

        ctx_after: list[NumberedSentence] = []
        ctx_words = 0
        scan = target_end
        while scan < n and ctx_words < OVERLAP_WORDS:
            ctx_words += _sentence_words(sentences[scan])
            ctx_after.append(sentences[scan])
            scan += 1

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

    log.info(logger, MODULE, "chunks_built",
             "Sentence chunks built",
             chunk_count=len(chunks),
             sentence_count=len(sentences),
             total_words=total_words,
             chunk_words=[sum(_sentence_words(s) for s in c.target_sentences) for c in chunks])

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
    """Convert sentence-level extraction output to ExtractedThesis format.

    Derives original_quote by joining the text of referenced sentence indices.
    """
    theses = []
    for claim in output.claims:
        # Build original_quote from sentence texts
        quote_parts = []
        for idx in sorted(claim.sentence_indices):
            sent = sentences_lookup.get(idx)
            if sent:
                quote_parts.append(sent.text)
        original_quote = " ".join(quote_parts)

        theses.append(ExtractedThesis(
            thesis_statement=claim.thesis_statement,
            speakers=claim.speakers,
            original_quote=original_quote,
            topic=claim.topic,
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

    output = await invoke_llm(
        system_prompt=SENTENCE_EXTRACTION_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=SENTENCE_EXTRACTION_USER.format(
            numbered_sentences=chunk.full_text,
            target_range_start=chunk.target_range[0],
            target_range_end=chunk.target_range[1] - 1,
            context_note=context_note,
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
        ),
        schema=SentenceExtractionOutput,
        semantic_validator=coverage_validator,
        max_tokens=16384,
        presence_penalty=0,
        activity_name=f"extract_chunk_{chunk.chunk_index}",
    )

    theses = _convert_to_theses(output, sentences_lookup)

    log.info(logger, MODULE, "chunk_extracted",
             "Claims extracted from sentence chunk",
             count=len(theses),
             chunk_index=chunk.chunk_index,
             not_claims=len(output.not_claims),
             topics=[t.topic for t in theses])

    return theses
