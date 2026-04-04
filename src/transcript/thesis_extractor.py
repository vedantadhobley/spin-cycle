"""Two-pass claim extraction from parsed transcripts (Phase 1).

Takes a TranscriptData (from parser registry) and extracts every verifiable
factual claim via two focused LLM passes:

  Pass 1 (Grouping): Group sentences by semantic continuity, label claim/not_claim.
  Pass 2 (Context Injection): Resolve pronouns/references in claim groups.

Flow:
  SpeakerTurn[] → sentencize → NumberedSentence[] → build_sentence_chunks
  → SentenceChunk[] → [per chunk] extract_dispositions (Pass 1)
  → GroupingOutput → [per chunk, claim groups only] inject_context (Pass 2)
  → ContextInjectionOutput → _build_theses → ExtractedThesis[]

SpaCy splits each turn into sentences with global indices. Pass 1 must
account for every sentence in the target range. A programmatic validator
enforces full coverage. original_quote is derived from sentence text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import partial

from src.llm import invoke_llm, validate_grouping, validate_context_injection
from src.prompts.extraction import (
    GROUPING_SYSTEM, GROUPING_USER,
    CONTEXT_INJECT_SYSTEM, CONTEXT_INJECT_USER,
)
from src.schemas.llm_outputs import (
    ExtractedThesis, GroupingOutput, ContextInjectionOutput,
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


def _format_sentences_inline(sentences: list[NumberedSentence]) -> str:
    """Format sentences as flowing text with inline [Sn] markers.

    Speaker name is only printed when it changes. Sentences flow as
    continuous prose so the model reads them as a transcript, not a
    spreadsheet.
    """
    parts = []
    prev_speaker = None
    for s in sentences:
        if s.section_header:
            parts.append(f"\n[Section: {s.section_header}]\n")
            prev_speaker = None
        if s.speaker != prev_speaker:
            if parts:
                parts.append("\n\n")
            parts.append(f"{s.speaker}: ")
            prev_speaker = s.speaker
        parts.append(f"[S{s.global_index}] {s.text} ")
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Bridge: GroupingOutput + ContextInjectionOutput → ExtractedThesis[]
# ---------------------------------------------------------------------------

def _build_theses(
    grouping: GroupingOutput,
    injection: ContextInjectionOutput,
    sentences_lookup: dict[int, NumberedSentence],
) -> list[ExtractedThesis]:
    """Build ExtractedThesis list from Pass 1 + Pass 2 outputs.

    - thesis_statement comes from Pass 2 (decontextualized_statement)
    - speakers comes from Pass 1 (group disposition)
    - original_quote is programmatic join of raw sentence texts
    """
    # Index Pass 1 groups and Pass 2 claims by group number
    group_lookup = {g.group: g for g in grouping.groups}
    injection_lookup = {c.group: c for c in injection.claims}

    # Collect sentence indices per group
    group_indices: dict[int, list[int]] = {}
    for s in grouping.sentences:
        group_indices.setdefault(s.group, []).append(s.index)

    theses = []
    for gid, indices in sorted(group_indices.items()):
        group = group_lookup.get(gid)
        injected = injection_lookup.get(gid)
        if not group or group.disposition != "claim" or not injected:
            continue

        # Build original_quote from sentence texts
        quote_parts = []
        for idx in sorted(indices):
            sent = sentences_lookup.get(idx)
            if sent:
                quote_parts.append(sent.text)
        original_quote = " ".join(quote_parts)

        theses.append(ExtractedThesis(
            thesis_statement=injected.decontextualized_statement,
            speakers=group.speakers,
            original_quote=original_quote,
        ))
    return theses


# ---------------------------------------------------------------------------
# Shared helpers
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


def _build_context_note(transcript: TranscriptData) -> str:
    """Build transcript metadata context note."""
    desc_line = f" Description: {transcript.description}." if transcript.description else ""
    return (
        f"Title: {transcript.title}. "
        f"Date: {transcript.date or 'unknown'}."
        f"{desc_line} "
        f"{transcript.word_count} words across {transcript.turn_count} speaker turns. "
        f"Speakers: {', '.join(transcript.speakers)}."
    )


# ---------------------------------------------------------------------------
# Pass 1: Grouping + Disposition
# ---------------------------------------------------------------------------

async def extract_dispositions(
    transcript: TranscriptData,
    chunk: SentenceChunk,
    enriched_speakers: list[dict],
    logger=None,
) -> GroupingOutput:
    """Pass 1: Group sentences and label each group claim/not_claim.

    No thesis writing — the model only decides structure and disposition.
    A coverage validator enforces that every target sentence has a group.

    Returns GroupingOutput (serialized to dict by the activity layer).
    """
    logger = logger or _default_logger

    log.info(logger, MODULE, "pass1_start",
             "Pass 1: grouping + disposition",
             chunk_index=chunk.chunk_index,
             total_chunks=chunk.total_chunks,
             target_range=chunk.target_range,
             target_sentences=len(chunk.target_sentences))

    sentence_speakers = {
        s.global_index: s.speaker for s in chunk.target_sentences
    }
    coverage_validator = partial(
        validate_grouping,
        target_range=chunk.target_range,
        sentence_speakers=sentence_speakers,
    )

    start = chunk.target_range[0]
    output = await invoke_llm(
        system_prompt=GROUPING_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=GROUPING_USER.format(
            numbered_sentences=chunk.full_text,
            target_range_start=start,
            target_range_end=chunk.target_range[1] - 1,
            context_note=_build_context_note(transcript),
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
            next_index=start + 1,
            next_next_index=start + 2,
        ),
        schema=GroupingOutput,
        semantic_validator=coverage_validator,
        activity_name=f"group_chunk_{chunk.chunk_index}",
    )

    claim_count = sum(1 for g in output.groups if g.disposition == "claim")
    not_claim_count = sum(1 for g in output.groups if g.disposition == "not_claim")
    log.info(logger, MODULE, "pass1_done",
             "Pass 1 complete",
             chunk_index=chunk.chunk_index,
             claim_groups=claim_count,
             not_claim_groups=not_claim_count,
             total_sentences=len(output.sentences))

    return output


# ---------------------------------------------------------------------------
# Pass 2: Context Injection
# ---------------------------------------------------------------------------

async def inject_context(
    transcript: TranscriptData,
    chunk: SentenceChunk,
    grouping: GroupingOutput,
    sentences_lookup: dict[int, NumberedSentence],
    enriched_speakers: list[dict],
    logger=None,
) -> list[ExtractedThesis]:
    """Pass 2: Resolve references in claim groups to make them standalone.

    Takes Pass 1 grouping output, looks up raw sentence texts, and asks the
    LLM to decontextualize each claim group.

    Returns list of ExtractedThesis (same downstream shape).
    """
    logger = logger or _default_logger

    # Collect claim groups and their sentence texts
    claim_groups = [g for g in grouping.groups if g.disposition == "claim"]
    if not claim_groups:
        return []

    # Build sentence indices per group
    group_indices: dict[int, list[int]] = {}
    for s in grouping.sentences:
        group_indices.setdefault(s.group, []).append(s.index)

    # Build claim groups text for the prompt
    claim_groups_text_parts = []
    expected_groups = []
    for g in claim_groups:
        indices = sorted(group_indices.get(g.group, []))
        raw_sentences = []
        for idx in indices:
            sent = sentences_lookup.get(idx)
            if sent:
                raw_sentences.append(f"[S{idx}] {sent.speaker}: {sent.text}")
        if not raw_sentences:
            continue
        expected_groups.append(g.group)
        speakers_str = ", ".join(g.speakers) if g.speakers else "Unknown"
        claim_groups_text_parts.append(
            f"### Group {g.group} (speakers: {speakers_str})\n"
            + "\n".join(raw_sentences)
        )

    if not expected_groups:
        return []

    claim_groups_text = "\n\n".join(claim_groups_text_parts)

    log.info(logger, MODULE, "pass2_start",
             "Pass 2: context injection",
             chunk_index=chunk.chunk_index,
             claim_groups=len(expected_groups))

    injection_validator = partial(
        validate_context_injection,
        expected_groups=expected_groups,
    )

    output = await invoke_llm(
        system_prompt=CONTEXT_INJECT_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=CONTEXT_INJECT_USER.format(
            full_text=chunk.full_text,
            context_note=_build_context_note(transcript),
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
            claim_groups_text=claim_groups_text,
        ),
        schema=ContextInjectionOutput,
        semantic_validator=injection_validator,
        activity_name=f"inject_chunk_{chunk.chunk_index}",
    )

    theses = _build_theses(grouping, output, sentences_lookup)

    log.info(logger, MODULE, "pass2_done",
             "Pass 2 complete",
             chunk_index=chunk.chunk_index,
             thesis_count=len(theses))

    return theses
