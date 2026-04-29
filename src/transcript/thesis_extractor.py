"""Embedding-based claim extraction from parsed transcripts (Phase 1).

Takes a TranscriptData (from parser registry) and extracts every verifiable
factual claim via decontextualization + programmatic grouping + synthesis:

  Decontextualize: LLM resolves references so each sentence stands alone
  Embed + group:   Programmatic cosine-similarity grouping (no LLM)
  Synthesize:      LLM combines standalone sentences into one claim per group

Flow:
  SpeakerTurn[] → sentencize → NumberedSentence[] → build_sentence_chunks
  → SentenceChunk[] → [per chunk] decontextualize_chunk
  → standalone sentences → embed + group_sentences (programmatic)
  → SentenceGroup[] → [per batch] synthesize_claims
  → ClaimSynthesisOutput → _build_theses → ExtractedThesis[]

SpaCy splits each turn into sentences with global indices. Grouping is
deterministic — same input always produces the same groups.
original_quote is derived from raw sentence text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import partial

from src.llm import invoke_llm, LLMInvocationError, validate_decontextualize, validate_claim_synthesis
from src.prompts.extraction import (
    DECONTEXT_SYSTEM, DECONTEXT_USER,
    SYNTHESIS_SYSTEM, SYNTHESIS_USER,
    SYNTHESIS_RETRY_SYSTEM, SYNTHESIS_RETRY_USER,
)
from src.schemas.llm_outputs import (
    ExtractedThesis, DecontextualizeOutput, ClaimSynthesisOutput,
)
from src.utils.ner import extract_claim_entities, check_entity_coverage
from src.transcript.parsers import TranscriptData, SpeakerTurn
from src.transcript.sentence_grouper import SentenceGroup
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
# Bridge: SentenceGroup[] + ClaimSynthesisOutput → ExtractedThesis[]
# ---------------------------------------------------------------------------

def _build_theses(
    groups: list[SentenceGroup],
    synthesis: ClaimSynthesisOutput,
    sentences_lookup: dict[int, NumberedSentence],
) -> list[ExtractedThesis]:
    """Build ExtractedThesis list from programmatic groups + synthesis output.

    - thesis_statement: synthesized claim from synthesis LLM call
    - speakers: from SentenceGroup.speaker
    - original_quote: programmatic join of raw sentence texts
    """
    synthesis_lookup = {c.group: c for c in synthesis.claims}

    theses = []
    for group in groups:
        synthesized = synthesis_lookup.get(group.group_id)
        if not synthesized:
            continue

        # Skip empty claims and meta-commentary (LLM saying "no claims")
        claim_text = synthesized.claim.strip()
        if not claim_text:
            continue
        claim_lower = claim_text.lower()
        if "no factual claim" in claim_lower or "no verifiable" in claim_lower:
            continue

        # Build original_quote from raw sentence texts
        quote_parts = []
        for idx in sorted(group.sentence_indices):
            sent = sentences_lookup.get(idx)
            if sent:
                quote_parts.append(sent.text)
        original_quote = " ".join(quote_parts)

        theses.append(ExtractedThesis(
            thesis_statement=synthesized.claim,
            speakers=[group.speaker],
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
# Decontextualization (per chunk)
# ---------------------------------------------------------------------------

async def decontextualize_chunk(
    transcript: TranscriptData,
    chunk: SentenceChunk,
    enriched_speakers: list[dict],
    logger=None,
) -> DecontextualizeOutput:
    """Resolve references in each target sentence so it stands alone.

    Uses the chunk's full_text (including context) for reference resolution,
    but only outputs standalone text for target-range sentences.

    Returns DecontextualizeOutput (serialized to dict by the activity layer).
    """
    logger = logger or _default_logger

    log.info(logger, MODULE, "decontext_start",
             "Decontextualizing chunk",
             chunk_index=chunk.chunk_index,
             total_chunks=chunk.total_chunks,
             target_range=chunk.target_range,
             target_sentences=len(chunk.target_sentences))

    coverage_validator = partial(
        validate_decontextualize,
        target_range=chunk.target_range,
    )

    start = chunk.target_range[0]
    output = await invoke_llm(
        system_prompt=DECONTEXT_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=DECONTEXT_USER.format(
            numbered_sentences=chunk.full_text,
            target_range_start=start,
            target_range_end=chunk.target_range[1] - 1,
            context_note=_build_context_note(transcript),
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
            next_index=start + 1,
        ),
        schema=DecontextualizeOutput,
        semantic_validator=coverage_validator,
        activity_name=f"decontext_chunk_{chunk.chunk_index}",
    )

    log.info(logger, MODULE, "decontext_done",
             "Decontextualization complete",
             chunk_index=chunk.chunk_index,
             sentence_count=len(output.sentences))

    return output


# ---------------------------------------------------------------------------
# Claim Synthesis (per batch of groups)
# ---------------------------------------------------------------------------

async def synthesize_claims(
    transcript: TranscriptData,
    groups: list[SentenceGroup],
    standalone_sentences: dict[int, dict],
    sentences_lookup: dict[int, NumberedSentence],
    enriched_speakers: list[dict],
    logger=None,
) -> list[ExtractedThesis]:
    """Synthesize standalone claims from programmatic sentence groups.

    The model distills what is being asserted about the world — combining
    already-decontextualized sentences into a single checkable claim per group.
    Speaker identity is in metadata, not the claim text.

    Two-phase approach:
    1. Batch LLM call for all groups (with NER entity hints in prompt)
    2. Per-group NER validation — groups missing entities get a targeted retry

    Returns list of ExtractedThesis (same downstream shape).
    """
    logger = logger or _default_logger

    if not groups:
        return []

    # Extract NER entities per group for coverage checking
    group_entities: dict[int, list[str]] = {}
    for g in groups:
        raw_texts = [
            sentences_lookup[idx].text
            for idx in sorted(g.sentence_indices) if idx in sentences_lookup
        ]
        entities = extract_claim_entities(raw_texts)
        if entities:
            group_entities[g.group_id] = entities

    # Build claim groups text with standalone sentences + entity hints
    claim_groups_text_parts = []
    expected_groups = []
    group_raw_sentences: dict[int, list[str]] = {}

    for g in groups:
        indices = sorted(g.sentence_indices)
        standalone_lines = []
        raw_lines = []
        for idx in indices:
            ss = standalone_sentences.get(idx)
            if ss:
                standalone_lines.append(f"[S{idx}] {ss['standalone']}")
            sent = sentences_lookup.get(idx)
            if sent:
                raw_lines.append(f"[S{idx}] {sent.text}")

        if not standalone_lines:
            continue

        expected_groups.append(g.group_id)
        group_raw_sentences[g.group_id] = raw_lines

        entity_hint = ""
        entities = group_entities.get(g.group_id)
        if entities:
            entity_hint = f"\nKey entities: {', '.join(entities)}"

        claim_groups_text_parts.append(
            f"### Group {g.group_id} (speaker: {g.speaker})\n"
            + "\n".join(standalone_lines)
            + entity_hint
        )

    if not expected_groups:
        return []

    claim_groups_text = "\n\n".join(claim_groups_text_parts)

    log.info(logger, MODULE, "synthesis_start",
             "Claim synthesis",
             group_count=len(expected_groups),
             groups_with_entities=len(group_entities))

    synthesis_validator = partial(
        validate_claim_synthesis,
        expected_groups=expected_groups,
    )

    output = await invoke_llm(
        system_prompt=SYNTHESIS_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=SYNTHESIS_USER.format(
            context_note=_build_context_note(transcript),
            speaker_descriptions=_build_speaker_desc(enriched_speakers),
            claim_groups_text=claim_groups_text,
        ),
        schema=ClaimSynthesisOutput,
        semantic_validator=synthesis_validator,
        activity_name="synthesize_claims",
    )

    # Per-group NER entity coverage check
    failed_groups: dict[int, dict] = {}
    for c in output.claims:
        entities = group_entities.get(c.group)
        if not entities:
            continue
        missing = check_entity_coverage(c.claim, entities)
        if missing:
            failed_groups[c.group] = {
                "missing": missing,
                "all_entities": entities,
                "previous_claim": c.claim,
            }

    if failed_groups:
        log.info(logger, MODULE, "ner_coverage_gaps",
                 "Entity coverage gaps detected, retrying failed groups",
                 failed_count=len(failed_groups),
                 total_groups=len(expected_groups),
                 gaps={gid: info["missing"] for gid, info in failed_groups.items()})

        retry_output = await _retry_failed_groups(
            transcript, groups, sentences_lookup,
            enriched_speakers, group_raw_sentences, failed_groups, logger,
        )

        if retry_output:
            retry_lookup = {c.group: c for c in retry_output.claims}
            merged = []
            for c in output.claims:
                if c.group in retry_lookup:
                    merged.append(retry_lookup[c.group])
                else:
                    merged.append(c)
            output.claims = merged

            # Log remaining gaps after retry
            still_missing = {}
            for c in output.claims:
                entities = group_entities.get(c.group)
                if not entities:
                    continue
                remaining = check_entity_coverage(c.claim, entities)
                if remaining:
                    still_missing[c.group] = remaining
            if still_missing:
                log.warning(logger, MODULE, "ner_gaps_after_retry",
                            "Some entity gaps remain after retry",
                            remaining_gaps=still_missing)

    theses = _build_theses(groups, output, sentences_lookup)

    log.info(logger, MODULE, "synthesis_done",
             "Claim synthesis complete",
             thesis_count=len(theses),
             groups_retried=len(failed_groups))

    return theses


async def _retry_failed_groups(
    transcript: TranscriptData,
    groups: list[SentenceGroup],
    sentences_lookup: dict[int, NumberedSentence],
    enriched_speakers: list[dict],
    group_raw_sentences: dict[int, list[str]],
    failed_groups: dict[int, dict],
    logger,
) -> ClaimSynthesisOutput | None:
    """Targeted retry for groups that failed NER entity coverage.

    Builds a focused prompt showing only the failed groups with:
    - Raw source sentences
    - Previous synthesized claim
    - Which entities were missing

    Returns ClaimSynthesisOutput with revised claims, or None on failure.
    """
    group_lookup = {g.group_id: g for g in groups}

    retry_parts = []
    retry_expected = []

    for gid, info in failed_groups.items():
        group = group_lookup.get(gid)
        if not group:
            continue
        retry_expected.append(gid)
        raw = "\n".join(group_raw_sentences.get(gid, []))

        retry_parts.append(
            f"### Group {gid} (speaker: {group.speaker})\n"
            f"Source sentences:\n{raw}\n"
            f"Your previous claim:\n{info.get('previous_claim', '')}\n"
            f"Key entities: {', '.join(info['all_entities'])}\n"
            f"MISSING entities: {', '.join(info['missing'])}"
        )

    if not retry_expected:
        return None

    retry_validator = partial(
        validate_claim_synthesis,
        expected_groups=retry_expected,
    )

    try:
        return await invoke_llm(
            system_prompt=SYNTHESIS_RETRY_SYSTEM.format(
                current_date=date.today().isoformat(),
            ),
            user_prompt=SYNTHESIS_RETRY_USER.format(
                context_note=_build_context_note(transcript),
                speaker_descriptions=_build_speaker_desc(enriched_speakers),
                retry_groups_text="\n\n".join(retry_parts),
            ),
            schema=ClaimSynthesisOutput,
            semantic_validator=retry_validator,
            max_retries=1,
            activity_name="synthesis_retry",
        )
    except LLMInvocationError as e:
        log.warning(logger, MODULE, "ner_retry_failed",
                    "Targeted retry failed, keeping original output",
                    error=str(e),
                    failed_groups=list(failed_groups.keys()))
        return None
