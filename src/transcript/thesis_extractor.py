"""Thesis-level extraction from parsed transcripts.

Takes a TranscriptData (from parser registry) and uses a single LLM pass over
the full transcript to identify 15-30 major arguments (theses), each with
supporting segment references.

Replaces the old batch-based atomic claim extractor. The old extractor is
archived at src/transcript/_archive_extractor.py.

Post-processing:
- Reference verification: fuzzy-match excerpts against actual segments
- Checkability enforcement: regex catch for future predictions
- Deduplication: >80% word overlap on thesis_statement = duplicate
"""

from __future__ import annotations

import re
from datetime import date

from src.llm import invoke_llm, validate_thesis_extraction
from src.prompts.extraction import THESIS_EXTRACTION_SYSTEM, THESIS_EXTRACTION_USER
from src.schemas.llm_outputs import (
    ExtractedThesis, ThesisExtractionOutput, SupportingReference,
)
from src.transcript.parsers import TranscriptData, NumberedSegment
from src.utils.logging import log, get_logger

MODULE = "thesis_extractor"
logger = get_logger()


# ---------------------------------------------------------------------------
# Future prediction regex (reused from old extractor)
# ---------------------------------------------------------------------------

_FUTURE_PATTERNS = re.compile(
    r"\b(?:"
    r"will (?:go|be|have|increase|decrease|rise|fall|grow|create|bring|save|get)"
    r"|going to (?:be|have|do|create|bring|see)"
    r"|anticipat(?:e|ing|ed) .{0,30}(?:growth|increase|improvement)"
    r"|expect(?:s|ing)? .{0,30}(?:growth|increase|improvement)"
    r")\b",
    re.IGNORECASE,
)

# Anonymous speaker patterns
_ANONYMOUS_SPEAKER = re.compile(
    r"^(?:speaker\s*\w{0,3}|unknown|unidentified|moderator|host|interviewer"
    r"|caller|audience\s*member|voice(?:\s*over)?)$",
    re.IGNORECASE,
)

_JUNK_DESCRIPTION = re.compile(
    r"(?:^(?:male|female)\s+given\s+name$"
    r"|^given\s+name$"
    r"|^(?:family|sur)\s*name"
    r"|scientific\s+article"
    r"|^Wikimedia\s+disambiguation"
    r"|^human\s+name$"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Speaker enrichment (reused from old extractor)
# ---------------------------------------------------------------------------

async def _enrich_speakers(speakers: list[str]) -> list[dict]:
    """Look up Wikidata descriptions for speakers.

    Skips anonymous/generic names and filters junk Wikidata hits.
    Returns list of dicts: [{"name": "...", "description": "..."|null}]
    """
    import asyncio
    from src.tools.wikidata import get_entity_description

    async def _lookup(name: str) -> dict:
        stripped = name.strip()
        if _ANONYMOUS_SPEAKER.match(stripped):
            return {"name": name, "description": None}
        if len(stripped.split()) < 2:
            return {"name": name, "description": None}
        try:
            desc = await get_entity_description(name)
            if desc and _JUNK_DESCRIPTION.search(desc):
                desc = None
            return {"name": name, "description": desc}
        except Exception:
            return {"name": name, "description": None}

    results = await asyncio.gather(*[_lookup(s) for s in speakers])
    return list(results)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _is_future_prediction(thesis: ExtractedThesis) -> bool:
    """Detect future predictions/promises in thesis statement."""
    return bool(_FUTURE_PATTERNS.search(thesis.thesis_statement))


def _enforce_checkability(theses: list[ExtractedThesis]) -> list[ExtractedThesis]:
    """Programmatically override checkability for future predictions."""
    for thesis in theses:
        if thesis.checkable and _is_future_prediction(thesis):
            thesis.checkable = False
            thesis.checkability_rationale = (
                "Future prediction — cannot be verified with current data"
            )
    return theses


def _verify_references(
    theses: list[ExtractedThesis],
    segments: list[NumberedSegment],
) -> list[ExtractedThesis]:
    """Check that segment_index values are in range."""
    max_idx = len(segments) - 1
    for thesis in theses:
        valid_refs = []
        for ref in thesis.supporting_references:
            if 0 <= ref.segment_index <= max_idx:
                valid_refs.append(ref)
            else:
                log.warning(logger, MODULE, "ref_out_of_range",
                            f"Segment index {ref.segment_index} out of range "
                            f"(max={max_idx}), dropping reference",
                            thesis=thesis.thesis_statement[:60])
        thesis.supporting_references = valid_refs
    return theses


def _deduplicate_theses(theses: list[ExtractedThesis]) -> list[ExtractedThesis]:
    """Remove duplicate theses with >80% word overlap on thesis_statement."""
    unique: list[ExtractedThesis] = []

    for thesis in theses:
        words = set(thesis.thesis_statement.lower().split())
        is_dup = False

        for existing in unique:
            existing_words = set(existing.thesis_statement.lower().split())
            if not words or not existing_words:
                continue
            overlap = len(words & existing_words)
            smaller = min(len(words), len(existing_words))
            if smaller > 0 and overlap / smaller > 0.80:
                # Merge references into the existing thesis
                existing_indices = {
                    r.segment_index for r in existing.supporting_references
                }
                for ref in thesis.supporting_references:
                    if ref.segment_index not in existing_indices:
                        existing.supporting_references.append(ref)
                        existing_indices.add(ref.segment_index)
                is_dup = True
                break

        if not is_dup:
            unique.append(thesis)

    if len(unique) < len(theses):
        log.info(logger, MODULE, "dedup",
                 f"Deduplicated {len(theses)} → {len(unique)} theses")

    return unique


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

async def extract_theses(
    transcript: TranscriptData,
    enriched_speakers: list[dict] | None = None,
) -> list[ExtractedThesis]:
    """Extract major theses from a transcript via single LLM call.

    Args:
        transcript: Parsed transcript with numbered segments.
        enriched_speakers: Pre-resolved speaker descriptions from Wikidata.
            If None, will enrich speakers automatically.

    Returns:
        List of ExtractedThesis objects (post-processed, verified, deduped).
    """
    if enriched_speakers is None:
        enriched_speakers = await _enrich_speakers(transcript.speakers)

    # Build speaker descriptions string
    speaker_lines = []
    for s in enriched_speakers:
        if s["description"]:
            speaker_lines.append(f"- {s['name']}: {s['description']}")
        else:
            speaker_lines.append(f"- {s['name']}")
    speaker_desc_str = "\n".join(speaker_lines) if speaker_lines else "(no speaker info)"

    # Build context note
    context_note = (
        f"Title: {transcript.title}. "
        f"Date: {transcript.date or 'unknown'}. "
        f"{transcript.word_count} words across {transcript.segment_count} segments. "
        f"Speakers: {', '.join(transcript.speakers)}."
    )

    log.info(logger, MODULE, "extracting",
             f"Extracting theses from {transcript.segment_count} segments "
             f"({transcript.word_count} words)",
             title=transcript.title)

    output = await invoke_llm(
        system_prompt=THESIS_EXTRACTION_SYSTEM.format(
            current_date=date.today().isoformat(),
        ),
        user_prompt=THESIS_EXTRACTION_USER.format(
            numbered_transcript=transcript.numbered_text,
            context_note=context_note,
            speaker_descriptions=speaker_desc_str,
        ),
        schema=ThesisExtractionOutput,
        semantic_validator=validate_thesis_extraction,
        temperature=0,
        max_tokens=16384,
        activity_name="extract_theses",
    )

    theses = output.theses

    # Post-processing pipeline
    theses = _verify_references(theses, transcript.segments)
    theses = _enforce_checkability(theses)
    theses = _deduplicate_theses(theses)

    # Drop theses with no valid references
    before = len(theses)
    theses = [t for t in theses if len(t.supporting_references) >= 1]
    if len(theses) < before:
        log.warning(logger, MODULE, "no_refs_dropped",
                    f"Dropped {before - len(theses)} theses with no valid references")

    # Stats
    checkable = sum(1 for t in theses if t.checkable)
    log.info(logger, MODULE, "extracted",
             f"{len(theses)} theses extracted ({checkable} checkable)",
             total=len(theses), checkable=checkable,
             topics=[t.topic for t in theses])

    return theses
