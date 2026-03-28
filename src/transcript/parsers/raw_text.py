"""Raw text transcript parser.

Handles copy-pasted transcripts with speaker labels in various formats:
  - "SPEAKER NAME: text..."
  - "**Speaker Name:** text..."
  - "Speaker Name: text..."

Speaker normalization merges variants like "DONALD TRUMP", "PRESIDENT TRUMP",
"President Donald Trump" into a single canonical form.

Editorial content (section headers, "ALSO READ:" lines, editor's notes) is
detected and separated from spoken content.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.transcript.parsers import (
    NumberedSegment, TranscriptData, register_parser,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum words per segment before splitting at paragraph breaks
MAX_SEGMENT_WORDS = 300

# Honorifics to strip for canonical name matching
_HONORIFICS = re.compile(
    r"^(?:President|Vice\s+President|Senator|Secretary|Representative|"
    r"Congressman|Congresswoman|Governor|Mayor|Ambassador|General|Admiral|"
    r"Colonel|Commander|Director|Chairman|Chairwoman|Chair|"
    r"Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?|Justice|Judge|Chief|"
    r"Prime\s+Minister|Chancellor|Minister|Speaker|Leader)\s+",
    re.IGNORECASE,
)

# Speaker line patterns (order matters — most specific first)
# "SPEAKER NAME:" or "**Speaker Name:**" or "Speaker Name:"
_SPEAKER_PATTERNS = [
    # Markdown bold: **Name:**
    re.compile(r"^\*\*(.+?)\*\*\s*:\s*(.*)$", re.DOTALL),
    # ALL CAPS NAME: text (at least 2 uppercase words before colon)
    re.compile(r"^([A-Z][A-Z\s.\-']{2,}?)\s*:\s*(.*)$", re.DOTALL),
    # Title-case Name: text (First Last: ...) — require at least 2 words
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*:\s*(.*)$", re.DOTALL),
]

# Editorial "ALSO READ:" lines — strip entirely
_ALSO_READ = re.compile(r"^\s*ALSO\s+READ\s*:", re.IGNORECASE)

# Section header: short line, no colon (or colon only at end), title-case
# Must be ≤ 8 words and not look like a speaker line
_SECTION_HEADER = re.compile(
    r"^([A-Z][A-Za-z0-9\s,'\-&:]+)$"
)

# Editor's note pattern (typically first line)
_EDITORS_NOTE = re.compile(
    r"^(?:Editor'?s?\s+Notes?|Notes?)\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
)

# Lines that look like speaker labels but aren't
_NOT_A_SPEAKER = re.compile(
    r"^(?:TRANSCRIPT|TRANSCRIPTION|NOTE|EDITOR|DISCLAIMER|SOURCE|CREDIT|PUBLISHED)\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Speaker normalization
# ---------------------------------------------------------------------------

def _strip_honorific(name: str) -> str:
    """Remove leading honorific from a speaker name."""
    return _HONORIFICS.sub("", name).strip()


def _name_tokens(name: str) -> set[str]:
    """Extract lowercase name tokens, stripping honorifics."""
    stripped = _strip_honorific(name)
    return {t.lower().strip(".,'") for t in stripped.split() if len(t) > 1}


def _normalize_speaker_name(name: str) -> str:
    """Normalize a raw speaker name: strip honorifics, title-case."""
    stripped = _strip_honorific(name.strip())
    # If the name is ALL CAPS, title-case it
    if stripped == stripped.upper() and len(stripped) > 2:
        stripped = stripped.title()
    return stripped.strip()


def _build_speaker_map(raw_names: list[str]) -> dict[str, str]:
    """Build a mapping from raw speaker names to canonical forms.

    Merges variants by token overlap: "DONALD TRUMP", "PRESIDENT TRUMP",
    "President Donald Trump" all map to the longest variant (most tokens).

    Returns:
        Dict mapping each raw name to its canonical form.
    """
    # Group by overlapping tokens
    canonical_groups: list[tuple[str, set[str]]] = []  # (canonical, tokens)

    for raw in raw_names:
        normalized = _normalize_speaker_name(raw)
        tokens = _name_tokens(raw)
        if not tokens:
            continue

        # Find matching group: share at least one significant token
        matched = False
        for i, (canon, canon_tokens) in enumerate(canonical_groups):
            overlap = tokens & canon_tokens
            # Need at least one non-trivial overlap
            if overlap and len(overlap) >= 1:
                # Keep the version with more tokens as canonical
                merged_tokens = canon_tokens | tokens
                if len(tokens) > len(canon_tokens):
                    canonical_groups[i] = (normalized, merged_tokens)
                else:
                    canonical_groups[i] = (canon, merged_tokens)
                matched = True
                break

        if not matched:
            canonical_groups.append((normalized, tokens))

    # Build raw → canonical map
    result: dict[str, str] = {}
    for raw in raw_names:
        tokens = _name_tokens(raw)
        if not tokens:
            result[raw] = _normalize_speaker_name(raw)
            continue
        for canon, canon_tokens in canonical_groups:
            if tokens & canon_tokens:
                result[raw] = canon
                break
        else:
            result[raw] = _normalize_speaker_name(raw)

    return result


def _build_alias_map(speaker_map: dict[str, str]) -> dict[str, list[str]]:
    """Build canonical → list of variant names (excluding the canonical itself)."""
    aliases: dict[str, set[str]] = {}
    for raw, canon in speaker_map.items():
        normalized_raw = _normalize_speaker_name(raw)
        if canon not in aliases:
            aliases[canon] = set()
        if normalized_raw != canon:
            aliases[canon].add(normalized_raw)
        # Also add the raw form if different
        if raw.strip() != canon:
            aliases[canon].add(raw.strip())
    return {k: sorted(v) for k, v in aliases.items() if v}


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

def _try_speaker_line(line: str) -> tuple[str, str] | None:
    """Try to parse a line as a speaker label.

    Returns (speaker_name, remaining_text) or None.
    """
    for pattern in _SPEAKER_PATTERNS:
        m = pattern.match(line)
        if m:
            name = m.group(1).strip().strip("*")
            text = m.group(2).strip() if m.lastindex >= 2 else ""
            # Reject if name is too long (probably not a speaker)
            if len(name.split()) > 6:
                continue
            # Reject if name is just one short word (likely not a speaker)
            if len(name) < 2:
                continue
            # Reject known non-speaker labels
            if _NOT_A_SPEAKER.match(name):
                continue
            return name, text
    return None


def _is_section_header(line: str, next_line: str | None = None) -> bool:
    """Check if a line is an editorial section header.

    Heuristics:
    - Short (≤ 8 words)
    - Title-case or ALL CAPS
    - No colon in the middle (distinguishes from speaker lines)
    - Next line exists and is a speaker line or longer text
    """
    stripped = line.strip()
    if not stripped or len(stripped.split()) > 8:
        return False
    # Must not contain a colon (speaker lines have colons)
    if ":" in stripped:
        return False
    # Must be title-case or ALL CAPS
    if not (stripped == stripped.upper() or stripped.istitle() or
            _SECTION_HEADER.match(stripped)):
        return False
    # Very short single words are ambiguous — skip
    if len(stripped.split()) < 2 and len(stripped) < 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def _split_long_segment(text: str, max_words: int = MAX_SEGMENT_WORDS) -> list[str]:
    """Split a long monologue at paragraph breaks.

    If the text exceeds max_words, split at double-newline paragraph
    boundaries. Each chunk aims to be ≤ max_words.
    """
    if len(text.split()) <= max_words:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) <= 1:
        return [text]  # Can't split further

    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        if current_words + para_words > max_words and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        chunks.append("\n\n".join(current))

    return chunks


@register_parser("raw_text")
def parse_raw_text(
    content: str,
    url: str = "",
    title: str = "",
    date: str | None = None,
) -> TranscriptData:
    """Parse a raw text transcript into TranscriptData.

    Args:
        content: The raw transcript text.
        url: Source URL (may be empty for pasted text).
        title: Transcript title.
        date: ISO date string, if known.

    Returns:
        TranscriptData with numbered segments.
    """
    lines = content.split("\n")

    # Pass 1: Detect editor's note (first non-empty line)
    editors_note = None
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _EDITORS_NOTE.match(stripped)
        if m:
            editors_note = m.group(1).strip()
            content_start = i + 1
        break

    # Skip "TRANSCRIPT:" header if present
    for i in range(content_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.upper() in ("TRANSCRIPT:", "TRANSCRIPT"):
            content_start = i + 1
        break

    # Pass 2: Parse lines into raw segments
    raw_segments: list[dict] = []  # {speaker, text, section_header}
    current_speaker: str | None = None
    current_text_lines: list[str] = []
    current_section: str | None = None
    raw_speaker_names: list[str] = []

    def _flush():
        nonlocal current_speaker, current_text_lines
        if current_speaker and current_text_lines:
            text = "\n\n".join(
                p.strip() for p in "\n".join(current_text_lines).split("\n\n")
                if p.strip()
            )
            if text.strip():
                raw_segments.append({
                    "speaker": current_speaker,
                    "text": text.strip(),
                    "section_header": current_section,
                })
        current_text_lines = []

    for i in range(content_start, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines (but preserve paragraph breaks in text)
        if not stripped:
            if current_text_lines:
                current_text_lines.append("")  # paragraph break
            continue

        # Skip "ALSO READ:" editorial lines
        if _ALSO_READ.match(stripped):
            continue

        # Check for section header
        next_line = None
        for j in range(i + 1, min(i + 3, len(lines))):
            if lines[j].strip():
                next_line = lines[j].strip()
                break

        if _is_section_header(stripped, next_line):
            _flush()
            current_section = stripped
            continue

        # Check for speaker line
        speaker_match = _try_speaker_line(stripped)
        if speaker_match:
            _flush()
            speaker_name, remainder = speaker_match
            current_speaker = speaker_name
            if speaker_name not in raw_speaker_names:
                raw_speaker_names.append(speaker_name)
            if remainder:
                current_text_lines = [remainder]
            else:
                current_text_lines = []
            # Consume the section header
            if current_section:
                raw_segments_section = current_section
                current_section = None
                # Attach to the next segment that gets flushed
                # We'll set it when flushing
                # Actually, store it and use on next flush
                current_section = raw_segments_section
            continue

        # Regular text line — append to current speaker
        if current_speaker:
            current_text_lines.append(stripped)

    _flush()

    # Build speaker normalization map
    speaker_map = _build_speaker_map(raw_speaker_names)
    alias_map = _build_alias_map(speaker_map)

    # Normalize speakers and build numbered segments
    # Split long monologues at paragraph breaks
    segments: list[NumberedSegment] = []
    idx = 0
    for raw_seg in raw_segments:
        canonical = speaker_map.get(raw_seg["speaker"], raw_seg["speaker"])
        chunks = _split_long_segment(raw_seg["text"])
        for ci, chunk in enumerate(chunks):
            segments.append(NumberedSegment(
                index=idx,
                speaker=canonical,
                text=chunk,
                section_header=raw_seg["section_header"] if ci == 0 else None,
            ))
            idx += 1

    # Deduplicated canonical speaker list (preserving order)
    speakers = list(dict.fromkeys(
        speaker_map.get(n, n) for n in raw_speaker_names
    ))

    return TranscriptData(
        url=url,
        title=title,
        date=date,
        speakers=speakers,
        segments=segments,
        source_format="raw_text",
        speaker_aliases=alias_map,
        editors_note=editors_note,
    )
