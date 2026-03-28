"""Parser registry and unified transcript data models.

All transcript parsers produce the same TranscriptData structure regardless of
source format (Rev.com HTML, raw text, etc.).  The registry dispatches URL or
content to the appropriate parser.

TranscriptData carries numbered segments with stable integer indices used as
reference keys throughout the thesis extraction pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable


# ---------------------------------------------------------------------------
# Unified data models
# ---------------------------------------------------------------------------

@dataclass
class NumberedSegment:
    """A single speaker segment with a stable integer index."""
    index: int              # 0-based, stable across pipeline
    speaker: str            # normalized canonical name
    text: str
    timestamp: str | None = None
    section_header: str | None = None  # editorial header preceding this segment


@dataclass
class TranscriptData:
    """Unified parsed transcript, source-format agnostic."""
    url: str
    title: str
    date: str | None
    speakers: list[str]                         # normalized, deduplicated
    segments: list[NumberedSegment]
    source_format: str                          # "raw_text", "revcom"
    speaker_aliases: dict[str, list[str]] = field(default_factory=dict)  # canonical → variants
    editors_note: str | None = None

    @property
    def word_count(self) -> int:
        return sum(len(seg.text.split()) for seg in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def numbered_text(self) -> str:
        """Format for LLM consumption: [0] SPEAKER: text"""
        parts = []
        for seg in self.segments:
            header = ""
            if seg.section_header:
                header = f"[Section: {seg.section_header}]\n"
            parts.append(f"{header}[{seg.index}] {seg.speaker}: {seg.text}")
        return "\n\n".join(parts)

    @property
    def display_text(self) -> str:
        """Merged same-speaker consecutive segments for frontend display."""
        if not self.segments:
            return ""
        blocks: list[str] = []
        current_speaker = self.segments[0].speaker
        current_paragraphs: list[str] = [self.segments[0].text]

        for seg in self.segments[1:]:
            if seg.speaker == current_speaker:
                current_paragraphs.append(seg.text)
            else:
                blocks.append(
                    f"{current_speaker}:\n"
                    + "\n\n".join(current_paragraphs)
                )
                current_speaker = seg.speaker
                current_paragraphs = [seg.text]

        blocks.append(
            f"{current_speaker}:\n"
            + "\n\n".join(current_paragraphs)
        )
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

# Type for parser functions: (content, url, title, date) -> TranscriptData
ParserFunc = Callable[..., Awaitable[TranscriptData] | TranscriptData]

_PARSERS: dict[str, ParserFunc] = {}


def register_parser(name: str):
    """Decorator to register a parser function."""
    def wrapper(func: ParserFunc):
        _PARSERS[name] = func
        return func
    return wrapper


def get_parser(name: str) -> ParserFunc:
    """Get a registered parser by name."""
    if name not in _PARSERS:
        raise ValueError(f"Unknown parser: {name}. Available: {list(_PARSERS.keys())}")
    return _PARSERS[name]


def available_parsers() -> list[str]:
    return list(_PARSERS.keys())


def detect_format(url: str) -> str:
    """Auto-detect parser format from URL."""
    if "rev.com" in url:
        return "revcom"
    # Default to revcom for URLs (most transcript sources are HTML)
    return "revcom"
