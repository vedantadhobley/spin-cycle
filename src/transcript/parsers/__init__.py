"""Parser registry and unified transcript data models.

All transcript parsers produce the same TranscriptData structure regardless of
source format (C-SPAN JSON, raw text, etc.).  The registry dispatches URL or
content to the appropriate parser.

Parsers produce SpeakerTurn lists which are normalized via normalize_turns(),
then chunked by the thesis_extractor for LLM consumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable


# ---------------------------------------------------------------------------
# Unified data models
# ---------------------------------------------------------------------------

@dataclass
class SpeakerTurn:
    """A single speaker turn — one contiguous block of speech by one person."""
    speaker: str            # normalized canonical name
    text: str
    section_header: str | None = None  # editorial header preceding this turn


@dataclass
class TranscriptData:
    """Unified parsed transcript, source-format agnostic."""
    url: str
    title: str
    date: str | None
    speakers: list[str]                         # normalized, deduplicated
    turns: list[SpeakerTurn]
    source_format: str                          # "raw_text", "revcom", "cspan"
    speaker_aliases: dict[str, list[str]] = field(default_factory=dict)  # canonical → variants
    editors_note: str | None = None
    # Optional overrides — used when reconstructing from slim metadata (no turns)
    _word_count_override: int | None = field(default=None, repr=False)
    _turn_count_override: int | None = field(default=None, repr=False)

    @property
    def word_count(self) -> int:
        if self._word_count_override is not None:
            return self._word_count_override
        return sum(len(t.text.split()) for t in self.turns)

    @property
    def turn_count(self) -> int:
        if self._turn_count_override is not None:
            return self._turn_count_override
        return len(self.turns)

    @property
    def display_text(self) -> str:
        """Screenplay-formatted text for frontend display."""
        if not self.turns:
            return ""
        blocks: list[str] = []
        for turn in self.turns:
            blocks.append(f"{turn.speaker}: {turn.text}")
        return "\n\n".join(blocks)


def normalize_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker turns (e.g. C-SPAN caption fragments).

    Does NOT merge across section_header boundaries — a new header starts a
    new turn even if the speaker is the same.
    """
    if not turns:
        return []

    merged: list[SpeakerTurn] = [
        SpeakerTurn(
            speaker=turns[0].speaker,
            text=turns[0].text,
            section_header=turns[0].section_header,
        )
    ]

    for turn in turns[1:]:
        prev = merged[-1]
        # Merge if same speaker AND no new section header
        if turn.speaker == prev.speaker and turn.section_header is None:
            prev.text = prev.text + "\n\n" + turn.text
        else:
            merged.append(SpeakerTurn(
                speaker=turn.speaker,
                text=turn.text,
                section_header=turn.section_header,
            ))

    return merged


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
    if "c-span.org" in url:
        return "cspan"
    if "rev.com" in url:
        return "revcom"
    return "revcom"
