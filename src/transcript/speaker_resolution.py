"""Shared deterministic speaker name resolution.

Runs BEFORE LLM attribution.  Parsers return raw speaker labels and
source-specific metadata; this module applies programmatic resolution to
standardize names before they reach the LLM or downstream pipeline.

Each source can register a resolver via ``@_register("format_name")``.
Sources without a registered resolver get base cleanup (conservative
honorific stripping).  The map-builder utilities (``build_last_name_map``,
``build_token_speaker_map``) are public — any resolver can compose them.
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# Merged honorifics regex (abbreviated + full forms across all sources)
# ---------------------------------------------------------------------------

HONORIFICS = re.compile(
    r"^(?:Pres\.|President|Vice\s+Pres\.|Vice\s+President|"
    r"Sec\.|Secretary|Sen\.|Senator|Rep\.|Representative|"
    r"Congressman|Congresswoman|Governor|Gov\.|"
    r"Mayor|Ambassador|Gen\.|General|Adm\.|Admiral|"
    r"Colonel|Commander|"
    r"Director|Dir\.|Chairman|Chairwoman|Chair|"
    r"Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?|"
    r"Justice|Judge|Chief|"
    r"Prime\s+Minister|Chancellor|Minister|"
    r"Speaker|Leader)\s+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def strip_honorific(name: str) -> str:
    """Strip leading honorific and title-case ALL CAPS names."""
    stripped = HONORIFICS.sub("", name).strip()
    if stripped == stripped.upper() and len(stripped) > 2:
        stripped = stripped.title()
    return stripped


def name_tokens(name: str) -> set[str]:
    """Extract lowercase name tokens after stripping honorifics."""
    stripped = strip_honorific(name)
    return {t.lower().strip(".,'") for t in stripped.split() if len(t) > 1}


def normalize_speaker_name(name: str) -> str:
    """Strip honorifics and title-case ALL CAPS names."""
    return strip_honorific(name.strip()).strip()


# ---------------------------------------------------------------------------
# Map builders — public utilities for use by any resolver
# ---------------------------------------------------------------------------

def build_last_name_map(
    raw_labels: list[str],
    known_names: list[str],
) -> dict[str, str]:
    """Map raw speaker labels to known names by last-name matching.

    Useful for any source that provides a list of proper speaker names
    alongside raw/abbreviated labels (e.g. C-SPAN person links).

    Skips generic roles (Host, Guest, etc.) that can't be resolved this way.
    """
    _SKIP = {
        ">>", "", "HOST", "GUEST", "CALLER", "REPORTER",
        "UNKNOWN", "MODERATOR", "NARRATOR",
    }

    # Build last-name -> proper-name lookup
    last_name_lookup: dict[str, str] = {}
    for name in known_names:
        parts = name.split()
        if parts:
            last_name_lookup[parts[-1].lower()] = name

    mapping: dict[str, str] = {}
    for label in raw_labels:
        if label.upper().strip() in _SKIP:
            continue
        bare = strip_honorific(label)
        bare_parts = bare.split()
        if bare_parts:
            last = bare_parts[-1].lower()
            if last in last_name_lookup:
                mapping[label] = last_name_lookup[last]

    return mapping


def build_token_speaker_map(raw_names: list[str]) -> dict[str, str]:
    """Build raw -> canonical mapping via token overlap.

    Merges variants: "DONALD TRUMP", "PRESIDENT TRUMP",
    "President Donald Trump" all map to the longest variant.
    """
    canonical_groups: list[tuple[str, set[str]]] = []

    for raw in raw_names:
        normalized = normalize_speaker_name(raw)
        tokens = name_tokens(raw)
        if not tokens:
            continue

        matched = False
        for i, (canon, canon_tokens) in enumerate(canonical_groups):
            if tokens & canon_tokens:
                merged_tokens = canon_tokens | tokens
                if len(tokens) > len(canon_tokens):
                    canonical_groups[i] = (normalized, merged_tokens)
                else:
                    canonical_groups[i] = (canon, merged_tokens)
                matched = True
                break

        if not matched:
            canonical_groups.append((normalized, tokens))

    result: dict[str, str] = {}
    for raw in raw_names:
        tokens = name_tokens(raw)
        if not tokens:
            result[raw] = normalize_speaker_name(raw)
            continue
        for canon, canon_tokens in canonical_groups:
            if tokens & canon_tokens:
                result[raw] = canon
                break
        else:
            result[raw] = normalize_speaker_name(raw)

    return result


def build_alias_map(speaker_map: dict[str, str]) -> dict[str, list[str]]:
    """Build canonical -> variant list (excluding the canonical itself)."""
    aliases: dict[str, set[str]] = {}
    for raw, canon in speaker_map.items():
        normalized_raw = normalize_speaker_name(raw)
        if canon not in aliases:
            aliases[canon] = set()
        if normalized_raw != canon:
            aliases[canon].add(normalized_raw)
        if raw.strip() != canon:
            aliases[canon].add(raw.strip())
    return {k: sorted(v) for k, v in aliases.items() if v}


# ---------------------------------------------------------------------------
# Resolver registry
# ---------------------------------------------------------------------------

ResolverFunc = Callable[[dict, dict], dict]
_RESOLVERS: dict[str, ResolverFunc] = {}


def _register(format_name: str):
    """Register a source-specific resolver."""
    def wrapper(func: ResolverFunc) -> ResolverFunc:
        _RESOLVERS[format_name] = func
        return func
    return wrapper


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_speakers(transcript_data: dict) -> dict:
    """Deterministic speaker name resolution for all transcript formats.

    Runs BEFORE LLM attribution.  Looks up a registered resolver for the
    source format; falls back to base cleanup for unregistered formats.

    Modifies transcript_data in place:
      - turns[].speaker resolved to proper names where possible
      - speaker_aliases built from raw -> canonical mappings
      - speakers list rebuilt
    """
    fmt = transcript_data.get("source_format", "")
    metadata = transcript_data.get("speaker_metadata") or {}

    resolver = _RESOLVERS.get(fmt, _resolve_base)
    return resolver(transcript_data, metadata)


# ---------------------------------------------------------------------------
# Registered resolvers
# ---------------------------------------------------------------------------

@_register("cspan")
def _resolve_cspan(transcript_data: dict, metadata: dict) -> dict:
    """C-SPAN: match raw labels to person names via last-name lookup,
    then strip honorifics on anything unmatched."""
    person_names = metadata.get("person_names", [])
    turns = transcript_data.get("turns", [])

    # Collect unique raw labels from turns
    raw_labels: list[str] = []
    for t in turns:
        s = t.get("speaker", "")
        if s and s != "Unknown" and s not in raw_labels:
            raw_labels.append(s)

    # Build raw label -> proper name map
    name_map = build_last_name_map(raw_labels, person_names)

    # Apply resolution and build aliases
    aliases: dict[str, list[str]] = {}
    for t in turns:
        raw = t.get("speaker", "")
        if raw == "Unknown" or not raw:
            continue

        if raw in name_map:
            resolved = name_map[raw]
        else:
            resolved = strip_honorific(raw)

        if resolved != raw:
            t["speaker"] = resolved
            if resolved not in aliases:
                aliases[resolved] = []
            if raw not in aliases[resolved]:
                aliases[resolved].append(raw)

    # Rebuild speakers: turn speakers + person names from HTML
    turn_speakers = list(dict.fromkeys(t.get("speaker", "") for t in turns))
    for pn in person_names:
        if pn not in turn_speakers:
            turn_speakers.append(pn)

    transcript_data["speakers"] = turn_speakers
    transcript_data["speaker_aliases"] = aliases
    return transcript_data


@_register("raw_text")
def _resolve_raw_text(transcript_data: dict, metadata: dict) -> dict:
    """Raw text: merge speaker variants via token-overlap normalization."""
    raw_speaker_names = metadata.get("raw_speaker_names", [])
    turns = transcript_data.get("turns", [])

    if not raw_speaker_names:
        return transcript_data

    # Build token-overlap speaker map and alias map
    speaker_map = build_token_speaker_map(raw_speaker_names)
    alias_map = build_alias_map(speaker_map)

    # Apply canonical forms to turns
    for t in turns:
        raw = t.get("speaker", "")
        if raw in speaker_map:
            t["speaker"] = speaker_map[raw]

    # Rebuild speakers
    speakers = list(dict.fromkeys(
        speaker_map.get(n, n) for n in raw_speaker_names
    ))

    transcript_data["speakers"] = speakers
    transcript_data["speaker_aliases"] = alias_map
    return transcript_data


# ---------------------------------------------------------------------------
# Base resolver (fallback for sources without a registered resolver)
# ---------------------------------------------------------------------------

def _resolve_base(transcript_data: dict, metadata: dict) -> dict:
    """Conservative honorific stripping for sources without structured metadata.

    Only strips when the result still has >= 2 name tokens, to avoid
    reducing "Chairman Caine" to just "Caine".
    """
    turns = transcript_data.get("turns", [])

    aliases: dict[str, list[str]] = {}
    for t in turns:
        raw = t.get("speaker", "")
        if not raw or raw == "Unknown":
            continue

        resolved = strip_honorific(raw)
        # Conservative: only apply if result keeps at least 2 words
        if len(resolved.split()) < 2:
            continue

        if resolved != raw:
            t["speaker"] = resolved
            if resolved not in aliases:
                aliases[resolved] = []
            if raw not in aliases[resolved]:
                aliases[resolved].append(raw)

    if aliases:
        existing = transcript_data.get("speaker_aliases") or {}
        for canon, variants in aliases.items():
            if canon in existing:
                for v in variants:
                    if v not in existing[canon]:
                        existing[canon].append(v)
            else:
                existing[canon] = variants
        transcript_data["speaker_aliases"] = existing

        transcript_data["speakers"] = list(dict.fromkeys(
            t.get("speaker", "") for t in turns
        ))

    return transcript_data
