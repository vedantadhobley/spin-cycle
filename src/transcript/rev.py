"""Rev.com transcript fetcher.

Rev.com transcripts use ``<p>`` elements inside a rich-text blog div.  The HTML
patterns are:

    Speaker header:
        <p>SpeakerName (<a href="...">MM:SS</a>):</p>

    Text paragraph (under most recent speaker):
        <p>Spoken text here.</p>

    Continuation (same speaker, new timestamp + text in one <p>):
        <p>(<a href="...">MM:SS</a>)<br/>Continuation text here.</p>

This module parses those ``<p>`` elements directly from BeautifulSoup, extracting
speaker labels, timestamps, and text content, then produces a TranscriptData
compatible with the rest of the pipeline.
"""

import re

import httpx
from bs4 import BeautifulSoup, Tag

from src.transcript.parsers import TranscriptData, SpeakerTurn
from src.utils.logging import log, get_logger

MODULE = "rev"
logger = get_logger()

# Speaker header: "SpeakerName (TIMESTAMP):" — the timestamp is an <a> tag so
# get_text() yields e.g. "SpeakerName (00:12):"
_SPEAKER_P_RE = re.compile(
    r"^(.+?)\s*"           # speaker name (greedy-minimal)
    r"\(\s*"               # opening paren
    r"(\d{1,2}:\d{2}"      # MM:SS
    r"(?::\d{2})?)"        # optional :SS for H:MM:SS
    r"\s*\)"               # closing paren
    r"\s*:\s*$",           # trailing colon
)

# Continuation paragraph: "(TIMESTAMP) text..." — <a> tag renders as the
# timestamp string, <br/> becomes nothing in get_text.
_CONTINUATION_P_RE = re.compile(
    r"^\(\s*"              # opening paren
    r"(\d{1,2}:\d{2}"      # MM:SS
    r"(?::\d{2})?)"        # optional :SS
    r"\s*\)"               # closing paren
    r"\s*(.*)$",           # optional remaining text
    re.DOTALL,
)


def _extract_p_text(p_tag: Tag) -> str:
    """Extract text from a <p> tag, collapsing <br/> to spaces."""
    parts = []
    for child in p_tag.children:
        if isinstance(child, Tag):
            if child.name == "br":
                parts.append(" ")
            else:
                parts.append(child.get_text())
        else:
            parts.append(str(child))
    return " ".join("".join(parts).split())  # normalize whitespace


def _find_transcript_div(soup: BeautifulSoup) -> Tag | None:
    """Find the largest blog-text-rich-text div (skips small copyright boxes)."""
    candidates = soup.find_all("div", class_="blog-text-rich-text")
    if candidates:
        return max(candidates, key=lambda d: len(d.get_text()))
    # Fallback selectors
    return (
        soup.find("article")
        or soup.find("div", class_="w-richtext")
    )


def _parse_transcript_html(content_div: Tag) -> list[SpeakerTurn]:
    """Parse transcript <p> elements into SpeakerTurn list."""
    turns: list[SpeakerTurn] = []
    current_speaker = ""
    current_lines: list[str] = []

    def _flush():
        if current_speaker and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                turns.append(SpeakerTurn(
                    speaker=current_speaker,
                    text=text,
                ))

    for p in content_div.find_all("p"):
        p_text = _extract_p_text(p)
        if not p_text:
            continue

        # Check for speaker header: "SpeakerName (MM:SS):"
        speaker_match = _SPEAKER_P_RE.match(p_text)
        if speaker_match:
            _flush()
            current_speaker = speaker_match.group(1).strip()
            current_lines = []
            continue

        # Check for continuation: "(MM:SS) optional text..."
        cont_match = _CONTINUATION_P_RE.match(p_text)
        if cont_match:
            remainder = cont_match.group(2).strip()
            if remainder and current_speaker:
                current_lines.append(remainder)
            continue

        # Regular text paragraph — add to current segment
        if current_speaker:
            current_lines.append(p_text)

    _flush()
    return turns


def is_rev_url(url: str) -> bool:
    """Check if a URL is from rev.com."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        return "rev.com" in host
    except Exception:
        return False


async def fetch_rev_transcript(url: str) -> TranscriptData:
    """Fetch and parse a Rev.com transcript page.

    Returns a TranscriptData with SpeakerTurns, compatible with the
    thesis extraction pipeline.
    """
    log.info(logger, MODULE, "fetch_start", "Fetching Rev.com transcript", url=url)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "SpinCycle/1.0 (fact-checking research)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract description — og:description or meta description
    description = None
    og_desc = soup.find("meta", {"property": "og:description"})
    if og_desc and og_desc.get("content"):
        description = og_desc["content"].strip()
    if not description:
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

    # Extract date — try multiple sources
    date = None
    # 1. Rev.com time-ago element
    time_tag = soup.find("span", class_="time-ago")
    if time_tag and time_tag.get("data-original-date"):
        date = time_tag["data-original-date"]
    # 2. OpenGraph / article meta tag
    if not date:
        meta_date = soup.find("meta", {"property": "article:published_time"})
        if meta_date:
            date = meta_date.get("content", "")
    # 3. JSON-LD structured data (schema.org datePublished)
    if not date:
        import json as _json

        for script_tag in soup.find_all("script", type="application/ld+json"):
            try:
                ld = _json.loads(script_tag.string or "")
                dp = ld.get("datePublished") if isinstance(ld, dict) else None
                if dp:
                    date = dp
                    break
            except (ValueError, TypeError):
                pass
    # Normalize to date-only
    if date and "T" in date:
        date = date.split("T")[0]

    # Extract transcript content
    content_div = _find_transcript_div(soup)
    if not content_div:
        raise ValueError(f"Could not find transcript content in {url}")

    raw_turns = _parse_transcript_html(content_div)
    if not raw_turns:
        raise ValueError(f"No transcript turns parsed from {url}")

    # Raw turns returned as-is — normalization (merging consecutive
    # same-speaker turns) happens in the attribute_speakers activity
    turns = raw_turns

    # Build speaker list from turns
    speakers = list(dict.fromkeys(t.speaker for t in turns))

    log.info(logger, MODULE, "fetch_done", "Rev.com transcript fetched",
             url=url, title=title,
             turn_count=len(turns), speaker_count=len(speakers))

    return TranscriptData(
        url=url,
        title=title,
        date=date,
        description=description,
        speakers=speakers,
        turns=turns,
        source_format="rev",
    )
