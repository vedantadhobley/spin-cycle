"""Fetch and parse transcripts from Rev.com.

Rev.com transcripts use ``<p>`` elements inside a rich-text blog div.  The HTML
patterns are:

    Speaker header:
        <p>SpeakerName (<a href="...">MM:SS</a>):</p>

    Text paragraph (under most recent speaker):
        <p>Spoken text here.</p>

    Continuation (same speaker, new timestamp + text in one <p>):
        <p>(<a href="...">MM:SS</a>)<br/>Continuation text here.</p>

This module parses those ``<p>`` elements directly from BeautifulSoup, extracting
speaker labels, timestamps, and text content.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag


@dataclass
class TranscriptSegment:
    """A single speaker segment from a transcript."""
    speaker: str
    timestamp: str  # "MM:SS" format
    timestamp_secs: float  # seconds from start
    text: str  # the spoken text (may span multiple paragraphs)


@dataclass
class Transcript:
    """A parsed transcript with metadata."""
    url: str
    title: str
    date: str | None  # ISO date if available
    speakers: list[str]  # unique speaker names
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        """Concatenate all segments into a single text block."""
        parts = []
        for seg in self.segments:
            parts.append(f"{seg.speaker} ({seg.timestamp}):\n{seg.text}")
        return "\n\n".join(parts)

    @property
    def display_text(self) -> str:
        """Clean text with consecutive same-speaker segments merged.

        Keeps the first timestamp per speaker turn, merges consecutive
        segments from the same speaker into paragraph blocks.
        """
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

        # Flush last block
        blocks.append(
            f"{current_speaker}:\n"
            + "\n\n".join(current_paragraphs)
        )
        return "\n\n".join(blocks)

    @property
    def word_count(self) -> int:
        return sum(len(seg.text.split()) for seg in self.segments)


@dataclass
class TranscriptListing:
    """A transcript entry from the Rev.com index page."""
    url: str
    title: str
    description: str
    date: str | None  # raw date string
    slug: str


# ---------------------------------------------------------------------------
# Transcript page parser — works directly on BeautifulSoup <p> elements
# ---------------------------------------------------------------------------

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


def _timestamp_to_secs(ts: str) -> float:
    """Convert MM:SS or H:MM:SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


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


def parse_transcript_html(content_div: Tag) -> list[TranscriptSegment]:
    """Parse transcript <p> elements into structured segments."""
    segments: list[TranscriptSegment] = []
    current_speaker = ""
    current_timestamp = ""
    current_secs = 0.0
    current_lines: list[str] = []

    def _flush():
        if current_speaker and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                segments.append(TranscriptSegment(
                    speaker=current_speaker,
                    timestamp=current_timestamp,
                    timestamp_secs=current_secs,
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
            current_timestamp = speaker_match.group(2)
            current_secs = _timestamp_to_secs(current_timestamp)
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
    return segments


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


async def fetch_transcript(url: str) -> Transcript:
    """Fetch and parse a Rev.com transcript page.

    Args:
        url: Full URL to the transcript page.

    Returns:
        Parsed Transcript with metadata and segments.
    """
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

    # Extract date — try multiple sources in priority order
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

        def _find_date_published(obj):
            """Recursively search JSON-LD for datePublished."""
            if isinstance(obj, dict):
                if "datePublished" in obj:
                    return obj["datePublished"]
                for v in obj.values():
                    found = _find_date_published(v)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = _find_date_published(item)
                    if found:
                        return found
            return None

        for script_tag in soup.find_all("script", type="application/ld+json"):
            try:
                ld = _json.loads(script_tag.string or "")
                found = _find_date_published(ld)
                if found:
                    date = found
                    break
            except (ValueError, TypeError):
                pass
    # Normalize to date-only (strip time portion if present)
    if date and "T" in date:
        date = date.split("T")[0]

    # Extract transcript content — pick the largest rich-text div
    content_div = _find_transcript_div(soup)
    if not content_div:
        raise ValueError(f"Could not find transcript content in {url}")

    segments = parse_transcript_html(content_div)

    # Extract unique speakers
    speakers = list(dict.fromkeys(seg.speaker for seg in segments))

    return Transcript(
        url=url,
        title=title,
        date=date,
        speakers=speakers,
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Index page parser
# ---------------------------------------------------------------------------

async def fetch_transcript_index(
    page: int = 1,
    base_url: str = "https://www.rev.com/transcripts",
) -> list[TranscriptListing]:
    """Fetch transcript listings from the Rev.com index page.

    Args:
        page: Page number (1-indexed).
        base_url: Base URL for the transcript index.

    Returns:
        List of TranscriptListing entries.
    """
    url = base_url if page <= 1 else f"{base_url}?cf4abd5b_page={page}"

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "SpinCycle/1.0 (fact-checking research)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    listings: list[TranscriptListing] = []

    # Find transcript cards — Rev.com uses a collection list
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.startswith("/transcripts/") or href == "/transcripts":
            continue

        # Extract slug
        slug = href.split("/transcripts/")[-1].strip("/")
        if not slug:
            continue

        # Get title from heading inside the link
        heading = link.find(["h2", "h3", "h4"])
        title = heading.get_text(strip=True) if heading else ""

        # Get description from sibling or parent
        desc = ""
        parent = link.parent
        if parent:
            desc_tag = parent.find("p")
            if desc_tag:
                desc = desc_tag.get_text(strip=True)

        # Get date
        date = None
        if parent:
            time_tag = parent.find("span", class_="time-ago")
            if time_tag and time_tag.get("data-original-date"):
                date = time_tag["data-original-date"]

        full_url = urljoin("https://www.rev.com", href)

        # Deduplicate (same slug may appear in multiple link elements)
        if any(l.slug == slug for l in listings):
            continue

        listings.append(TranscriptListing(
            url=full_url,
            title=title,
            description=desc,
            date=date,
            slug=slug,
        ))

    return listings
