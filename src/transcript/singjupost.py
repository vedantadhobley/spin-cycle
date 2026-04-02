"""SingjuPost transcript fetcher.

SingjuPost is a WordPress blog that publishes full transcripts of political
speeches and events. The HTML structure uses the Neve theme with a
"Continue Reading" split:

    div.entry-content
      div.content-visible      ← first ~half of content (preview)
      div.content-hidden       ← rest of content (behind JS toggle)

Inside each content div:
    <h2 class="wp-block-heading">  — editorial section headers (NOT transcript)
    <p>PRESIDENT TRUMP: text...</p> — transcript paragraphs
    <div> ALSO READ: ...           — promo links (strip)
    <p> Editor's Notes: ...        — preamble (strip)
    <p> TRANSCRIPT:                — preamble marker (strip)

All <h2> section headers are editorial additions by singjupost, NOT part of
the original transcript. They are stripped during extraction.

After extracting clean text, we delegate to the raw_text parser for speaker
detection.
"""

import re

import httpx
from bs4 import BeautifulSoup, Tag

from src.transcript.parsers import TranscriptData
from src.transcript.parsers.raw_text import parse_raw_text
from src.utils.logging import log, get_logger

MODULE = "singjupost"
logger = get_logger()

# Patterns for content we strip
_ALSO_READ_RE = re.compile(r"ALSO\s+READ\s*:", re.IGNORECASE)
_PREAMBLE_RE = re.compile(
    r"^(?:Editor'?s?\s+Notes?\s*:|TRANSCRIPT\s*:?\s*$)", re.IGNORECASE
)


def _extract_paragraphs_from_div(div: Tag) -> list[str]:
    """Extract transcript paragraphs from a content div.

    Walks children, skips editorial headers, ALSO READ blocks, ads,
    share buttons, and preamble. Stops at Related Posts.
    """
    paragraphs: list[str] = []

    for child in div.children:
        if not isinstance(child, Tag):
            continue

        # Stop at Related Posts
        if child.name == "h3":
            text = child.get_text(strip=True)
            if "Related" in text:
                break

        # Skip editorial section headers — these are NOT part of the transcript
        if child.name == "h2":
            continue

        # Skip ALSO READ promo divs
        if child.name == "div":
            text = child.get_text(strip=True)
            if _ALSO_READ_RE.search(text):
                continue
            # Skip social share wrappers
            classes = " ".join(child.get("class", []))
            if any(p in classes for p in ["dpsp-", "share", "social"]):
                continue
            continue

        # Paragraphs
        if child.name == "p":
            text = child.get_text(strip=True)
            if not text:
                continue
            # Skip preamble lines
            if _PREAMBLE_RE.match(text):
                continue
            # Skip ALSO READ in paragraph form
            if _ALSO_READ_RE.match(text):
                continue
            paragraphs.append(text)

    return paragraphs


def is_singjupost_url(url: str) -> bool:
    """Check if a URL is from singjupost.com."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        return "singjupost.com" in host
    except Exception:
        return False


async def fetch_singjupost_transcript(url: str) -> TranscriptData:
    """Fetch and parse a SingjuPost transcript page.

    Fetches the HTML, extracts clean transcript text from both the
    visible and hidden content divs, then delegates to the raw_text
    parser for speaker detection.
    """
    log.info(logger, MODULE, "fetch_start", "Fetching SingjuPost transcript",
             url=url)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "SpinCycle/1.0 (fact-checking research)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title from <h1> or og:title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og_title = soup.find("meta", {"property": "og:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()

    # Extract date from meta tags
    date = None
    meta_date = soup.find("meta", {"property": "article:published_time"})
    if meta_date and meta_date.get("content"):
        date = meta_date["content"].strip()
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
    if date and "T" in date:
        date = date.split("T")[0]

    # Extract description
    description = None
    og_desc = soup.find("meta", {"property": "og:description"})
    if og_desc and og_desc.get("content"):
        description = og_desc["content"].strip()

    # Find content area — entry-content contains content-visible + content-hidden
    entry = soup.find("div", class_="entry-content")
    if not entry:
        raise ValueError(f"Could not find entry-content in {url}")

    # Extract from both visible and hidden content divs
    all_paragraphs: list[str] = []

    visible = entry.find("div", class_="content-visible")
    hidden = entry.find("div", class_="content-hidden")

    if visible or hidden:
        # Split layout — combine both halves
        if visible:
            all_paragraphs.extend(_extract_paragraphs_from_div(visible))
        if hidden:
            all_paragraphs.extend(_extract_paragraphs_from_div(hidden))
    else:
        # No split — extract directly from entry-content
        all_paragraphs.extend(_extract_paragraphs_from_div(entry))

    clean_text = "\n\n".join(all_paragraphs)

    if not clean_text or len(clean_text) < 100:
        raise ValueError(f"No transcript content extracted from {url}")

    log.info(logger, MODULE, "text_extracted",
             "Clean transcript text extracted from HTML",
             paragraph_count=len(all_paragraphs),
             char_count=len(clean_text))

    # Delegate to raw_text parser for speaker detection
    td = parse_raw_text(
        content=clean_text,
        url=url,
        title=title,
        date=date,
    )
    td.source_format = "singjupost"
    td.description = description or td.description

    log.info(logger, MODULE, "fetch_done", "SingjuPost transcript fetched",
             url=url, title=td.title,
             turn_count=td.turn_count, speaker_count=len(td.speakers),
             word_count=td.word_count)

    return td
