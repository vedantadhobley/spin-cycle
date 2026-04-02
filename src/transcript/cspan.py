"""C-SPAN transcript fetcher — pure data extraction.

Uses Playwright (headless Chromium) to solve CloudFront WAF JS challenges.
Navigates to the program page, intercepts the transcript API response that
the page loads automatically, and extracts metadata from HTML.

The transcript API returns closed-caption data with speaker labels:
  - ``speakername``: actual name (e.g. "Mimi Geerges") — not always present
  - ``cc_name``: caption label (e.g. "HOST", "SEC. RUBIO", ">>")

This parser is a pure extractor — it returns raw speaker labels and metadata.
Speaker name resolution (honorific stripping, cc_name mapping to proper names)
happens in ``speaker_resolution.resolve_speakers()`` in the activity layer.

Turns are returned as raw caption segments (no same-speaker merging).
The attribute_speakers activity normalizes once after LLM attribution.

Text arrives in ALL CAPS from the captioning system — we title-case it for
readability and normalize multi-line caption blocks into paragraphs.
"""

import json
import re
import asyncio

from src.transcript.parsers import TranscriptData, SpeakerTurn, clean_speaker_name
from src.utils.logging import log, get_logger

MODULE = "cspan"
logger = get_logger()

_PROGRAM_ID_RE = re.compile(r"/(\d+)(?:\?|$)")
_CSPAN_DOMAIN = re.compile(r"(?:^|\.)c-span\.org$", re.IGNORECASE)
_GENERIC_SPEAKER_ID = re.compile(r"^spk_\d+$", re.IGNORECASE)

# Module-level singleton
_session: "CSpanSession | None" = None
_session_lock = asyncio.Lock()


class CSpanSession:
    """WAF-solving browser session manager.

    Launches Chromium once, keeps BrowserContext alive across requests so
    WAF cookies persist. Recreates context if it dies or gets challenged.
    """

    def __init__(self):
        self._browser = None
        self._context = None
        self._playwright = None

    async def ensure_context(self):
        """Launch Chromium and create context if not already running."""
        if self._context:
            try:
                _ = self._context.pages
                return
            except Exception:
                await self._close_context()

        from playwright.async_api import async_playwright

        if not self._playwright:
            self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        # Remove navigator.webdriver flag that WAFs use to detect automation
        await self._context.add_init_script(
            'Object.defineProperty(navigator, "webdriver", {get: () => undefined});'
        )
        log.info(logger, MODULE, "browser_started", "Playwright Chromium launched")

    async def fetch_program(self, url: str, program_id: str) -> tuple[str, dict]:
        """Navigate to program page and intercept transcript API response.

        Returns (page_html, transcript_json). The page automatically requests
        the transcript API — we intercept that response rather than making a
        separate call, which avoids WAF issues with direct API navigation.
        """
        await self.ensure_context()
        page = await self._context.new_page()

        transcript_data = {}

        async def _on_response(response):
            nonlocal transcript_data
            resp_url = response.url
            if (
                "transcript" in resp_url
                and f"videoId={program_id}" in resp_url
                and not transcript_data
            ):
                try:
                    body = await response.text()
                    if len(body) > 100:  # Skip the small metadata response
                        transcript_data = json.loads(body)
                except Exception:
                    pass

        page.on("response", _on_response)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # Wait for transcript API response to be intercepted
            for _ in range(20):  # up to 10s
                if transcript_data:
                    break
                await page.wait_for_timeout(500)

            html = await page.content()
            return html, transcript_data
        finally:
            await page.close()

    async def _close_context(self):
        """Close browser context and browser."""
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        self._context = None
        self._browser = None

    async def close(self):
        """Graceful shutdown — close context, browser, and Playwright."""
        await self._close_context()
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


async def _get_session() -> CSpanSession:
    """Get or create the singleton CSpanSession."""
    global _session
    async with _session_lock:
        if _session is None:
            _session = CSpanSession()
        return _session


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_program_id(url: str) -> str | None:
    """Extract numeric program ID from a C-SPAN URL.

    Works with URLs like:
        https://www.c-span.org/program/white-house-event/.../676455
        https://www.c-span.org/program/676455
    """
    m = _PROGRAM_ID_RE.search(url)
    return m.group(1) if m else None


def is_cspan_url(url: str) -> bool:
    """Check if a URL is from c-span.org."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        return bool(_CSPAN_DOMAIN.search(host))
    except Exception:
        return False


def _clean_caption_text(text: str) -> str:
    """Clean ALL CAPS closed-caption text into readable prose.

    C-SPAN captions arrive as multi-line ALL CAPS blocks with hard line breaks.
    Merge lines into paragraphs and sentence-case the text.
    """
    # Collapse hard line breaks into spaces (caption line wraps)
    collapsed = re.sub(r"\n+", " ", text)
    # Normalize whitespace
    collapsed = " ".join(collapsed.split())
    if not collapsed:
        return collapsed
    # Check if mostly uppercase (>70% of alpha chars) — handles edge cases
    # where a few lowercase chars or special chars prevent exact equality
    alpha_chars = [c for c in collapsed if c.isalpha()]
    if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7:
        # Sentence-case: lowercase everything, then capitalize after . ? !
        collapsed = collapsed.lower()
        collapsed = collapsed[0].upper() + collapsed[1:]
        collapsed = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            collapsed,
        )
    return collapsed


def _extract_person_names(soup) -> list[str]:
    """Extract proper names from /person/ links on the program page."""
    names = []
    seen = set()
    for a in soup.find_all("a", href=True):
        if "/person/" in a["href"]:
            name = clean_speaker_name(a.get_text().strip())
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


# ---------------------------------------------------------------------------
# Core fetcher
# ---------------------------------------------------------------------------

async def fetch_cspan_transcript(url_or_id: str) -> TranscriptData:
    """Fetch a C-SPAN transcript by URL or program ID.

    Steps:
    1. Extract program ID from URL (or use directly if numeric)
    2. Navigate to program page via Playwright (solves WAF)
    3. Intercept transcript API response (loaded by page JS automatically)
    4. Extract title/date from HTML meta tags
    5. Parse caption segments into TranscriptData
    """
    if url_or_id.isdigit():
        program_id = url_or_id
        url = f"https://www.c-span.org/program/{program_id}"
    else:
        program_id = extract_program_id(url_or_id)
        if not program_id:
            raise ValueError(
                f"Could not extract C-SPAN program ID from: {url_or_id}"
            )
        url = url_or_id

    session = await _get_session()

    log.info(logger, MODULE, "fetch_start", "Fetching C-SPAN transcript",
             program_id=program_id, url=url)

    # Single page navigation: get HTML + intercept transcript API response
    html, transcript_json = await session.fetch_program(url, program_id)

    if not transcript_json or not transcript_json.get("parts"):
        raise ValueError(f"No transcript data found for program {program_id}")

    # Extract metadata from HTML
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title:
        title = og_title.get("content", "")
    if not title:
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.sub(r"\s*\|.*$", "", title)

    # Extract description — og:description
    description = None
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        description = og_desc["content"].strip()

    # Extract date — try multiple sources
    date = None
    # 1. <time datetime="YYYY-MM-DD"> element (most reliable)
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        date = time_tag["datetime"]
    # 2. JSON-LD uploadDate
    if not date:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                ld = json.loads(script.string or "")
                upload = (
                    ld.get("uploadDate")
                    or (ld.get("video") or {}).get("uploadDate")
                )
                if upload:
                    date = upload
                    break
            except (ValueError, TypeError):
                pass
    # 3. OG meta tags (fallback)
    if not date:
        for prop in ("article:published_time", "og:updated_time"):
            meta_date = soup.find("meta", property=prop)
            if meta_date and meta_date.get("content"):
                date = meta_date["content"]
                break
    # Normalize to date-only
    if date and "T" in date:
        date = date.split("T")[0]

    # Extract person names from HTML for speaker attribution
    person_names = _extract_person_names(soup)

    # Parse transcript parts — raw extraction, no resolution
    parts = transcript_json["parts"]
    raw_turns: list[SpeakerTurn] = []

    for i, part in enumerate(parts):
        # C-SPAN provides speaker identity in two fields:
        #   speakername: actual name (e.g. "Mimi Geerges") — not always present
        #   cc_name: caption label (e.g. "HOST", "SEC. RUBIO", ">>")
        # Use both: prefer speakername when it's a real name, fall back to
        # cc_name when speakername is absent or a generic ID (spk_0, spk_1)
        speakername = (part.get("speakername") or "").strip()
        cc_name = (part.get("cc_name") or "").strip()

        if speakername and not _GENERIC_SPEAKER_ID.match(speakername):
            speaker_raw = speakername
        elif cc_name and cc_name != ">>" and not _GENERIC_SPEAKER_ID.match(cc_name):
            speaker_raw = cc_name
        else:
            speaker_raw = ""

        text = (part.get("text") or "").strip()
        if not text:
            continue

        if not speaker_raw:
            speaker = "Unknown"
        else:
            speaker = speaker_raw  # raw label, resolution happens later

        text = _clean_caption_text(text)
        if not text:
            continue

        raw_turns.append(SpeakerTurn(speaker=speaker, text=text))

    if not raw_turns:
        raise ValueError(f"No transcript turns parsed for program {program_id}")

    # Raw caption segments returned as-is — resolution and normalization
    # happen in the activity layer (resolve_speakers -> attribute_speakers
    # -> _normalize_transcript).
    turns = raw_turns

    # Build speaker list: raw turn speakers + person names from HTML links.
    # Person names may not appear in turns when all cc_names are ">>"
    # but the LLM attribution activity needs them to know who to attribute to.
    turn_speakers = list(dict.fromkeys(t.speaker for t in turns))
    for pn in person_names:
        if pn not in turn_speakers:
            turn_speakers.append(pn)

    log.info(logger, MODULE, "fetch_done", "C-SPAN transcript fetched",
             program_id=program_id, title=title,
             turn_count=len(turns), speaker_count=len(turn_speakers),
             person_names=person_names)

    return TranscriptData(
        url=url,
        title=title,
        date=date,
        description=description,
        speakers=turn_speakers,
        turns=turns,
        source_format="cspan",
        speaker_aliases={},
        speaker_metadata={"person_names": person_names},
    )


async def cleanup():
    """Shut down the Playwright browser. Call on worker shutdown."""
    global _session
    if _session:
        await _session.close()
        _session = None
        log.info(logger, MODULE, "cleanup", "Playwright session closed")
