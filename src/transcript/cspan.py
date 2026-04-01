"""C-SPAN transcript fetcher.

Uses Playwright (headless Chromium) to solve CloudFront WAF JS challenges.
Navigates to the program page, intercepts the transcript API response that
the page loads automatically, and extracts metadata from HTML.

The transcript API returns closed-caption data with speaker labels in the
`cc_name` field (e.g. "GOV. ABBOT", "SEN. SCHUMER"). Some programs use ">>"
as a generic caption marker without speaker attribution.

Speaker attribution post-processing:
  1. Extract proper names from HTML /person/ links on the program page
  2. Map cc_name labels to proper names by last-name matching
     (e.g. "SEC. RUBIO" → "Marco Rubio")
  3. For unnamed (>>) segments when there's one primary speaker:
     short segments with question marks → "Reporter",
     otherwise → the primary speaker

Text arrives in ALL CAPS from the captioning system — we title-case it for
readability and normalize multi-line caption blocks into paragraphs.
"""

import json
import re
import asyncio

from src.transcript.parsers import TranscriptData, SpeakerTurn, normalize_turns
from src.utils.logging import log, get_logger

MODULE = "cspan"
logger = get_logger()

_PROGRAM_ID_RE = re.compile(r"/(\d+)(?:\?|$)")
_CSPAN_DOMAIN = re.compile(r"(?:^|\.)c-span\.org$", re.IGNORECASE)

# Honorific stripping for C-SPAN speaker names
# Includes abbreviated forms C-SPAN captioners use (PRES., SEC., SEN., REP., GOV.)
_HONORIFICS = re.compile(
    r"^(?:Pres\.|President|Vice\s+Pres\.|Vice\s+President|"
    r"Sec\.|Secretary|Sen\.|Senator|Rep\.|Representative|"
    r"Congressman|Congresswoman|Governor|Gov\.|"
    r"Mayor|Ambassador|Gen\.|General|Adm\.|Admiral|"
    r"Director|Dir\.|Chairman|Chairwoman|Chair|"
    r"Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Speaker|Leader)\s+",
    re.IGNORECASE,
)

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


def _strip_speaker_honorific(name: str) -> str:
    """Strip common political honorifics and title-case ALL CAPS names."""
    stripped = _HONORIFICS.sub("", name).strip()
    if stripped == stripped.upper() and len(stripped) > 2:
        stripped = stripped.title()
    return stripped


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


# ---------------------------------------------------------------------------
# Speaker attribution
# ---------------------------------------------------------------------------

# Max word count for a segment to be considered a "reporter question"
_REPORTER_MAX_WORDS = 40


def _extract_person_names(soup) -> list[str]:
    """Extract proper names from /person/ links on the program page."""
    names = []
    seen = set()
    for a in soup.find_all("a", href=True):
        if "/person/" in a["href"]:
            name = a.get_text().strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _build_cc_name_map(cc_names: list[str], person_names: list[str]) -> dict[str, str]:
    """Map cc_name labels to proper names by last-name matching.

    Examples:
        cc_names=["SEC. RUBIO", "PRES. TRUMP"], person_names=["Marco Rubio", "Donald J. Trump"]
        → {"SEC. RUBIO": "Marco Rubio", "PRES. TRUMP": "Donald J. Trump"}
    """
    mapping: dict[str, str] = {}

    # Build last-name → proper-name lookup from person links
    last_name_lookup: dict[str, str] = {}
    for name in person_names:
        parts = name.split()
        if parts:
            # Use the last word as the last name
            last = parts[-1].lower()
            last_name_lookup[last] = name

    for cc in cc_names:
        # Skip generic roles — these don't map to specific people
        upper = cc.upper().strip()
        if upper in (">>", "", "HOST", "GUEST", "CALLER", "REPORTER"):
            continue

        # Strip honorific to get the bare name, then match by last name
        bare = _strip_speaker_honorific(cc)
        bare_parts = bare.split()
        if bare_parts:
            last = bare_parts[-1].lower()
            if last in last_name_lookup:
                mapping[cc] = last_name_lookup[last]

    return mapping


def _attribute_unnamed_turns(
    turns: list[SpeakerTurn],
    person_names: list[str],
    cc_name_map: dict[str, str],
) -> None:
    """Attribute unnamed (>>) turns using the primary speaker heuristic.

    If there's one dominant speaker (from person links or cc_name labels),
    assign long unnamed turns to them and short question turns to
    "Reporter".

    Modifies turns in place.
    """
    if not person_names:
        return

    # Determine the primary speaker: if only one person is listed, use them.
    # If multiple, check if one dominates the named turns.
    named_counts: dict[str, int] = {}
    for turn in turns:
        if turn.speaker not in ("Unknown", "Reporter"):
            named_counts[turn.speaker] = named_counts.get(turn.speaker, 0) + 1

    # Candidates: person link names + anyone who already has named turns
    primary = None
    if len(person_names) == 1:
        primary = person_names[0]
    elif named_counts:
        # Use the most frequently named speaker
        top_speaker = max(named_counts, key=named_counts.get)
        total_named = sum(named_counts.values())
        # Only use as primary if they dominate (>60% of named turns)
        if named_counts[top_speaker] / total_named > 0.6:
            primary = top_speaker

    if not primary:
        return

    unknown_count = sum(1 for t in turns if t.speaker == "Unknown")
    if unknown_count == 0:
        return

    log.info(logger, MODULE, "speaker_attribution",
             "Attributing unnamed turns",
             primary_speaker=primary, unnamed_count=unknown_count)

    for turn in turns:
        if turn.speaker != "Unknown":
            continue
        words = len(turn.text.split())
        has_question = "?" in turn.text
        if words <= _REPORTER_MAX_WORDS and has_question:
            turn.speaker = "Reporter"
        else:
            turn.speaker = primary


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

    # Parse transcript parts
    parts = transcript_json["parts"]
    raw_turns: list[SpeakerTurn] = []
    raw_speakers: list[str] = []

    # Collect unique cc_names for mapping
    unique_cc_names: list[str] = []
    for part in parts:
        cn = (part.get("cc_name") or "").strip()
        if cn and cn != ">>" and cn not in unique_cc_names:
            unique_cc_names.append(cn)

    # Build cc_name → proper name mapping from person links
    cc_name_map = _build_cc_name_map(unique_cc_names, person_names)

    for i, part in enumerate(parts):
        # C-SPAN uses cc_name for speaker in caption data
        speaker_raw = (part.get("cc_name") or part.get("speakername") or "").strip()
        text = (part.get("text") or "").strip()
        if not text:
            continue

        # ">>" is a generic caption speaker-change marker
        if speaker_raw in (">>", ""):
            speaker = "Unknown"
        elif speaker_raw in cc_name_map:
            # Map to proper name from person links
            speaker = cc_name_map[speaker_raw]
            if speaker_raw not in raw_speakers:
                raw_speakers.append(speaker_raw)
        else:
            speaker = _strip_speaker_honorific(speaker_raw)
            if speaker_raw not in raw_speakers:
                raw_speakers.append(speaker_raw)

        text = _clean_caption_text(text)
        if not text:
            continue

        raw_turns.append(SpeakerTurn(speaker=speaker, text=text))

    if not raw_turns:
        raise ValueError(f"No transcript turns parsed for program {program_id}")

    # Attribute unnamed turns using primary speaker heuristic
    _attribute_unnamed_turns(raw_turns, person_names, cc_name_map)

    # Merge consecutive same-speaker caption fragments
    turns = normalize_turns(raw_turns)

    # Build speaker list from what's actually in the turns now
    turn_speakers = list(dict.fromkeys(t.speaker for t in turns))

    # Build aliases: cc_name → proper name (or honorific-stripped)
    aliases: dict[str, list[str]] = {}
    for raw in raw_speakers:
        proper = cc_name_map.get(raw, _strip_speaker_honorific(raw))
        if raw.strip() != proper:
            if proper not in aliases:
                aliases[proper] = []
            if raw.strip() not in aliases[proper]:
                aliases[proper].append(raw.strip())

    log.info(logger, MODULE, "fetch_done", "C-SPAN transcript fetched",
             program_id=program_id, title=title,
             turn_count=len(turns), speaker_count=len(turn_speakers),
             person_names=person_names,
             cc_name_mapping=cc_name_map)

    return TranscriptData(
        url=url,
        title=title,
        date=date,
        description=description,
        speakers=turn_speakers,
        turns=turns,
        source_format="cspan",
        speaker_aliases=aliases if aliases else {},
    )


async def cleanup():
    """Shut down the Playwright browser. Call on worker shutdown."""
    global _session
    if _session:
        await _session.close()
        _session = None
        log.info(logger, MODULE, "cleanup", "Playwright session closed")
