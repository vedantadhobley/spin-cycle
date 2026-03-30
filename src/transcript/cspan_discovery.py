"""C-SPAN discovery client via JW Player API.

Fetches trending/recent C-SPAN programs from the open JW Player playlist
endpoint. No WAF — plain httpx is sufficient.

Playlist: https://jw.c-spanvideo.org/v2/playlists/DBw2z8ej
Returns up to 1024 items with rich metadata (title, speakers, category, etc.).
"""

import re
import time
from datetime import datetime, timezone

import httpx

from src.utils.logging import log, get_logger

MODULE = "cspan_discovery"
logger = get_logger()

PLAYLIST_URL = "https://jw.c-spanvideo.org/v2/playlists/DBw2z8ej"
CACHE_TTL = 300  # 5 minutes

# In-memory cache
_cache: list[dict] | None = None
_cache_time: float = 0

_PROGRAM_ID_RE = re.compile(r"/(\d+)(?:\?|$)")


def _parse_item(item: dict) -> dict | None:
    """Parse a single JW Player playlist item into a CSpanProgram dict."""
    link = item.get("link", "")
    m = _PROGRAM_ID_RE.search(link)
    if not m:
        return None

    program_id = m.group(1)

    # Parse pubdate (unix timestamp)
    pubdate = item.get("pubdate")
    date_str = ""
    if pubdate:
        try:
            dt = datetime.fromtimestamp(int(pubdate), tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            pass

    # Parse speakers (comma-separated string)
    speakers_raw = item.get("speakers", "")
    speakers = (
        [s.strip() for s in speakers_raw.split(",") if s.strip()]
        if speakers_raw else []
    )

    return {
        "program_id": program_id,
        "title": item.get("title", ""),
        "category": item.get("category", ""),
        "format": item.get("format", ""),
        "speakers": speakers,
        "date": date_str,
        "duration": item.get("duration", 0),
        "url": link,
        "description": item.get("description", ""),
    }


async def fetch_available(limit: int = 100) -> list[dict]:
    """Fetch available C-SPAN programs from JW Player.

    Returns structured list of program metadata, cached for 5 minutes.
    """
    global _cache, _cache_time

    if _cache is not None and (time.time() - _cache_time) < CACHE_TTL:
        return _cache[:limit]

    log.info(logger, MODULE, "fetch_start", "Fetching C-SPAN discovery feed")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(PLAYLIST_URL, params={"format": "json"})
        resp.raise_for_status()

    data = resp.json()
    items = data.get("playlist", [])

    programs = []
    for item in items:
        parsed = _parse_item(item)
        if parsed:
            programs.append(parsed)

    _cache = programs
    _cache_time = time.time()

    log.info(logger, MODULE, "fetch_done", "C-SPAN discovery complete",
             total_items=len(programs))

    return programs[:limit]
