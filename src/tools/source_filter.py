"""Source quality filter for search results.

Hard filter — if a domain is blocked, the result is silently dropped.

Two-layer filtering:
1. HARD BLOCKLIST — Non-news sources that should never be used: social media,
   forums, video platforms, content farms, AI aggregators, fact-check sites.

2. MBFC FACTUAL CHECK — For potential news sources, check the cached MBFC
   factual rating. Block sources with "mixed" or worse factual reporting.
   This is principled filtering based on accuracy, not political bias.

Also provides warm_mbfc_cache_background() — after bootstrap, every MBFC domain
is already in the DB. This function is now a lightweight no-op that maintains
the interface for search tool callers.
"""

from urllib.parse import urlparse
from typing import Optional

from src.tools.source_ratings import extract_domain
from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

# Hard blocklist — things that are NOT news sources at all.
# These can never be cited as evidence regardless of content quality.
HARD_BLOCKED_DOMAINS = {
    # Social media / forums — user-generated, unvetted content
    "reddit.com",
    "old.reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "threads.net",
    "tumblr.com",
    "4chan.org",
    "boards.4chan.org",
    "discord.com",
    "t.me",
    "linkedin.com",
    "pinterest.com",

    # Blog platforms — occasionally good, mostly unvetted personal content
    "medium.com",
    "substack.com",
    "blogger.com",
    "wordpress.com",
    "livejournal.com",

    # Content farms / SEO spam — not journalism
    "ehow.com",
    "wikihow.com",
    "answers.com",
    "ask.com",
    "reference.com",
    "investopedia.com",
    "healthline.com",
    "webmd.com",
    "verywellhealth.com",

    # Video platforms — not citable text sources
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "twitch.tv",
    "dailymotion.com",
    "rumble.com",
    "bitchute.com",

    # AI-generated / aggregator sites — not primary sources
    "perplexity.ai",
    "you.com",
    "consensus.app",
    "chatgpt.com",
    "claude.ai",
    "bard.google.com",

    # Fact-check sites — we verify independently, don't cite other fact-checkers
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "leadstories.com",
    "checkyourfact.com",

    # Encyclopedia/reference — good for background, not primary evidence
    # (Wikipedia has its own tool, shouldn't come from web search)
    "wikipedia.org",
    "britannica.com",

    # Dictionaries / language reference — never evidence for factual claims
    "merriam-webster.com",
    "dictionary.cambridge.org",
    "collinsdictionary.com",
    "dictionary.com",
    "thesaurus.com",
    "wordreference.com",
    "wiktionary.org",
    "urbandictionary.com",
    "etymonline.com",

    # Name meaning / baby name sites — SearXNG matches person names to these
    "namediscoveries.com",
    "behindthename.com",
    "babynamewizard.com",
    "nameberry.com",
    "names.org",
    "babynames.com",
    "momlovesbest.com",

    # Shopping / commerce / banking — not evidence sources
    "amazon.com",
    "ebay.com",
    "etsy.com",
    "walmart.com",
    "target.com",
    "alibaba.com",

    # Recipe / lifestyle — SearXNG word-matches to these
    "allrecipes.com",
    "food.com",
    "tasty.co",
    "yelp.com",
    "tripadvisor.com",
    "theboheme.com",

    # Microsoft community / tech support forums
    "answers.microsoft.com",
    "support.microsoft.com",
    "support.google.com",
    "support.apple.com",

    # Gaming forums — SearXNG matches "EU" subdomain to EU-related queries
    "blizzard.com",
    "forums.blizzard.com",
    "eu.forums.blizzard.com",

    # App stores — not evidence sources
    "apps.apple.com",
    "play.google.com",

    # Corporate homepages / PR blogs — not independent sources
    "about.google",
    "about.fb.com",
    "blog.google",
    "search.google",
    "newsroom.fb.com",

    # Generic search engines — not primary sources
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "yahoo.com",
    "baidu.com",

    # Entertainment databases — not news or evidence sources
    "imdb.com",
    "m.imdb.com",
    "rottentomatoes.com",
    "metacritic.com",
    "tvtropes.org",
    "letterboxd.com",
    "goodreads.com",

    # Niche forums / medical trackers — not journalism
    "flutrackers.com",
    "patient.info",
    "mayoclinic.org",
    "clevelandclinic.org",
    "medscape.com",

    # Sports / weather / travel — rarely evidence for claims
    "espn.com",
    "weather.com",
    "accuweather.com",
    "booking.com",
    "airbnb.com",
}

# Factual ratings that we block — "mixed" and worse
# MBFC scale: very-high > high > mostly-factual > mixed > low > very-low
BLOCKED_FACTUAL_RATINGS = {"mixed", "low", "very-low"}


def _get_cached_mbfc_rating(domain: str) -> Optional[str]:
    """Check MBFC cache for a domain's factual rating.

    Returns the factual_reporting rating if cached, None if not in cache.
    This is sync and cache-only — no network calls.
    """
    try:
        # Import here to avoid circular imports
        from src.tools.source_ratings import get_source_rating_sync

        rating = get_source_rating_sync(domain)
        if rating:
            return rating.get("factual_reporting")
        return None
    except Exception as e:
        log.warning(logger, MODULE, "mbfc_cache_check_failed",
                    "MBFC cache check failed", domain=domain, error=str(e))
        return None


def is_blocked(url: str) -> bool:
    """Check if a URL should be blocked.

    Two-layer check:
    1. Hard blocklist — social media, video, content farms (always blocked)
    2. MBFC factual rating — news sources with "mixed" or worse are blocked

    If a domain isn't in the hard blocklist AND isn't in MBFC cache,
    it passes through (we err on the side of allowing unknown sources).
    """
    if not url:
        return False

    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()

        # Check hard blocklist (handles subdomains)
        # e.g. "old.reddit.com" → check "old.reddit.com", "reddit.com", "com"
        parts = hostname.split(".")
        for i in range(len(parts)):
            domain = ".".join(parts[i:])
            if domain in HARD_BLOCKED_DOMAINS:
                return True

        # Not hard blocked — check MBFC factual rating
        clean_domain = extract_domain(url)
        if clean_domain:
            factual = _get_cached_mbfc_rating(clean_domain)
            if factual and factual in BLOCKED_FACTUAL_RATINGS:
                log.debug(logger, MODULE, "mbfc_block",
                          f"Blocked {clean_domain} — factual rating: {factual}")
                return True

        return False
    except Exception as e:
        log.warning(logger, MODULE, "is_blocked_error",
                    "is_blocked check failed, allowing domain through",
                    url=url, error=str(e))
        return False


def _block_reason(url: str) -> str | None:
    """Return the reason a URL is blocked, or None if allowed.

    Mirrors is_blocked() logic but returns a descriptive reason string.
    """
    if not url:
        return None

    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()

        # Check hard blocklist (handles subdomains)
        parts = hostname.split(".")
        for i in range(len(parts)):
            domain = ".".join(parts[i:])
            if domain in HARD_BLOCKED_DOMAINS:
                return "hard_blocked"

        # Not hard blocked — check MBFC factual rating
        clean_domain = extract_domain(url)
        if clean_domain:
            factual = _get_cached_mbfc_rating(clean_domain)
            if factual and factual in BLOCKED_FACTUAL_RATINGS:
                return f"low_factual ({factual})"

        return None
    except Exception:
        return None


def filter_results(results: list[dict], url_key: str = "url") -> list[dict]:
    """Filter a list of search results, dropping blocked domains.

    Args:
        results: List of result dicts from any search tool.
        url_key: The key in each dict that contains the URL.

    Returns:
        Filtered list with blocked domains removed.
    """
    filtered = []
    blocked_details: list[dict] = []

    for r in results:
        url = r.get(url_key, "")
        reason = _block_reason(url)
        if reason:
            domain = extract_domain(url) or urlparse(url).hostname or url
            blocked_details.append({"domain": domain, "reason": reason})
        else:
            filtered.append(r)

    if blocked_details:
        blocked_summary = "; ".join(
            f"{b['domain']} ({b['reason']})" for b in blocked_details
        )
        log.info(logger, MODULE, "source_filter", "Filtered low-quality sources",
                 blocked_count=len(blocked_details), kept_count=len(filtered),
                 blocked=blocked_summary)

    return filtered


async def warm_mbfc_cache_background(results: list[dict], url_key: str = "url") -> None:
    """No-op after MBFC index bootstrap — all domains are already in the DB.

    Kept for interface stability with search tool callers. After bootstrap,
    every MBFC-tracked domain has bias/factual in the DB. Unknown domains
    are simply not in MBFC (nothing to scrape).
    """
    pass
