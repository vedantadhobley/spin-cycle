"""Source quality filter for search results.

Filters out low-quality, user-generated, or non-citable sources from
search results before the research agent ever sees them. This is a
hard filter — if a domain is blocked, the result is silently dropped.

Two-layer filtering:
1. HARD BLOCKLIST — Non-news sources that should never be used: social media,
   forums, video platforms, content farms, AI aggregators, fact-check sites.
   These aren't news sources at all.

2. MBFC FACTUAL CHECK — For potential news sources, check the cached MBFC
   factual rating. Block sources with "mixed" or worse factual reporting.
   This is principled filtering based on accuracy, not political bias.

The agent prompt also instructs the LLM to prefer authoritative sources,
but this filter catches what the LLM might miss.
"""

from urllib.parse import urlparse
from typing import Optional

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
        logger.debug(f"MBFC cache check failed for {domain}: {e}")
        return None


def _extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


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
        clean_domain = _extract_domain(url)
        if clean_domain:
            factual = _get_cached_mbfc_rating(clean_domain)
            if factual and factual in BLOCKED_FACTUAL_RATINGS:
                log.debug(logger, MODULE, "mbfc_block", 
                          f"Blocked {clean_domain} — factual rating: {factual}")
                return True

        return False
    except Exception:
        return False


def filter_results(results: list[dict], url_key: str = "url") -> list[dict]:
    """Filter a list of search results, dropping blocked domains.

    Args:
        results: List of result dicts from any search tool.
        url_key: The key in each dict that contains the URL.

    Returns:
        Filtered list with blocked domains removed.
    """
    filtered = []
    blocked_count = 0

    for r in results:
        url = r.get(url_key, "")
        if is_blocked(url):
            blocked_count += 1
        else:
            filtered.append(r)

    if blocked_count > 0:
        log.debug(logger, MODULE, "source_filter", "Filtered low-quality sources",
                  blocked_count=blocked_count, kept_count=len(filtered))

    return filtered


async def populate_mbfc_cache(results: list[dict], url_key: str = "url") -> None:
    """Pre-populate MBFC cache for domains in search results.

    Call this BEFORE filter_results() so that is_blocked() has MBFC data
    for unknown domains. Without this, domains not in the seed data would
    pass through unfiltered.

    This async-scrapes MBFC for any domain not already cached. Scraping
    failures are silently ignored — unknown domains pass through and the
    judge will see them tagged as "[Unrated source]".

    Runs concurrently with a cap to avoid hammering MBFC.
    """
    import asyncio

    # Collect unique domains that aren't hard-blocked and aren't cached
    domains_to_check = set()
    for r in results:
        url = r.get(url_key, "")
        if not url:
            continue

        domain = _extract_domain(url)
        if not domain:
            continue

        # Skip if hard-blocked (no point checking MBFC)
        parts = domain.split(".")
        hard_blocked = False
        for i in range(len(parts)):
            if ".".join(parts[i:]) in HARD_BLOCKED_DOMAINS:
                hard_blocked = True
                break
        if hard_blocked:
            continue

        # Skip if already cached
        cached_rating = _get_cached_mbfc_rating(domain)
        if cached_rating is not None:
            continue

        domains_to_check.add(domain)

    if not domains_to_check:
        return

    log.debug(logger, MODULE, "mbfc_populate_start",
              "Pre-populating MBFC cache for search results",
              domain_count=len(domains_to_check))

    # Import here to avoid circular imports
    from src.tools.source_ratings import get_source_rating

    # Scrape concurrently, cap at 4 to avoid hammering MBFC
    sem = asyncio.Semaphore(4)

    async def _check(domain: str) -> None:
        async with sem:
            try:
                await get_source_rating(domain)
            except Exception:
                pass  # Failures are fine — domain will be "unrated"

    await asyncio.gather(*[_check(d) for d in domains_to_check])

    log.debug(logger, MODULE, "mbfc_populate_done",
              "MBFC cache populated",
              domain_count=len(domains_to_check))
