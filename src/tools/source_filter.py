"""Source quality filter for search results.

Filters out low-quality, user-generated, or non-citable sources from
search results before the research agent ever sees them. This is a
hard filter — if a domain is blocked, the result is silently dropped.

The agent prompt also instructs the LLM to prefer authoritative sources,
but this filter catches what the LLM might miss.
"""

from urllib.parse import urlparse

from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

# Domains to block entirely — user-generated content, forums, social media,
# content farms, and other non-citable sources.
BLOCKED_DOMAINS = {
    # Social media / forums
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
    "medium.com",          # Some good content, but mostly unvetted blogs
    "substack.com",        # Same — some good, mostly unvetted

    # Content farms / SEO spam
    "ehow.com",
    "wikihow.com",
    "answers.com",
    "ask.com",
    "reference.com",
    "investopedia.com",    # Too generic / SEO-driven for fact-checking
    "healthline.com",
    "webmd.com",
    "verywellhealth.com",

    # Video platforms (not citable text sources)
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "twitch.tv",
    "dailymotion.com",

    # AI-generated / aggregator sites
    "perplexity.ai",
    "you.com",
    "consensus.app",

    # Fact-check sites (we do our own verification)
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "leadstories.com",
    "checkyourfact.com",

    # Tabloids / unreliable news sources
    "dailymail.co.uk",
    "mail.co.uk",
    "mailonline.com",
    "thesun.co.uk",
    "sun.co.uk",
    "mirror.co.uk",
    "express.co.uk",
    "nypost.com",
    "pagesix.com",
    "tmz.com",
    "buzzfeed.com",
    "buzzfeednews.com",
    "dailycaller.com",
    "dailywire.com",
    "breitbart.com",
    "infowars.com",
    "naturalnews.com",
    "thegatewaypundit.com",
    "occupydemocrats.com",
    "newsmax.com",
    "oann.com",
    "rawstory.com",
    "theblaze.com",
    "washingtontimes.com",
    "epochtimes.com",
    "rt.com",
    "sputniknews.com",
}


def is_blocked(url: str) -> bool:
    """Check if a URL belongs to a blocked domain.

    Handles subdomains — e.g. "old.reddit.com" matches the "reddit.com" block.
    """
    if not url:
        return False

    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()

        # Check exact match and parent domains
        # e.g. "old.reddit.com" → check "old.reddit.com", "reddit.com", "com"
        parts = hostname.split(".")
        for i in range(len(parts)):
            domain = ".".join(parts[i:])
            if domain in BLOCKED_DOMAINS:
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
        log.info(logger, MODULE, "source_filter", "Filtered low-quality sources",
                 blocked_count=blocked_count, kept_count=len(filtered))

    return filtered
