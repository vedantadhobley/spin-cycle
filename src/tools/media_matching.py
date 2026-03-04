"""Media outlet domain matching and publisher ownership checking.

Used by seed ranking (research phase) and judge annotation to detect
when evidence comes from media outlets affiliated with interested parties.

Two checks:
1. URL matching (url_matches_media): Does the source URL match a known
   media outlet name? Handles domain aliases (e.g., "Washington Post" →
   washingtonpost.com, wapo).

2. Publisher ownership (check_publisher_ownership): Does the MBFC
   'ownership' field mention any interested party? Catches indirect
   ownership ties (e.g., Rupert Murdoch → Wall Street Journal).
"""

from src.tools.source_ratings import get_source_rating_sync

# Common media outlet domain aliases for better URL matching.
# Maps common name variations to their likely domain patterns.
MEDIA_DOMAIN_ALIASES = {
    "washington post": ["washingtonpost", "wapo"],
    "new york times": ["nytimes", "nyt"],
    "wall street journal": ["wsj"],
    "fox news": ["foxnews", "fox"],
    "los angeles times": ["latimes"],
    "chicago tribune": ["chicagotribune"],
    "new york post": ["nypost"],
    "huffington post": ["huffpost", "huffingtonpost"],
    "daily mail": ["dailymail"],
    "the guardian": ["theguardian", "guardian"],
    "the atlantic": ["theatlantic", "atlantic"],
    "financial times": ["ft.com"],
    "daily beast": ["thedailybeast", "dailybeast"],
}


def url_matches_media(url_lower: str, media_outlet: str) -> bool:
    """Check if a URL matches a media outlet name, handling common variations.

    This handles cases like multi-word outlet names matching their
    concatenated domain (e.g., "Some Outlet" matching someoutlet.com).

    Args:
        url_lower: Lowercase URL to check
        media_outlet: Name of media outlet

    Returns:
        True if URL appears to be from this media outlet
    """
    media_lower = media_outlet.lower()

    # Check known aliases first
    for name, aliases in MEDIA_DOMAIN_ALIASES.items():
        if name in media_lower or media_lower in name:
            for alias in aliases:
                if alias in url_lower:
                    return True

    # Fall back to generic normalization: strip spaces, strip leading "the"
    stripped = media_lower.lstrip()
    if stripped.startswith("the "):
        stripped = stripped[4:]
    normalized = stripped.replace(" ", "")
    if normalized and len(normalized) > 3 and normalized in url_lower:
        return True

    # Also try hyphenated: "washington-post"
    hyphenated = media_lower.replace(" ", "-")
    if hyphenated in url_lower:
        return True

    return False


def check_publisher_ownership(url: str, all_parties: list[str]) -> str | None:
    """Check if a source URL's publisher is owned by an interested party.

    Uses the MBFC 'ownership' field (already cached from populate_mbfc_cache)
    to cross-reference against all_parties. Catches cases like:
    - A news outlet owned by Person X when Person X is in all_parties

    Returns the matching party name if found, None otherwise.
    """
    if not url or url == "N/A" or not all_parties:
        return None

    rating = get_source_rating_sync(url)
    if not rating or not rating.get("ownership"):
        return None

    ownership_lower = rating["ownership"].lower()

    for party in all_parties:
        party_lower = party.lower()
        if len(party_lower) < 4:
            continue  # Skip very short names to avoid false matches
        if party_lower in ownership_lower:
            return party

    return None
