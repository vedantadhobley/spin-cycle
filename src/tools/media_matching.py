"""Media outlet matching, publisher ownership checking, and MBFC owner extraction.

Used by seed ranking (research phase) and judge annotation to detect
when evidence comes from media outlets affiliated with interested parties.

Three functions:
1. extract_owners_from_mbfc: Extract PERSON/ORG names from MBFC ownership
   strings via SpaCy NER. Used by research phase to Wikidata-expand
   media owners and discover their networks before seed ranking.

2. url_matches_media: Does the source URL match a known media outlet
   name? Handles domain aliases (e.g., "Washington Post" →
   washingtonpost.com, wapo).

3. check_publisher_ownership: Does the MBFC 'ownership' field mention
   any interested party? Catches indirect ownership ties
   (e.g., Rupert Murdoch → Wall Street Journal).
"""

from src.tools.source_ratings import get_source_rating_sync
from src.utils.logging import log, get_logger
from src.utils.ner import extract_entity_names

MODULE = "media_matching"
logger = get_logger()

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


# Generic corporate/industry terms that appear in MBFC ownership strings
# but aren't meaningful party identifiers.  Single-word parties matching
# these are skipped in check_publisher_ownership() to prevent false positives
# (e.g. "Media" in all_parties matching every ownership string that contains
# the word "media").
_GENERIC_OWNERSHIP_TERMS = {
    "media", "group", "inc", "corp", "corporation", "company",
    "holdings", "enterprises", "foundation", "institute", "network",
    "news", "digital", "publishing", "entertainment", "international",
    "limited", "ltd", "llc", "partners", "trust", "association",
}

# Generic ownership strings that won't yield useful PERSON/ORG entities.
_GENERIC_OWNERSHIP = {
    "state-funded", "state funded", "non-profit", "nonprofit",
    "government", "government-funded", "government funded",
    "publicly traded", "publicly-traded", "public", "private",
    "employee-owned", "employee owned", "independent",
    "cooperative", "co-operative", "trust", "foundation",
    "university", "academic", "unknown", "n/a", "",
}


def extract_owners_from_mbfc(ownership: str | None) -> list[str]:
    """Extract PERSON/ORG names from an MBFC free-text ownership string.

    MBFC ownership fields contain strings like:
        "Owned by Rupert Murdoch's News Corporation"
        "Jeff Bezos (since 2013)"
        "State-Funded"  → no useful entities

    Uses SpaCy NER to extract real person/org names. Skips generic
    strings that describe funding models rather than specific owners.

    Returns:
        List of owner names suitable for Wikidata expansion.
    """
    if not ownership:
        return []

    # Skip generic ownership descriptors
    if ownership.strip().lower() in _GENERIC_OWNERSHIP:
        return []

    owners = extract_entity_names(ownership, labels={"PERSON", "ORG"})
    log.info(logger, MODULE, "owners_extracted",
             "Extracted owner entities from MBFC ownership string",
             ownership=ownership, owner_count=len(owners), owners=owners)
    return owners


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
                    log.info(logger, MODULE, "url_media_match",
                             "URL matches affiliated media via alias",
                             url=url_lower, media=media_outlet, alias=alias)
                    return True

    # Fall back to generic normalization: strip spaces, strip leading "the"
    stripped = media_lower.lstrip()
    if stripped.startswith("the "):
        stripped = stripped[4:]
    normalized = stripped.replace(" ", "")
    if normalized and len(normalized) > 3 and normalized in url_lower:
        log.info(logger, MODULE, "url_media_match",
                 "URL matches affiliated media via normalized name",
                 url=url_lower, media=media_outlet)
        return True

    # Also try hyphenated: "washington-post"
    hyphenated = media_lower.replace(" ", "-")
    if hyphenated in url_lower:
        log.info(logger, MODULE, "url_media_match",
                 "URL matches affiliated media via hyphenated name",
                 url=url_lower, media=media_outlet)
        return True

    return False


def check_publisher_ownership(url: str, all_parties: list[str]) -> str | None:
    """Check if a source URL's publisher is owned by an interested party.

    Uses the MBFC 'ownership' field (already cached from warm_mbfc_cache_background)
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
        party_lower = party.lower().strip()
        if len(party_lower) < 4:
            continue  # Skip very short names to avoid false matches
        # Skip single-word generic corporate terms (e.g. "Media", "Group")
        words = party_lower.split()
        if len(words) == 1 and party_lower in _GENERIC_OWNERSHIP_TERMS:
            log.debug(logger, MODULE, "generic_term_skipped",
                      "Skipped generic corporate term in ownership check",
                      party=party, url=url)
            continue
        if party_lower in ownership_lower:
            log.info(logger, MODULE, "publisher_ownership_match",
                     "Publisher ownership matches interested party",
                     url=url, party=party,
                     ownership=rating["ownership"])
            return party

    return None
