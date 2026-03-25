"""Detect when interested parties are quoted or cited in evidence text.

Three-layer name matching:
1. Exact party name (word-boundary matched)
2. Programmatic variants (last names from person names)
3. Wikidata aliases (e.g., "U.S." for "United States", "Trump" for "Donald Trump")

Each match must appear in an attribution context (said, according to, etc.)
to avoid flagging mere mentions.

Used by the judge to flag self-serving statements: when a claim is ABOUT
entity X and the evidence quotes X's statements, those statements cannot
independently verify claims about X.
"""


# Words that indicate quoted/cited speech
_ATTRIBUTION_WORDS = frozenset({
    "said", "says", "stated", "states", "reported", "reports",
    "announced", "announces", "confirmed", "confirms", "denied", "denies",
    "claimed", "claims", "concluded", "concludes", "found", "finds",
    "determined", "issued", "released", "testified", "told",
    "declared", "asserted", "maintained", "contended", "argued",
})

# Possessive patterns indicating source attribution
_POSSESSIVE_INDICATORS = (
    "'s statement", "'s investigation", "'s conclusion",
    "'s report", "'s official", "'s finding", "'s analysis",
    "'s determination", "'s position", "'s claim", "'s assessment",
    "'s spokesperson", "'s press",
)

# Name stopwords — never useful as standalone match variants
_NAME_STOPWORDS = frozenset({
    "the", "of", "and", "for", "in", "on", "at", "to", "a", "an",
    "by", "with", "from",
})

# Word boundary characters for precise matching
_BOUNDARY_CHARS = frozenset(' .,;:!?\'"()[]{}\n\t-/')


def _has_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if a match has word boundaries on both sides."""
    start_ok = (start == 0) or (text[start - 1] in _BOUNDARY_CHARS)
    end_ok = (end >= len(text)) or (text[end] in _BOUNDARY_CHARS)
    return start_ok and end_ok


def _find_with_boundary(text: str, pattern: str) -> int:
    """Find pattern in text with word boundaries. Returns index or -1."""
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            return -1
        if _has_word_boundary(text, idx, idx + len(pattern)):
            return idx
        start = idx + 1


def _generate_name_variants(party: str) -> list[str]:
    """Extract identifying words from a multi-word party name.

    Person names → last name ("Donald Trump" → ["Trump"])
    For organizations, Wikidata aliases are the primary mechanism;
    this provides a programmatic fallback.
    """
    words = party.split()
    if len(words) < 2:
        return []

    significant = [w for w in words if w.lower() not in _NAME_STOPWORDS and len(w) > 2]
    if len(significant) < 2:
        return []

    last = significant[-1]
    if last[0:1].isupper():
        return [last]

    return []


def _check_attribution_near(content_lower: str, name_idx: int, name_len: int) -> bool:
    """Check for attribution patterns near a name mention.

    Uses proximity windows rather than exact adjacency to catch patterns like
    "Trump administration said" and "according to the Iranian government".
    """
    name_end = name_idx + name_len
    content_len = len(content_lower)

    # Pattern 1: "according to [the] [name]" — check 25 chars before name
    before_start = max(0, name_idx - 25)
    before = content_lower[before_start:name_idx]
    if "according to " in before:
        return True

    # Pattern 2: [name] ... [attribution word] within 50 chars
    after = content_lower[name_end:min(content_len, name_end + 50)]
    for attr in _ATTRIBUTION_WORDS:
        attr_idx = after.find(attr)
        if attr_idx >= 0 and _has_word_boundary(after, attr_idx, attr_idx + len(attr)):
            return True

    # Pattern 3: [name]'s [possessive indicator] within 40 chars
    poss_window = content_lower[name_end:min(content_len, name_end + 40)]
    for poss in _POSSESSIVE_INDICATORS:
        if poss in poss_window:
            return True

    return False


def detect_claim_subject_quotes(
    content: str,
    interested_parties: list[str],
    party_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    """Detect when interested parties are quoted/cited in evidence.

    When a claim is ABOUT entity X, and the evidence quotes X's statements,
    those are self-serving statements — not independent verification.

    Matching uses three layers:
    1. Exact party name (word-boundary matched)
    2. Programmatic variants (last names from person names)
    3. Wikidata aliases (if provided — e.g., "U.S." for "United States")

    Each match must appear in an attribution context (said, according to, etc.)
    to avoid flagging mere mentions.

    Args:
        content: The evidence text to analyze
        interested_parties: List of interested parties from decomposition's all_parties
        party_aliases: Optional dict mapping party names to Wikidata aliases

    Returns:
        List of interested parties that appear to be quoted/cited in the content
    """
    if not content or not interested_parties:
        return []

    content_lower = content.lower()
    quoted_parties: set[str] = set()

    # Build name → original party mapping with all variants
    match_terms: dict[str, str] = {}

    for party in interested_parties:
        party_lower = party.lower()
        if len(party_lower) >= 3:
            match_terms[party_lower] = party

        # Programmatic variants (last name extraction)
        for variant in _generate_name_variants(party):
            v_lower = variant.lower()
            if len(v_lower) >= 3 and v_lower not in match_terms:
                match_terms[v_lower] = party

        # Wikidata aliases
        if party_aliases and party in party_aliases:
            for alias in party_aliases[party]:
                a_lower = alias.lower()
                if len(a_lower) >= 2 and a_lower not in match_terms:
                    match_terms[a_lower] = party

    # Check each match term against content
    for name_lower, original_party in match_terms.items():
        if original_party in quoted_parties:
            continue

        name_idx = _find_with_boundary(content_lower, name_lower)
        if name_idx == -1:
            continue

        if _check_attribution_near(content_lower, name_idx, len(name_lower)):
            quoted_parties.add(original_party)

    return list(quoted_parties)
