"""Detect when interested parties are quoted or cited in evidence text.

Pure text analysis — word boundary matching and attribution pattern detection.
No external dependencies beyond stdlib.

Used by the judge to flag self-serving statements: when a claim is ABOUT
entity X and the evidence quotes X's statements, those statements cannot
independently verify claims about X.
"""


def detect_claim_subject_quotes(content: str, interested_parties: list[str]) -> list[str]:
    """Detect when interested parties are quoted/cited in evidence.

    When a claim is ABOUT entity X, and the evidence quotes X's statements,
    those are self-serving statements — not independent verification.

    This function detects attribution patterns for the interested parties.
    The interested_parties list should already be expanded by decomposition
    (e.g., ["FBI", "DOJ", "Executive Branch"]) — no hardcoded expansion needed.

    Args:
        content: The evidence text to analyze
        interested_parties: List of interested parties from decomposition's all_parties

    Returns:
        List of interested parties that appear to be quoted/cited in the content
    """
    if not content or not interested_parties:
        return []

    content_lower = content.lower()
    quoted_parties = []

    # Word boundary characters for more precise matching
    # This prevents "FBI" matching in "FIBRIN" or "DOJ" in "DOJO"
    boundary_chars = set(' .,;:!?\'"()[]{}\n\t-/')

    # Attribution patterns — words that indicate something is being quoted/cited
    attribution_words = [
        "said", "says", "stated", "states", "reported", "reports",
        "announced", "announces", "confirmed", "confirms", "denied", "denies",
        "claimed", "claims", "concluded", "concludes", "found", "finds",
        "determined", "issued", "released", "testified", "told",
    ]

    def has_word_boundary(text: str, start: int, end: int) -> bool:
        """Check if a match has word boundaries on both sides."""
        # Start boundary: either at beginning or preceded by boundary char
        start_ok = (start == 0) or (text[start - 1] in boundary_chars)
        # End boundary: either at end or followed by boundary char
        end_ok = (end >= len(text)) or (text[end] in boundary_chars)
        return start_ok and end_ok

    def find_with_boundary(text: str, pattern: str) -> int:
        """Find pattern in text with word boundaries. Returns index or -1."""
        start = 0
        while True:
            idx = text.find(pattern, start)
            if idx == -1:
                return -1
            if has_word_boundary(text, idx, idx + len(pattern)):
                return idx
            start = idx + 1
        return -1

    for party in interested_parties:
        party_lower = party.lower()

        # Skip very short names to avoid false matches
        if len(party_lower) < 3:
            continue

        # Check if party appears in content with word boundaries
        # This prevents "FBI" matching "FIBRIN" or "DOJ" matching "DOJO"
        party_idx = find_with_boundary(content_lower, party_lower)
        if party_idx == -1:
            continue

        # Check for attribution patterns around this party name
        is_quoted = False

        # Pattern 1: "according to [the] X"
        if find_with_boundary(content_lower, f"according to {party_lower}") >= 0 or \
           find_with_boundary(content_lower, f"according to the {party_lower}") >= 0:
            is_quoted = True

        # Pattern 2: "[the] X [attribution_word]" (e.g., "FBI said", "the FBI confirmed")
        if not is_quoted:
            for attr in attribution_words:
                if find_with_boundary(content_lower, f"{party_lower} {attr}") >= 0 or \
                   find_with_boundary(content_lower, f"the {party_lower} {attr}") >= 0:
                    is_quoted = True
                    break

        # Pattern 3: "X's [statement/investigation/conclusion/report/official]"
        if not is_quoted:
            possessives = ["'s statement", "'s investigation", "'s conclusion", "'s report", "'s official", "'s finding", "'s analysis", "'s determination"]
            for poss in possessives:
                if find_with_boundary(content_lower, f"{party_lower}{poss}") >= 0:
                    is_quoted = True
                    break

        # Pattern 4: "X spokesperson/official/director said" (title + attribution)
        if not is_quoted:
            titles = ["spokesperson", "official", "director", "chief", "head", "representative"]
            for title in titles:
                title_pattern = f"{party_lower} {title}"
                idx = find_with_boundary(content_lower, title_pattern)
                if idx >= 0:
                    # Check if followed by attribution word within ~60 chars
                    after = content_lower[idx:idx+60]
                    for attr in attribution_words:
                        if attr in after:
                            is_quoted = True
                            break
                if is_quoted:
                    break

        if is_quoted:
            quoted_parties.append(party)

    return quoted_parties
