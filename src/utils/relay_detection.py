"""Detect when evidence relays an interested party's authority position.

Unlike quote_detection.py which catches direct attribution ("X said",
"according to X"), this module detects POSITION RELAYING — when evidence
derives its factual basis from a document, designation, or determination
created by an interested party.

Uses SpaCy dependency parsing (en_core_web_sm) to analyze grammatical
structure rather than substring matching. Key distinction: "The State
Department designated Iran" fires (interested party is grammatical agent
of authority verb), but "Researchers found that Iran funds terrorism"
does not (researchers are not an interested party).

Three detection layers:
1. Authority-agent: interested party is subject of designation/classification verb
2. Document-attribution: evidence cites a document authored by interested party
3. Reaffirmation: evidence describes reaffirming/endorsing an interested party's position
"""

from src.utils.logging import log, get_logger

MODULE = "relay_detection"
logger = get_logger()

# Verbs indicating an authority making a determination/designation.
# Matched by LEMMA (not surface form) — SpaCy handles conjugation.
_AUTHORITY_VERB_LEMMAS = frozenset({
    "designate", "classify", "determine", "declare", "list",
    "conclude", "confirm", "affirm", "certify", "assess",
    "rule", "decide", "proclaim", "deem", "label",
    "sanction", "name", "identify",
})

# Verbs indicating relaying/endorsing another entity's position.
_RELAY_VERB_LEMMAS = frozenset({
    "reaffirm", "endorse", "reiterate", "echo", "uphold",
    "maintain", "adopt", "recognize", "acknowledge",
    "reauthorize", "renew",
})

# Nouns that act as proxies for an authority (e.g., "a State Department report").
_DOCUMENT_NOUNS = frozenset({
    "report", "study", "assessment", "analysis", "review",
    "determination", "designation", "finding", "conclusion",
    "memo", "brief", "document", "resolution", "list",
    "data", "survey", "investigation", "statement", "release",
    "bulletin", "advisory", "order", "directive", "act",
})

# Nouns that represent formal designations/classifications (dobj of relay verbs).
_DESIGNATION_NOUNS = frozenset({
    "designation", "classification", "determination", "status",
    "listing", "label", "sanction", "finding", "ruling",
    "decision", "position", "assessment", "conclusion",
})

# Maximum chars to parse per evidence item (relay patterns appear in
# headlines and ledes, not deep in articles). Caps SpaCy time at ~50ms.
_MAX_PARSE_CHARS = 3000

# Name stopwords — never useful as standalone match terms
_NAME_STOPWORDS = frozenset({
    "the", "of", "and", "for", "in", "on", "at", "to", "a", "an",
    "by", "with", "from",
})


def _build_match_set(
    interested_parties: list[str],
    party_aliases: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Build lowercased match terms → original party name mapping.

    Same approach as quote_detection.py for consistency.
    """
    match_terms: dict[str, str] = {}

    for party in interested_parties:
        party_lower = party.lower()
        if len(party_lower) >= 3:
            match_terms[party_lower] = party

        # Programmatic last-name variants
        words = party.split()
        if len(words) >= 2:
            significant = [w for w in words
                           if w.lower() not in _NAME_STOPWORDS and len(w) > 2]
            if len(significant) >= 2 and significant[-1][0:1].isupper():
                last = significant[-1].lower()
                if last not in match_terms:
                    match_terms[last] = party

        # Wikidata aliases
        if party_aliases and party in party_aliases:
            for alias in party_aliases[party]:
                a_lower = alias.lower()
                if len(a_lower) >= 2 and a_lower not in match_terms:
                    match_terms[a_lower] = party

    return match_terms


def _span_text(token) -> str:
    """Get the full noun phrase text for a token by walking its compound
    and flat children leftward. This reconstructs multi-word names that
    SpaCy splits into individual tokens.

    "Trump administration" → compound(Trump) + head(administration)
    "State Department" → compound(State) + head(Department)
    """
    # Collect left compound/flat children + the token itself
    parts = []
    for child in token.lefts:
        if child.dep_ in ("compound", "flat", "amod"):
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)


def _token_matches_party(token, match_terms: dict[str, str]) -> str | None:
    """Check if a token (or its compound span, or its NER span) matches
    an interested party. Returns the original party name or None.

    Uses three strategies:
    1. Token text (single word like "Trump")
    2. Compound span ("Trump administration", "State Department")
    3. NER entity span (SpaCy's entity recognition)
    """
    # Strategy 1: Single token
    token_lower = token.text.lower()
    if token_lower in match_terms:
        return match_terms[token_lower]

    # Strategy 2: Compound noun phrase
    span = _span_text(token).lower()
    if span != token_lower and span in match_terms:
        return match_terms[span]

    # Strategy 3: NER entity span
    if token.ent_type_ in ("ORG", "PERSON", "GPE"):
        # Walk to the full entity span
        ent_text = token.text
        for ent in token.doc.ents:
            if ent.start <= token.i < ent.end:
                ent_text = ent.text
                break
        ent_lower = ent_text.lower()
        if ent_lower in match_terms:
            return match_terms[ent_lower]

    return None


def _find_subject_party(verb_token, match_terms: dict[str, str]) -> str | None:
    """Find the grammatical subject of a verb and check if it matches
    an interested party. Handles both active and passive voice.

    Active: "The State Department designated Iran..."
            nsubj(designated) → State Department → match
    Passive: "Iran was designated by the State Department..."
             nsubjpass(designated), pobj(by) → State Department → match
    """
    for child in verb_token.children:
        # Active voice: direct subject
        if child.dep_ in ("nsubj", "nsubjpass"):
            party = _token_matches_party(child, match_terms)
            if party:
                return party

            # Subject might be a document noun with a party possessor
            # "A State Department report confirmed..."
            if child.lemma_.lower() in _DOCUMENT_NOUNS:
                doc_party = _find_document_author(child, match_terms)
                if doc_party:
                    return doc_party

        # Passive voice: agent in "by" prepositional phrase
        if child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    party = _token_matches_party(pobj, match_terms)
                    if party:
                        return party

    return None


def _find_document_author(
    doc_token, match_terms: dict[str, str]
) -> str | None:
    """For a document noun (report, assessment, etc.), find its author
    via possessive or compound modifiers.

    "State Department's report" → poss(State Department) → match
    "Trump administration report" → compound(Trump administration) → match
    "a report by the Pentagon" → prep(by) → pobj(Pentagon) → match
    """
    for child in doc_token.children:
        # Possessive: "State Department's report"
        if child.dep_ == "poss":
            party = _token_matches_party(child, match_terms)
            if party:
                return party

        # Compound: "Trump administration report"
        if child.dep_ in ("compound", "amod"):
            party = _token_matches_party(child, match_terms)
            if party:
                return party

        # Prepositional: "report by the Pentagon"
        if child.dep_ == "prep" and child.text.lower() in ("by", "from", "of"):
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    party = _token_matches_party(pobj, match_terms)
                    if party:
                        return party

    return None


def _detect_authority_agent(sent, match_terms: dict[str, str]) -> dict | None:
    """Layer 1: Detect when an interested party is the grammatical agent
    of an authority/designation verb.

    Fires for: "The State Department designated Iran as..."
    Does NOT fire for: "Researchers found that Iran funds terrorism"
    """
    for token in sent:
        if token.pos_ != "VERB":
            continue
        if token.lemma_.lower() not in _AUTHORITY_VERB_LEMMAS:
            continue

        party = _find_subject_party(token, match_terms)
        if party:
            return {
                "party": party,
                "relay_type": "authority_agent",
                "verb": token.text,
                "sentence": sent.text.strip()[:200],
            }

    return None


def _detect_document_attribution(sent, match_terms: dict[str, str]) -> dict | None:
    """Layer 2: Detect 'according to [party's document]' and 'citing [party] data'.

    Fires for: "according to the State Department's annual report"
    Fires for: "citing State Department data"
    Does NOT fire for: "according to researchers" (no document noun)
    """
    for token in sent:
        # "according to X" pattern
        if token.text.lower() == "according" and token.dep_ == "prep":
            for child in token.children:
                if child.text.lower() == "to":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            # Direct party reference is caught by quote_detection
                            # We catch document references here
                            if pobj.lemma_.lower() in _DOCUMENT_NOUNS:
                                doc_party = _find_document_author(
                                    pobj, match_terms
                                )
                                if doc_party:
                                    return {
                                        "party": doc_party,
                                        "relay_type": "document_attribution",
                                        "verb": "according to",
                                        "sentence": sent.text.strip()[:200],
                                    }

        # "citing X data/report" pattern
        if token.lemma_.lower() in ("cite", "reference", "invoke") and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("dobj", "pobj"):
                    if child.lemma_.lower() in _DOCUMENT_NOUNS:
                        doc_party = _find_document_author(child, match_terms)
                        if doc_party:
                            return {
                                "party": doc_party,
                                "relay_type": "document_attribution",
                                "verb": token.text,
                                "sentence": sent.text.strip()[:200],
                            }
                    # "citing State Department data" — party is compound of document
                    party = _token_matches_party(child, match_terms)
                    if party:
                        return {
                            "party": party,
                            "relay_type": "document_attribution",
                            "verb": token.text,
                            "sentence": sent.text.strip()[:200],
                        }

    return None


def _detect_reaffirmation(sent, match_terms: dict[str, str]) -> dict | None:
    """Layer 3: Detect reaffirmation/endorsement of an interested party's position.

    Fires for: "Congress passed a resolution reaffirming the designation"
    Fires for: "The act renewed the sanctions listing"
    Does NOT fire for: "The study confirmed the findings" (no designation noun)
    """
    for token in sent:
        if token.pos_ != "VERB":
            continue
        if token.lemma_.lower() not in _RELAY_VERB_LEMMAS:
            continue

        # Check if the object of the relay verb is a designation/classification noun
        for child in token.children:
            if child.dep_ in ("dobj", "attr", "oprd"):
                if child.lemma_.lower() in _DESIGNATION_NOUNS:
                    # The designation itself might name the interested party
                    doc_party = _find_document_author(child, match_terms)
                    if doc_party:
                        return {
                            "party": doc_party,
                            "relay_type": "reaffirmation",
                            "verb": token.text,
                            "sentence": sent.text.strip()[:200],
                        }

                    # Or the subject of the relay verb might be a party
                    # (less common but still circular)
                    subject_party = _find_subject_party(token, match_terms)
                    if subject_party:
                        return {
                            "party": subject_party,
                            "relay_type": "reaffirmation",
                            "verb": token.text,
                            "sentence": sent.text.strip()[:200],
                        }

    return None


def detect_authority_relay(
    content: str,
    interested_parties: list[str],
    party_aliases: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Detect when evidence relays an interested party's authority position.

    Uses SpaCy dependency parsing — NOT substring matching. Each detection
    requires the interested party to be the grammatical agent of a
    designation verb, the author of a cited document, or the originator
    of a reaffirmed position.

    Args:
        content: Evidence text to analyze (truncated to first 3000 chars).
        interested_parties: List of party names from all_parties.
        party_aliases: Optional Wikidata aliases (same format as quote_detection).

    Returns:
        List of relay detections, each a dict with:
          party: str — which interested party's position is being relayed
          relay_type: str — "authority_agent" | "document_attribution" | "reaffirmation"
          verb: str — the trigger verb/phrase
          sentence: str — the sentence where relay was detected (truncated)
    """
    if not content or not interested_parties:
        return []

    from src.utils.ner import _get_nlp

    try:
        nlp = _get_nlp()
    except Exception:
        log.warning(logger, MODULE, "spacy_unavailable",
                    "SpaCy not available for relay detection")
        return []

    match_terms = _build_match_set(interested_parties, party_aliases)
    if not match_terms:
        return []

    # Parse first N chars — relay patterns are in headlines/ledes
    text = content[:_MAX_PARSE_CHARS]
    try:
        doc = nlp(text)
    except Exception as e:
        log.debug(logger, MODULE, "parse_error",
                  "SpaCy parse failed", error=str(e))
        return []

    detections: list[dict] = []
    seen_parties: set[str] = set()

    for sent in doc.sents:
        # Try each layer in order. Short-circuit per sentence:
        # one detection per sentence is enough.
        for detector in (
            _detect_authority_agent,
            _detect_document_attribution,
            _detect_reaffirmation,
        ):
            result = detector(sent, match_terms)
            if result and result["party"] not in seen_parties:
                detections.append(result)
                seen_parties.add(result["party"])
                break  # Next sentence

    if detections:
        log.info(logger, MODULE, "relay_detected",
                 "Authority relay patterns found",
                 count=len(detections),
                 parties=[d["party"] for d in detections],
                 types=[d["relay_type"] for d in detections])

    return detections


def analyze_relay_in_evidence(
    evidence_items: list[dict],
    interested_parties: list[str],
    party_aliases: dict[str, list[str]] | None = None,
) -> dict:
    """Analyze a batch of evidence items for relay patterns.

    Returns a summary dict:
      total: int — total items analyzed
      relay_count: int — items with relay detections
      relay_pct: int — percentage (0-100)
      relay_parties: list[str] — unique parties whose positions are relayed
      detections: list[dict] — all individual detections
    """
    if not evidence_items or not interested_parties:
        return {
            "total": 0, "relay_count": 0, "relay_pct": 0,
            "relay_parties": [], "detections": [],
        }

    all_detections: list[dict] = []
    relay_count = 0
    relay_parties: set[str] = set()

    for ev in evidence_items:
        content = ev.get("content", "")
        if not content:
            continue
        detections = detect_authority_relay(
            content, interested_parties, party_aliases,
        )
        if detections:
            relay_count += 1
            for d in detections:
                relay_parties.add(d["party"])
                all_detections.append(d)

    total = len(evidence_items)
    return {
        "total": total,
        "relay_count": relay_count,
        "relay_pct": int(100 * relay_count / total) if total > 0 else 0,
        "relay_parties": sorted(relay_parties),
        "detections": all_detections,
    }
