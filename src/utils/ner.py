"""Named Entity Recognition via SpaCy.

Loads the en_core_web_sm model once on first use, then extracts PERSON
and ORG entities from text in milliseconds. Used to discover entities
in evidence articles that weren't in the original claim, so we can
Wikidata-expand them and check for conflicts of interest.

SpaCy is NOT an LLM — it's a lightweight, deterministic statistical model
that runs on CPU. No API calls, no cloud, no token costs. The small model
is ~12MB and sits in ~50-100MB of process memory.
"""

from typing import Optional

from src.utils.logging import log, get_logger

MODULE = "ner"
logger = get_logger()

# Lazy-loaded SpaCy model (loaded once on first call)
_nlp = None


def _get_nlp():
    """Load SpaCy model on first use. Cached in module global."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            log.info(logger, MODULE, "model_loaded",
                     "SpaCy model loaded", model="en_core_web_sm")
        except OSError:
            log.warning(logger, MODULE, "model_not_found",
                        "SpaCy model not found — run: python -m spacy download en_core_web_sm",
                        model="en_core_web_sm")
            raise
    return _nlp


def extract_entities(text: str, labels: Optional[set[str]] = None) -> list[dict]:
    """Extract named entities from text.

    Args:
        text: Input text (article content, evidence snippet, etc.)
        labels: Entity labels to extract. Defaults to {"PERSON", "ORG"}.
            Other useful labels: "GPE" (countries), "NORP" (groups),
            "LAW" (legislation), "EVENT" (named events).

    Returns:
        List of {"text": "entity name", "label": "PERSON"|"ORG"|...}
        Deduplicated by (text, label).
    """
    if not text:
        return []

    if labels is None:
        labels = {"PERSON", "ORG"}

    nlp = _get_nlp()

    # SpaCy has a max length limit — truncate long texts
    max_len = nlp.max_length
    if len(text) > max_len:
        text = text[:max_len]

    doc = nlp(text)

    seen = set()
    entities = []
    for ent in doc.ents:
        if ent.label_ not in labels:
            continue

        # Clean up entity text
        name = ent.text.strip()
        if len(name) < 2:
            continue

        key = (name, ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({"text": name, "label": ent.label_})

    return entities


def extract_quoted_entities(content: str) -> list[str]:
    """Extract all PERSON and ORG entities from evidence text via SpaCy NER.

    Unlike detect_claim_subject_quotes() (in quote_detection.py) which checks
    known parties against content, this function discovers NEW entities mentioned
    anywhere in the evidence. These can then be Wikidata-expanded to find
    connections to known interested parties.

    Uses SpaCy's en_core_web_sm model — a deterministic statistical model
    that runs on CPU in milliseconds. Not an LLM, no API calls.

    Returns deduplicated list of entity names (people and organizations).
    """
    if not content:
        return []
    return extract_entity_names(content, labels={"PERSON", "ORG"})


# Labels for claim entity extraction — broad set to catch all factual content
# that should survive decontextualization
CLAIM_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "NORP", "DATE", "CARDINAL",
    "MONEY", "EVENT", "FAC", "PRODUCT", "QUANTITY", "ORDINAL", "LAW",
}


def extract_claim_entities(sentences: list[str]) -> list[str]:
    """Extract named entities from claim group sentences for coverage checking.

    Runs SpaCy NER with a broad label set to catch all factual content:
    people, organizations, places, dates, numbers, nationalities, etc.

    Used by Pass 2 context injection to build a per-group entity checklist.
    After the LLM produces a decontextualized statement, we check which
    entities from the source survived. Missing entities trigger a targeted retry.

    Returns deduplicated list of entity text strings (min 3 chars).
    """
    combined = " ".join(sentences)
    if not combined.strip():
        return []

    entities = extract_entities(combined, labels=CLAIM_ENTITY_LABELS)

    filtered = []
    seen: set[str] = set()
    for e in entities:
        text = e["text"].strip()
        if len(text) < 3:
            continue
        lower = text.lower()
        if lower in seen:
            continue
        seen.add(lower)
        filtered.append(text)

    return filtered


def check_entity_coverage(statement: str, entities: list[str]) -> list[str]:
    """Return entities from the checklist not found in the statement.

    Simple case-insensitive substring matching. Catches obvious drops
    (USS Cole, Beirut, 45,000) while accepting reasonable reformulations
    (Iran → Iran's, Obama → President Obama).
    """
    if not entities:
        return []
    statement_lower = statement.lower()
    return [e for e in entities if e.lower() not in statement_lower]


def extract_entity_names(text: str, labels: Optional[set[str]] = None) -> list[str]:
    """Extract just the entity name strings (no labels).

    Convenience wrapper around extract_entities(). Returns deduplicated
    list of entity names.
    """
    entities = extract_entities(text, labels)
    seen = set()
    names = []
    for e in entities:
        name = e["text"]
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names
