"""Programmatic entity expansion for claim decomposition.

After the LLM extracts facts and interested parties from a claim,
this module expands those parties via Wikidata to discover:
  - Executives (CEO, founder, chairperson)
  - Family connections (spouse, parent, child, sibling) + 2-hop in-laws
  - Parent/subsidiary relationships
  - Affiliated media holdings

This expansion is CRITICAL for self-serving source detection downstream.
When a claim is about a person, we need to know that statements from their
in-laws and relatives are also self-serving. Wikidata provides this mapping
with 2-hop family expansion (e.g., Person A → Spouse → Father-in-law).

## Why programmatic, not agentic

Previously this was a ReAct agent that MIGHT call wikidata_lookup, with a
programmatic fallback that guaranteed expansion anyway. The agent loop added
latency and LLM calls for no benefit. Now it's just: LLM extracts entities
→ code calls Wikidata. Deterministic, cached, no wasted tool calls.
"""

from src.tools.wikidata import get_ownership_chain, collect_all_connected_parties
from src.utils.logging import log, get_logger

MODULE = "decompose"
logger = get_logger()

# Corporate entity keywords — if party name contains these, try Wikidata
_CORP_INDICATORS = {
    "inc", "corp", "llc", "ltd", "company", "technologies", "tech",
    "labs", "ai", "group", "holdings", "media", "news", "foundation",
}

def _should_expand(name: str) -> bool:
    """Check if a party name is worth querying Wikidata for.

    Returns True for:
    - Names with corporate indicators
    - People's names (2+ words, likely a person)
    - Single-word proper nouns (likely orgs)

    Returns False for:
    - Abstract concepts ("the economy", "democracy")
    - Very short names (< 3 chars)
    """
    name_lower = name.lower().strip()

    if len(name_lower) < 3:
        return False

    # Corporate indicators
    for indicator in _CORP_INDICATORS:
        if indicator in name_lower:
            return True

    # People's names: 2+ words, starts with uppercase
    # (e.g., "Jane Smith", "John Doe")
    words = name.split()
    if len(words) >= 2 and words[0][0:1].isupper():
        return True

    # Single-word proper nouns that might be orgs (e.g., "Reuters", "FBI")
    if len(words) == 1 and name[0:1].isupper() and len(name) >= 3:
        return True

    return False


async def expand_interested_parties(interested_parties: dict) -> dict:
    """Expand interested parties via Wikidata.

    Takes the raw interested_parties dict from decompose output and
    enriches it with Wikidata data: executives, family, media holdings.

    Args:
        interested_parties: Dict with keys: direct, institutional,
            affiliated_media, reasoning

    Returns:
        Updated dict with:
        - all_parties: flat list of ALL connected entities
        - affiliated_media: expanded with discovered media holdings
        - wikidata_context: formatted text for injection into prompts
    """
    direct = interested_parties.get("direct", [])
    institutional = interested_parties.get("institutional", [])
    affiliated_media = interested_parties.get("affiliated_media", [])

    # Collect all parties to potentially expand
    all_input = list(set(direct + institutional))
    parties_to_expand = [p for p in all_input if _should_expand(p)]

    if not parties_to_expand:
        log.debug(logger, MODULE, "no_expansion", "No parties to expand",
                  parties=all_input)
        all_parties = list(set(direct + institutional + affiliated_media))
        return {
            **interested_parties,
            "all_parties": all_parties,
            "wikidata_context": "",
        }

    log.info(logger, MODULE, "expansion_start", "Expanding parties via Wikidata",
             parties=parties_to_expand)

    expanded_people = set(direct + institutional)
    expanded_media = set(affiliated_media)
    context_lines = []

    for party in parties_to_expand[:8]:  # Cap to avoid too many queries
        try:
            result = await get_ownership_chain(party)
            if result.get("error"):
                log.debug(logger, MODULE, "not_found",
                          "Entity not found in Wikidata", entity=party)
                continue

            connected = collect_all_connected_parties(result)

            # Add discovered people
            for person in connected["people"]:
                expanded_people.add(person)

            # Add discovered media
            for media in connected["media"]:
                expanded_media.add(media)

            # Add discovered orgs
            for org in connected["orgs"]:
                expanded_people.add(org)  # Goes into all_parties for detection

            # Build context line for this entity
            context_parts = []
            if connected["people"]:
                context_parts.append(f"connected people: {', '.join(connected['people'][:10])}")
            if connected["media"]:
                context_parts.append(f"media holdings: {', '.join(connected['media'])}")
            if connected["orgs"]:
                context_parts.append(f"affiliated orgs: {', '.join(connected['orgs'][:5])}")

            if context_parts:
                context_lines.append(f"  - {party}: {'; '.join(context_parts)}")

            log.info(logger, MODULE, "expanded", "Entity expanded",
                     entity=party,
                     people_added=len(connected["people"]),
                     media_added=len(connected["media"]))

        except Exception as e:
            log.warning(logger, MODULE, "expansion_error",
                        "Wikidata expansion failed for entity",
                        entity=party, error=str(e))
            continue

    all_parties = list(expanded_people | expanded_media)

    # Build context text for injection into research/judge prompts
    wikidata_context = ""
    if context_lines:
        wikidata_context = (
            "INTERESTED PARTY CONNECTIONS (via Wikidata):\n"
            + "\n".join(context_lines)
            + "\n\nStatements from ANY of these connected entities are "
            "self-serving and cannot independently verify claims about them."
        )

    log.info(logger, MODULE, "expansion_done", "Wikidata expansion complete",
             original_count=len(all_input),
             expanded_count=len(all_parties),
             media_count=len(expanded_media))

    return {
        "direct": direct,
        "institutional": institutional,
        "affiliated_media": list(expanded_media),
        "reasoning": interested_parties.get("reasoning"),
        "all_parties": all_parties,
        "wikidata_context": wikidata_context,
    }
