"""Domain logic for claim decomposition.

Full pipeline: normalize → LLM extract → NER augment → Wikidata expand.

The LLM extracts atomic verifiable facts (each with categories and seed
queries for search routing) plus interested parties from the claim. Then
this module expands those parties via Wikidata to discover:
  - Executives (CEO, founder, chairperson)
  - Family connections (spouse, parent, child, sibling) + 2-hop in-laws
  - Parent/subsidiary relationships
  - Affiliated media holdings

This expansion is CRITICAL for self-serving source detection downstream.
When a claim is about a person, we need to know that statements from their
in-laws and relatives are also self-serving. Wikidata provides this mapping
with 2-hop family expansion (e.g., Person A → Spouse → Father-in-law).

Returns a structured dict with:
  - facts: list of {text, categories, seed_queries} — drives seed search routing
  - thesis_info: {thesis, structure, key_test, interested_parties} — the
    interested_parties dict (all_parties, affiliated_media, wikidata_context)
    flows through to research (conflict detection) and judge (annotation)

The Temporal activity wrapper in verify_activities.py calls decompose() here.
"""

from src.llm import invoke_llm, LLMInvocationError, validate_normalize, validate_decompose
from src.prompts.verification import (
    NORMALIZE_SYSTEM, NORMALIZE_USER, DECOMPOSE_SYSTEM, DECOMPOSE_USER,
)
from src.schemas.llm_outputs import NormalizeOutput, DecomposeOutput
from src.prompts.linguistic_patterns import get_linguistic_patterns
from src.tools.wikidata import get_ownership_chain, collect_all_connected_parties
from src.schemas.interested_parties import InterestedPartiesDict
from src.utils.logging import log, get_logger
from src.utils.text_cleanup import cleanup_text

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


async def expand_interested_parties(interested_parties: dict) -> InterestedPartiesDict:
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


def normalize_interested_parties(raw) -> InterestedPartiesDict:
    """Normalize interested_parties to a consistent structure.

    The decomposition now returns interested_parties as an object with:
    - direct: Organizations immediately involved
    - institutional: Parent/governing bodies
    - affiliated_media: News outlets with ownership ties
    - reasoning: Brief explanation of relationships

    For backward compatibility, also handles legacy list format and
    InterestedParties pydantic model objects.

    Returns:
        Dict with keys: all_parties (flat list), direct, institutional,
        affiliated_media, reasoning
    """
    # Handle InterestedParties pydantic model
    if hasattr(raw, 'model_dump'):
        raw = raw.model_dump()

    if isinstance(raw, list):
        # Legacy format: just a flat list of party names
        return {
            "all_parties": raw,
            "direct": raw,
            "institutional": [],
            "affiliated_media": [],
            "reasoning": None,
        }

    if isinstance(raw, dict):
        # New format with categorized parties
        direct = raw.get("direct", [])
        institutional = raw.get("institutional", [])
        affiliated_media = raw.get("affiliated_media", [])
        reasoning = raw.get("reasoning")

        # Combine all for easy lookup
        all_parties = list(set(direct + institutional + affiliated_media))

        return {
            "all_parties": all_parties,
            "direct": direct,
            "institutional": institutional,
            "affiliated_media": affiliated_media,
            "reasoning": reasoning,
        }

    # Fallback for unexpected format
    return {
        "all_parties": [],
        "direct": [],
        "institutional": [],
        "affiliated_media": [],
        "reasoning": None,
    }


async def decompose(claim_text: str) -> dict:
    """Full decompose pipeline: normalize → LLM extract → NER augment → Wikidata expand.

    Pipeline:
    1. LLM normalizes claim (bias neutralization, operationalization, opinion
       separation, coreference resolution, reference grounding, speculation handling)
    2. LLM extracts facts, thesis, and interested parties from normalized claim
    3. Code expands interested parties via Wikidata (programmatic, cached)
       - Corporate: CEO, founder, chairperson, parent/subsidiary
       - Family: spouse, parents, children, siblings + 2-hop in-laws
       - Media: ownership ties to news outlets

    Returns:
        {"facts": [{"text", "categories", "seed_queries"}, ...],
         "thesis_info": {"thesis", "normalized_claim", "structure", ...}}
    """
    log.info(logger, MODULE, "start", "Decomposing claim", claim=claim_text)

    # Step 1: Normalize claim
    norm_output = None
    try:
        norm_output = await invoke_llm(
            system_prompt=NORMALIZE_SYSTEM,
            user_prompt=NORMALIZE_USER.format(claim_text=claim_text),
            schema=NormalizeOutput,
            semantic_validator=validate_normalize,
            max_retries=1,
            temperature=0,
            activity_name="normalize",
        )
        normalized = cleanup_text(norm_output.normalized_claim) or norm_output.normalized_claim
        if norm_output.changes:
            log.info(logger, MODULE, "normalized",
                     "Claim normalized", original=claim_text,
                     normalized=normalized, changes=norm_output.changes)
        else:
            log.info(logger, MODULE, "no_normalization_needed",
                     "No normalization needed", claim=claim_text)
    except LLMInvocationError:
        normalized = claim_text
        log.warning(logger, MODULE, "normalization_failed",
                    "Normalization failed, using raw claim", claim=claim_text)

    # Step 2: Decompose the NORMALIZED claim
    decompose_system_with_patterns = DECOMPOSE_SYSTEM + "\n\n" + get_linguistic_patterns()

    try:
        output = await invoke_llm(
            system_prompt=decompose_system_with_patterns,
            user_prompt=DECOMPOSE_USER.format(claim_text=normalized),
            schema=DecomposeOutput,
            semantic_validator=validate_decompose,
            max_retries=2,
            temperature=0,
            activity_name="decompose",
        )

        # Clean up LLM output text using LanguageTool
        facts = [
            {
                "text": cleanup_text(f.text.strip()) or f.text.strip(),
                "categories": f.categories,
                "seed_queries": f.seed_queries,
            }
            for f in output.facts if f and f.text and f.text.strip()
        ]
        if output.thesis:
            output.thesis = cleanup_text(output.thesis) or output.thesis

        # Normalize interested parties from LLM output
        raw_parties = normalize_interested_parties(output.interested_parties)

        # NER PASS 1 (of 2): Extract entities from the CLAIM TEXT.
        # Catches people/orgs the LLM missed. Pass 2 in judge.py runs on
        # EVIDENCE TEXT — articles that don't exist yet at decompose time.
        from src.utils.ner import extract_entities
        try:
            claim_entities = extract_entities(claim_text, labels={"PERSON", "ORG"})
            existing = {
                p.lower()
                for p in raw_parties.get("direct", []) + raw_parties.get("institutional", [])
            }
            for ent in claim_entities:
                if ent["text"].lower() not in existing and len(ent["text"]) >= 3:
                    if ent["label"] == "PERSON":
                        raw_parties["direct"].append(ent["text"])
                    else:
                        raw_parties["institutional"].append(ent["text"])
                    existing.add(ent["text"].lower())
        except Exception as e:
            log.warning(logger, MODULE, "ner_fallback",
                       "SpaCy NER failed, continuing with LLM parties only",
                       error=str(e))

        # Expand via Wikidata (programmatic — adds executives, family, media)
        expanded_parties = await expand_interested_parties(raw_parties)

        thesis_info = {
            "thesis": output.thesis,
            "normalized_claim": normalized,
            "normalization_changes": norm_output.changes if norm_output else [],
            "structure": output.structure,
            "key_test": output.key_test,
            "interested_parties": expanded_parties,
        }

        log.info(logger, MODULE, "facts_extracted",
                 "Decomposition complete",
                 interested_parties=expanded_parties.get("all_parties"),
                 fact_count=len(facts))

    except LLMInvocationError as e:
        log.warning(logger, MODULE, "invocation_failed",
                    "LLM invocation failed after retries, using original claim as single fact",
                    error=str(e), attempts=e.attempts)
        facts = [{"text": claim_text}]
        thesis_info = {
            "thesis": None,
            "structure": "simple",
            "key_test": None,
            "interested_parties": normalize_interested_parties([]),
        }

    log.info(logger, MODULE, "done", "Claim decomposed",
             claim=claim_text, sub_count=len(facts),
             thesis=thesis_info.get("thesis"),
             structure=thesis_info.get("structure"))
    return {"facts": facts, "thesis_info": thesis_info}
