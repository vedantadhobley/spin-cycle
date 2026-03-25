"""Domain logic for claim decomposition.

Full pipeline: normalize → LLM extract → quality validate → NER augment → Wikidata expand.

The LLM extracts atomic verifiable facts (each with categories and seed
queries for search routing) plus interested parties from the claim. A
post-decompose quality validator then checks the sub-claim list for two
structural issues the LLM can't self-enforce during generation:

  1. Semantic duplicates — logically equivalent sub-claims phrased differently
     (e.g., "No X did Y" ≡ "Y happened for every X")
  2. Group enumeration — individual member checks instead of one group-level
     claim (e.g., 7 G7 country facts instead of 1 group fact)

If issues are found, decompose is retried once with the feedback injected
into the prompt. Then this module expands parties via Wikidata to discover:
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

from datetime import date

from src.llm import invoke_llm, LLMInvocationError, validate_normalize, validate_decompose
from src.prompts.verification import (
    NORMALIZE_SYSTEM, NORMALIZE_USER, DECOMPOSE_SYSTEM, DECOMPOSE_USER,
    build_claim_date_line,
)
from src.schemas.llm_outputs import (
    NormalizeOutput, DecomposeOutput, AtomicFact,
)
from src.tools.wikidata import get_ownership_chain, collect_all_connected_parties, get_entity_aliases
from src.schemas.interested_parties import InterestedPartiesDict
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
    party_aliases: dict[str, list[str]] = {}

    for party in parties_to_expand[:8]:  # Cap to avoid too many queries
        try:
            result = await get_ownership_chain(party)
            if result.get("error"):
                log.debug(logger, MODULE, "not_found",
                          "Entity not found in Wikidata", entity=party)
                continue

            # Fetch aliases for this entity (e.g., "Trump" for "Donald Trump")
            qid = result.get("qid")
            if qid:
                aliases = await get_entity_aliases(qid)
                if aliases:
                    party_aliases[party] = aliases
                    log.debug(logger, MODULE, "aliases_collected",
                              "Collected aliases for party",
                              party=party, aliases=aliases[:5])

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
        "party_aliases": party_aliases,
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


_SUBCLAIM_QUALITY_SYSTEM = """\
You are reviewing a list of sub-claims extracted from a claim.

Check for TWO issues:

1. SEMANTIC DUPLICATES: Are any sub-claims logically equivalent?
   "No X did Y" ≡ "Y happened for every X" (logical negation)
   "X never did Y" ≡ "There is no record of X doing Y" (same assertion)
   These are duplicates even though they use different words.

2. GROUP ENUMERATION: Do sub-claims individually check members of a named \
group (G7, NATO, EU, Fortune 500) when one group-level claim would suffice?
   "France has UHC" + "Germany has UHC" + ... = should be one claim about G7.

Return JSON: {has_duplicates, duplicate_pairs, has_enumeration, \
enumerated_indices, reasoning}"""

_SUBCLAIM_QUALITY_USER = """\
CLAIM: {claim_text}

SUB-CLAIMS:
{numbered_list}"""


def _normalize_text(text: str) -> str:
    """Normalize subclaim text for deduplication comparison."""
    import re
    # Strip trailing/leading punctuation and whitespace, collapse internal whitespace
    t = text.strip().rstrip(".!?,;:")
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def _dedup_facts(facts: list[AtomicFact]) -> tuple[list[AtomicFact], list[str]]:
    """Remove near-duplicate subclaims programmatically before LLM check.

    Catches trivial duplicates (punctuation-only differences, whitespace,
    case) that the LLM quality checker sometimes misses.

    Returns:
        (deduped_facts, issues) — issues describes what was removed.
    """
    seen: dict[str, int] = {}  # normalized text → first index
    keep = []
    issues = []

    for i, f in enumerate(facts):
        norm = _normalize_text(f.text)
        if norm in seen:
            issues.append(
                f"Sub-claims {seen[norm]} and {i} are near-identical "
                f"(\"{facts[seen[norm]].text}\" ≡ \"{f.text}\"). Removed duplicate."
            )
        else:
            seen[norm] = i
            keep.append(f)

    return keep, issues


async def _validate_subclaim_quality(
    facts: list[AtomicFact],
    claim_text: str,
) -> tuple[list[AtomicFact], list[str]]:
    """Check decomposed sub-claims for near-duplicate facts.

    Programmatic check only — catches trivial near-duplicates
    (punctuation, case, whitespace differences). The LLM semantic
    duplicate check was removed due to a 100% false positive rate
    that caused valid decompositions to be collapsed.

    Returns:
        (clean_facts, issues) — clean_facts has trivial dupes removed.
        issues is always empty (no LLM retry triggered).
    """
    if len(facts) < 2:
        return facts, []

    deduped, dedup_issues = _dedup_facts(facts)
    if dedup_issues:
        log.info(logger, MODULE, "programmatic_dedup",
                 "Removed near-duplicate subclaims",
                 removed=len(facts) - len(deduped), issues=dedup_issues)

    return deduped, []


def _validate_decompose_consistency(output: DecomposeOutput) -> None:
    """Log warnings for rubric inconsistencies. Permissive — never rejects."""
    facts = output.facts
    fact_count = len(facts)
    structure = output.structure

    # structure="simple" but >3 facts
    if structure == "simple" and fact_count > 3:
        log.warning(logger, MODULE, "simple_many_facts",
                    f"Structure is 'simple' but {fact_count} facts extracted",
                    structure=structure, fact_count=fact_count)

    # structure="parallel_comparison" but <2 facts
    if structure == "parallel_comparison" and fact_count < 2:
        log.warning(logger, MODULE, "parallel_few_facts",
                    f"Structure is 'parallel_comparison' but only {fact_count} facts",
                    structure=structure, fact_count=fact_count)

    # structure="causal" but no fact has CAUSAL category
    if structure == "causal":
        has_causal = any("CAUSAL" in f.categories for f in facts)
        if not has_causal:
            log.warning(logger, MODULE, "causal_no_category",
                        "Structure is 'causal' but no fact has CAUSAL category",
                        structure=structure)

    # All facts have only GENERAL category
    if facts and all(f.categories == ["GENERAL"] for f in facts):
        log.warning(logger, MODULE, "all_general_categories",
                    "All facts have only GENERAL category — consider more specific categories",
                    fact_count=fact_count)

    # claim_analysis mentions "causal"/"temporal" but structure doesn't match
    if output.claim_analysis:
        analysis_lower = output.claim_analysis.lower()
        if "causal" in analysis_lower and structure not in ("causal",):
            log.warning(logger, MODULE, "analysis_structure_mismatch",
                        f"claim_analysis mentions 'causal' but structure is '{structure}'",
                        structure=structure)
        if "temporal" in analysis_lower and structure not in ("temporal_sequence",):
            log.warning(logger, MODULE, "analysis_structure_mismatch",
                        f"claim_analysis mentions 'temporal' but structure is '{structure}'",
                        structure=structure)


async def decompose(claim_text: str, speaker: str | None = None,
                    claim_date: str | None = None,
                    transcript_title: str | None = None,
                    speaker_description: str = "") -> dict:
    """Full decompose pipeline: normalize → extract → quality validate → NER → Wikidata.

    Pipeline:
    1. LLM normalizes claim (bias neutralization, operationalization, opinion
       separation, coreference resolution, reference grounding, speculation handling)
    2. LLM extracts facts, thesis, and interested parties from normalized claim
    3. Quality validator checks for semantic duplicates and group enumeration
       (LLM call, ~6-8s). If issues found, retries decompose once with feedback.
    4. SpaCy NER augments entity extraction from claim text
    5. If speaker provided, add them as a direct interested party
    6. Code expands interested parties via Wikidata (programmatic, cached)
       - Corporate: CEO, founder, chairperson, parent/subsidiary
       - Family: spouse, parents, children, siblings + 2-hop in-laws
       - Media: ownership ties to news outlets

    Args:
        claim_text: The claim to decompose.
        speaker: Optional name of the person making the claim. Automatically
                 added as a direct interested party and Wikidata-expanded.

    Returns:
        {"facts": [{"text", "categories", "seed_queries"}, ...],
         "thesis_info": {"thesis", "normalized_claim", "structure", ...}}
    """
    log.info(logger, MODULE, "start", "Decomposing claim",
             claim=claim_text, speaker=speaker)

    # Use pre-resolved description from transcript extraction if available,
    # otherwise fall back to Wikidata lookup.
    if not speaker_description and speaker:
        from src.tools.wikidata import get_entity_description
        desc = await get_entity_description(speaker)
        if desc:
            speaker_description = desc
            log.info(logger, MODULE, "speaker_enriched",
                     "Speaker description from Wikidata",
                     speaker=speaker, description=desc)
    elif speaker_description:
        log.info(logger, MODULE, "speaker_pre_enriched",
                 "Using pre-resolved speaker description",
                 speaker=speaker, description=speaker_description)

    # Build context lines for prompts
    if speaker and speaker_description:
        speaker_line = f"\nSpeaker: {speaker} ({speaker_description})"
    elif speaker:
        speaker_line = f"\nSpeaker: {speaker}"
    else:
        speaker_line = ""
    transcript_context = (
        f"\nSource transcript: {transcript_title}" if transcript_title else ""
    )

    # Step 1: Normalize claim
    norm_output = None
    today = date.today().isoformat()
    claim_date_line = build_claim_date_line(claim_date, claim_text)
    try:
        norm_output = await invoke_llm(
            system_prompt=NORMALIZE_SYSTEM.format(
                current_date=today, claim_date_line=claim_date_line),
            user_prompt=NORMALIZE_USER.format(
                claim_text=claim_text, speaker_line=speaker_line,
                transcript_context=transcript_context),
            schema=NormalizeOutput,
            semantic_validator=validate_normalize,
            max_retries=1,
            temperature=0,
            max_tokens=16384,
            activity_name="normalize",
        )
        normalized = norm_output.normalized_claim
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
    decompose_system = DECOMPOSE_SYSTEM.format(
        current_date=today, claim_date_line=claim_date_line)

    try:
        output = await invoke_llm(
            system_prompt=decompose_system,
            user_prompt=DECOMPOSE_USER.format(
                claim_text=normalized, speaker_line=speaker_line,
                transcript_context=transcript_context),
            schema=DecomposeOutput,
            semantic_validator=validate_decompose,
            max_retries=2,
            temperature=0,
            max_tokens=16384,
            activity_name="decompose",
        )

        # Post-hoc rubric consistency check (permissive warnings)
        _validate_decompose_consistency(output)

        # Rubric summary logging
        all_categories = set()
        for f in output.facts:
            all_categories.update(f.categories)
        log.info(logger, MODULE, "rubric_summary",
                 "Decompose rubric complete",
                 claim_analysis=output.claim_analysis[:100] if output.claim_analysis else "",
                 structure=output.structure,
                 fact_count=len(output.facts),
                 categories=sorted(all_categories))

        # Post-validation: check for semantic duplicates / group enumeration
        # Returns deduped facts (trivial dupes removed) + LLM-detected issues
        clean_facts = output.facts
        if len(output.facts) >= 2:
            clean_facts, llm_issues = await _validate_subclaim_quality(
                output.facts, normalized,
            )
            if llm_issues:
                # LLM found semantic dupes or group enumeration — retry decompose
                feedback = "\n".join(f"- {issue}" for issue in llm_issues)
                retry_prompt = (
                    DECOMPOSE_USER.format(
                        claim_text=normalized, speaker_line=speaker_line,
                        transcript_context=transcript_context)
                    + f"\n\nYOUR PREVIOUS OUTPUT HAD STRUCTURAL ISSUES:\n{feedback}\n"
                    "Fix these issues in your new output."
                )
                try:
                    output = await invoke_llm(
                        system_prompt=decompose_system,
                        user_prompt=retry_prompt,
                        schema=DecomposeOutput,
                        semantic_validator=validate_decompose,
                        max_retries=1,
                        temperature=0.1,
                        max_tokens=16384,
                        activity_name="decompose_retry",
                    )
                    clean_facts = output.facts
                    log.info(logger, MODULE, "decompose_retry_success",
                             "Decompose retry succeeded after quality feedback",
                             fact_count=len(output.facts))
                except LLMInvocationError:
                    log.warning(logger, MODULE, "decompose_retry_failed",
                                "Decompose retry failed, using deduped output")

        facts = [
            {
                "text": f.text.strip(),
                "verification_target": f.verification_target.strip() if f.verification_target else "",
                "categories": f.categories,
                "seed_queries": f.seed_queries,
                "category_rationale": f.category_rationale,
            }
            for f in clean_facts if f and f.text and f.text.strip()
        ]

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

        # Add speaker as direct interested party (if provided and not already present)
        if speaker:
            existing_lower = {
                p.lower()
                for p in raw_parties.get("direct", []) + raw_parties.get("institutional", [])
            }
            if speaker.lower() not in existing_lower:
                raw_parties["direct"].append(speaker)
                log.info(logger, MODULE, "speaker_added",
                         "Speaker added as interested party",
                         speaker=speaker)

        # Expand via Wikidata (programmatic — adds executives, family, media)
        expanded_parties = await expand_interested_parties(raw_parties)

        thesis_info = {
            "thesis": output.thesis,
            "normalized_claim": normalized,
            "normalization_changes": norm_output.changes if norm_output else [],
            "structure": output.structure,
            "key_test": output.key_test,
            "claim_analysis": output.claim_analysis,
            "structure_justification": output.structure_justification,
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
    return {
        "facts": facts,
        "thesis_info": thesis_info,
        "speaker_description": speaker_description,
    }
