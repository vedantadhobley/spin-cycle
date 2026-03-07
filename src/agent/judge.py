"""Domain logic for evidence evaluation (judging sub-claims).

Pipeline: filter → rank → enrich → annotate → LLM judge.

This module contains the judgment logic extracted from verify_activities.py.
The Temporal activity wrapper in verify_activities.py calls judge() here.
"""

from datetime import date
from urllib.parse import urlparse

from src.llm import invoke_llm, LLMInvocationError, validate_judge
from src.prompts.verification import JUDGE_SYSTEM, JUDGE_USER
from src.schemas.llm_outputs import JudgeOutput
from src.db.session import get_sync_session
from src.db.models import SourceRating
from src.tools.source_ratings import get_source_rating_sync
from src.tools.media_matching import url_matches_media, check_publisher_ownership
from src.tools.wikidata import get_ownership_chain, collect_all_connected_parties
from src.utils.logging import log, get_logger
from src.utils.quote_detection import detect_claim_subject_quotes
from src.utils.ner import extract_quoted_entities
from src.schemas.interested_parties import InterestedPartiesDict
from src.utils.text_cleanup import cleanup_text

MODULE = "judge"
logger = get_logger()
MAX_JUDGE_EVIDENCE = 20


def _format_source_tag(rating: dict | None) -> str:
    """Format a rating as a readable tag for evidence annotation.

    Examples:
        "[Center (-0.5) | Very High factual | UK Wire Service]"
        "[Right (+3.0) | Mixed factual | US Corporate TV]"
        "[Right | Very Low factual | Russia State-Funded] ⚠️"
        "[USA Government | FBI] ⚠️ INTERESTED PARTY"
        "[Right (+7.9) | Mixed factual | USA Government] ⚠️ INTERESTED PARTY"
    """
    if not rating:
        return "[Unrated source]"

    is_gov = rating.get("is_government", False)
    has_mbfc = rating.get("bias") or rating.get("factual_reporting")

    parts = []
    warnings = []

    # Add MBFC bias/factual data if present
    if rating.get("bias"):
        bias_display = rating["bias"].replace("-", " ").title()
        if rating.get("bias_score") is not None:
            score = rating["bias_score"]
            sign = "+" if score > 0 else ""
            bias_display += f" ({sign}{score})"
        parts.append(bias_display)

    if rating.get("factual_reporting"):
        factual_display = rating["factual_reporting"].replace("-", " ").title()
        if "factual" not in factual_display.lower():
            factual_display += " factual"
        parts.append(factual_display)
        if rating["factual_reporting"] in ("low", "very-low"):
            warnings.append("low-factual")

    # Country + Media Type / Agency info
    if is_gov:
        country = rating.get("country", "")
        agency = rating.get("agency_name", "")
        if has_mbfc:
            if country:
                parts.append(f"{country} Government")
            else:
                parts.append("Government")
        else:
            if country:
                parts.append(f"{country} Government")
            else:
                parts.append("Government")
            if agency:
                parts.append(agency)
    else:
        location_type = []
        if rating.get("country"):
            country_abbrev = {
                "United States": "US", "United Kingdom": "UK", "Russia": "Russia",
                "China": "China", "Germany": "Germany", "France": "France", "USA": "USA",
            }
            country = country_abbrev.get(rating["country"], rating["country"])
            location_type.append(country)
        if rating.get("media_type"):
            location_type.append(rating["media_type"])
        if location_type:
            parts.append(" ".join(location_type))

        # Check for state-funded (ownership contains state indicators)
        if rating.get("ownership"):
            ownership_lower = rating["ownership"].lower()
            if any(x in ownership_lower for x in ["state", "government", "kremlin", "chinese communist", "state-funded"]):
                warnings.append("state-funded")

    # Build final tag
    if not parts:
        return "[Unrated source]"

    tag = f"[{' | '.join(parts)}]"

    # Add warnings
    if is_gov:
        tag += " ⚠️ INTERESTED PARTY"
    elif warnings:
        tag += " ⚠️"

    return tag


def _find_rating_by_name(name: str) -> dict | None:
    """Find a source rating by outlet name (reverse lookup).

    Searches the source_ratings table for domains matching the outlet name.
    "Outlet Name" → finds "outletname.com" via normalized substring matching.
    """
    if not name or len(name) < 3:
        return None

    # Normalize: strip "the", remove spaces, lowercase
    normalized = name.lower().replace("the ", "").replace(" ", "")

    try:
        with get_sync_session() as session:
            from sqlalchemy import select
            stmt = select(SourceRating).where(
                SourceRating.domain.ilike(f"%{normalized}%")
            ).limit(1)
            cached = session.execute(stmt).scalar_one_or_none()

            if cached:
                return {
                    "domain": cached.domain,
                    "bias": cached.bias,
                    "bias_score": cached.bias_score,
                    "factual_reporting": cached.factual_reporting,
                    "credibility": cached.credibility,
                    "country": cached.country,
                    "media_type": cached.media_type,
                    "ownership": cached.ownership,
                    "traffic": cached.traffic,
                    "mbfc_url": cached.mbfc_url,
                }
    except Exception as e:
        log.warning(logger, MODULE, "reverse_lookup_failed",
                    "Rating reverse lookup failed", name=name, error=str(e))

    return None


async def judge(
    claim_text: str,
    sub_claim: str,
    evidence: list[dict],
    interested_parties: InterestedPartiesDict,
) -> dict:
    """Evaluate evidence and return a verdict.

    Args:
        claim_text: Original full claim (context for interpretation).
        sub_claim: Atomic fact being judged.
        evidence: Evidence dicts from research phase.
        interested_parties: Pre-expanded dict (all_parties, affiliated_media, etc).

    Returns:
        Dict: sub_claim, verdict, confidence, reasoning, evidence.
    """
    all_parties = interested_parties.get("all_parties", [])
    affiliated_media = interested_parties.get("affiliated_media", [])

    log.info(logger, MODULE, "start", "Judging sub-claim",
             sub_claim=sub_claim, evidence_count=len(evidence),
             interested_parties=interested_parties)

    # No evidence → unverifiable (no point asking the LLM)
    if not evidence:
        log.info(logger, MODULE, "no_evidence",
                 "No evidence found, returning unverifiable",
                 sub_claim=sub_claim)
        return {
            "sub_claim": sub_claim,
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reasoning": "No evidence was found for this claim.",
            "evidence": [],
        }

    # Rank and cap evidence
    source_evidence = _rank_evidence(evidence)

    # Log which URLs the judge will see — critical for debugging verdicts
    evidence_urls = [
        ev.get("source_url") or "N/A"
        for ev in source_evidence
    ]
    source_types = {}
    for ev in source_evidence:
        st = ev.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1
    log.debug(logger, MODULE, "evidence_summary",
              "Evidence prepared for judge",
              sub_claim=sub_claim,
              evidence_count=len(source_evidence),
              source_types=source_types,
              urls=evidence_urls)

    # Pre-judge enrichment: entities + source publishers
    all_parties, affiliated_media = await _enrich_parties_from_evidence(
        source_evidence, all_parties, affiliated_media,
    )

    # Format evidence for LLM prompt with annotations
    evidence_text = _annotate_evidence(
        source_evidence, all_parties, affiliated_media, interested_parties,
    )

    # Invoke the judge LLM
    try:
        output = await invoke_llm(
            system_prompt=JUDGE_SYSTEM.format(
                current_date=date.today().isoformat(),
            ),
            user_prompt=JUDGE_USER.format(
                claim_text=claim_text,
                sub_claim=sub_claim,
                evidence_text=evidence_text,
            ),
            schema=JudgeOutput,
            semantic_validator=validate_judge,
            max_retries=2,
            temperature=0,
            activity_name="judge",
        )

        verdict = output.verdict
        confidence = output.confidence
        reasoning = output.reasoning

    except LLMInvocationError as e:
        log.warning(logger, MODULE, "invocation_failed",
                    "LLM invocation failed after retries",
                    error=str(e), attempts=e.attempts)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to parse LLM judgment after {e.attempts} attempts"

    # Clean up reasoning text using LanguageTool
    reasoning = cleanup_text(reasoning)

    log.info(logger, MODULE, "done", "Sub-claim judged",
             sub_claim=sub_claim, verdict=verdict, confidence=confidence)

    return {
        "sub_claim": sub_claim,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": evidence,
    }


def _rank_evidence(source_evidence: list[dict]) -> list[dict]:
    """Rank evidence by quality and cap to MAX_JUDGE_EVIDENCE.

    Ranked by quality (MBFC, source type, TLD, content richness) rather
    than discovery order — prevents high-quality late-discovered sources
    from being dropped.
    """
    if len(source_evidence) > MAX_JUDGE_EVIDENCE:
        from src.utils.evidence_ranker import rank_and_select, format_ranking_log
        original_count = len(source_evidence)
        source_evidence, dropped = rank_and_select(
            source_evidence, max_items=MAX_JUDGE_EVIDENCE,
        )
        log.info(logger, MODULE, "evidence_ranked",
                 "Ranked and selected evidence for judge prompt",
                 original=original_count, selected=len(source_evidence))
        log.debug(logger, MODULE, "evidence_ranking_detail",
                  "Evidence ranking breakdown",
                  **format_ranking_log(source_evidence, dropped))
    return source_evidence


async def _enrich_parties_from_evidence(
    source_evidence: list[dict],
    all_parties: list[str],
    affiliated_media: list[str],
) -> tuple[list[str], list[str]]:
    """Lightweight Wikidata enrichment from evidence content (judge-time cleanup).

    SpaCy NER extracts people/orgs from evidence text → Wikidata expand in parallel.
    Only adds entities whose Wikidata graph overlaps with existing parties.

    Note: The heavy lifting (MBFC ownership → Wikidata, evidence NER → Wikidata)
    now happens in the research phase. This is a lightweight cleanup pass that
    catches any entities the research phase missed (e.g., from page fetches
    that weren't in the seed evidence).

    Returns new lists — does NOT mutate the input lists.
    """
    import asyncio

    enriched_parties = list(all_parties)
    enriched_media = list(affiliated_media)

    async def _enrich_entity(entity: str) -> None:
        """Wikidata-expand an entity and add connections to interested parties."""
        try:
            result = await get_ownership_chain(entity)
            if result.get("error"):
                return

            connected = collect_all_connected_parties(result)
            all_connected = (
                set(connected["people"])
                | set(connected["orgs"])
                | set(connected["media"])
            )
            overlap = all_connected & set(enriched_parties)

            if overlap:
                enriched_parties.append(entity)
                for person in connected["people"]:
                    if person not in enriched_parties:
                        enriched_parties.append(person)
                for media in connected["media"]:
                    if media not in enriched_media:
                        enriched_media.append(media)

                log.info(logger, MODULE, "enrichment_found",
                         "Evidence entity connects to interested party",
                         entity=entity, overlap=list(overlap)[:5])
        except Exception as e:
            log.warning(logger, MODULE, "enrichment_error",
                       "Entity enrichment failed",
                       entity=entity, error=str(e))

    # NER PASS 2 (of 2): Extract entities from EVIDENCE TEXT (articles).
    # Distinct from Pass 1 in decompose.py which runs on claim text.
    # Evidence mentions people/orgs not in the original claim — those
    # may connect to known interested parties via Wikidata.
    all_content = " ".join(ev.get("content", "") for ev in source_evidence)
    discovered_entities = extract_quoted_entities(all_content)

    all_parties_lower = {p.lower() for p in enriched_parties}
    new_entities = [
        e for e in discovered_entities
        if e.lower() not in all_parties_lower
    ]

    if new_entities:
        log.debug(logger, MODULE, "ner_entities",
                  "SpaCy NER extracted new entities from evidence",
                  new_entities=new_entities[:15])

        # Parallel Wikidata expansion (was sequential before)
        await asyncio.gather(
            *[_enrich_entity(entity) for entity in new_entities[:8]]
        )

    return enriched_parties, enriched_media


def _annotate_evidence(
    source_evidence: list[dict],
    all_parties: list[str],
    affiliated_media: list[str],
    interested_parties: dict,
) -> str:
    """Format evidence with MBFC ratings, interest checks, and bias tracking.

    Four interest checks per evidence item:
    1. Affiliated media URL match
    2. Interested party quoted in content
    3. Publisher owned by interested party
    4. Sub-source references with poor ratings

    Returns formatted evidence text for injection into the judge prompt.
    """
    evidence_parts = []
    bias_distribution = {"left": 0, "center": 0, "right": 0, "unrated": 0}
    interested_party_count = 0
    affiliated_media_count = 0
    party_quotes_count = 0
    publisher_ownership_count = 0
    sub_source_count = 0

    for i, ev in enumerate(source_evidence, 1):
        source = ev.get("source_type", "unknown")
        title = ev.get("title", "")
        content = ev.get("content", "")
        url = ev.get("source_url") or "N/A"

        # Get source rating (from cache)
        rating = get_source_rating_sync(url) if url != "N/A" else None
        rating_tag = _format_source_tag(rating)

        interest_warnings = []

        # Check 1: Is the source URL from affiliated media?
        if affiliated_media and url != "N/A":
            url_lower = url.lower()
            for media_outlet in affiliated_media:
                if url_matches_media(url_lower, media_outlet):
                    interest_warnings.append(
                        f"⚠️ AFFILIATED MEDIA: Source is {media_outlet}, "
                        f"which has ownership ties to claim subject."
                    )
                    interested_party_count += 1
                    affiliated_media_count += 1
                    log.debug(logger, MODULE, "affiliated_media_detected",
                              "Source URL matches affiliated media",
                              media=media_outlet, url=url)
                    break

        # Check 2: Does the content quote statements from interested parties?
        if all_parties:
            quoted_entities = detect_claim_subject_quotes(content, all_parties)
            if quoted_entities:
                interested_party_count += 1
                party_quotes_count += 1
                entities_str = ", ".join(quoted_entities)
                interest_warnings.append(
                    f"⚠️ QUOTES INTERESTED PARTY: {entities_str} — "
                    f"Self-serving statement, NOT independent verification."
                )
                log.debug(logger, MODULE, "interested_party_quoted",
                          "Evidence quotes interested party",
                          entities=quoted_entities, url=url)

        # Check 3: Is the source publisher owned by an interested party?
        if not interest_warnings and all_parties and url != "N/A":
            owner_match = check_publisher_ownership(url, all_parties)
            if owner_match:
                interested_party_count += 1
                publisher_ownership_count += 1
                interest_warnings.append(
                    f"⚠️ PUBLISHER OWNED BY INTERESTED PARTY: "
                    f"Source publisher is owned by {owner_match}."
                )
                log.debug(logger, MODULE, "publisher_ownership_detected",
                          "Source publisher owned by interested party",
                          owner=owner_match, url=url)

        # Check 4: Does the evidence reference other publications as sub-sources?
        if content:
            try:
                from src.utils.ner import extract_entities
                # Extract the primary source domain to avoid self-matching
                primary_domain = ""
                if url != "N/A":
                    primary_domain = urlparse(url).netloc.lower().replace("www.", "")

                content_orgs = extract_entities(content, labels={"ORG"})
                for org in content_orgs:
                    org_name = org["text"]

                    # Skip if org name matches the primary source domain
                    org_normalized = org_name.lower().replace(" ", "")
                    if primary_domain and org_normalized in primary_domain:
                        continue

                    sub_rating = _find_rating_by_name(org_name)
                    if sub_rating:
                        sub_bias = sub_rating.get("bias", "unknown")
                        sub_factual = sub_rating.get("factual_reporting", "unknown")

                        # Only warn for concerning sub-sources
                        low_factual = (
                            sub_factual.lower() in ("low", "very low", "mixed")
                            if sub_factual else False
                        )
                        extreme_bias = (
                            sub_bias.lower() in ("extreme-left", "extreme-right")
                            if sub_bias else False
                        )
                        if not low_factual and not extreme_bias:
                            continue

                        sub_source_count += 1
                        interest_warnings.append(
                            f"⚠️ SUB-SOURCE: References {org_name} "
                            f"[{sub_bias} | {sub_factual} factual reporting]"
                        )
                        log.debug(logger, MODULE, "sub_source_detected",
                                  "Evidence references another publication",
                                  sub_source=org_name, bias=sub_bias,
                                  factual=sub_factual, url=url)
            except Exception as e:
                log.debug(logger, MODULE, "sub_source_ner_error",
                          "NER failed during sub-source detection", error=str(e))

        # Track bias distribution for judge context
        if rating and rating.get("bias"):
            bias = rating["bias"]
            if bias in ("left", "extreme-left"):
                bias_distribution["left"] += 1
            elif bias in ("right", "extreme-right"):
                bias_distribution["right"] += 1
            elif bias in ("center", "left-center", "right-center"):
                bias_distribution["center"] += 1
            else:
                bias_distribution["unrated"] += 1
        else:
            bias_distribution["unrated"] += 1

        header = f"[{i}] {rating_tag} Source: {source}"
        if title:
            header += f" | {title}"
        header += f" | URL: {url}"

        # Add interest warnings
        for warning in interest_warnings:
            header += f"\n    {warning}"

        evidence_parts.append(f"{header}\n{content}")

    evidence_text = "\n\n".join(evidence_parts)

    # Determine bias skew for logging
    total_rated = (
        bias_distribution["left"]
        + bias_distribution["center"]
        + bias_distribution["right"]
    )
    bias_skew = "none"
    if total_rated >= 3:
        if bias_distribution["left"] / total_rated > 0.6:
            bias_skew = "left"
        elif bias_distribution["right"] / total_rated > 0.6:
            bias_skew = "right"

    log.info(logger, MODULE, "annotation_summary",
             "Evidence annotation complete",
             evidence_count=len(source_evidence),
             affiliated_media_warnings=affiliated_media_count,
             party_quote_warnings=party_quotes_count,
             publisher_ownership_warnings=publisher_ownership_count,
             sub_source_warnings=sub_source_count,
             bias_distribution=bias_distribution,
             bias_skew=bias_skew)

    # Add bias distribution warning if evidence is skewed
    if total_rated >= 3:
        left_pct = bias_distribution["left"] / total_rated
        right_pct = bias_distribution["right"] / total_rated
        if left_pct > 0.6:
            evidence_text += (
                "\n\n⚠️ BIAS WARNING: Evidence skews LEFT "
                "({}% of rated sources). Consider whether right-leaning "
                "sources have covered this differently.".format(
                    int(left_pct * 100)
                )
            )
        elif right_pct > 0.6:
            evidence_text += (
                "\n\n⚠️ BIAS WARNING: Evidence skews RIGHT "
                "({}% of rated sources). Consider whether left-leaning "
                "sources have covered this differently.".format(
                    int(right_pct * 100)
                )
            )

    # Add warning if significant portion of evidence is from interested parties
    if interested_party_count > 0 and all_parties:
        pct_interested = int(
            100 * interested_party_count / len(source_evidence)
        )
        if pct_interested >= 30:
            parties_str = ", ".join(all_parties[:5])
            if len(all_parties) > 5:
                parties_str += f" (+{len(all_parties) - 5} more)"
            reasoning = (
                interested_parties.get("reasoning")
                or "These parties have institutional or financial stake "
                "in the claim's outcome."
            )
            evidence_text += (
                f"\n\n⚠️ INTERESTED PARTY WARNING: {pct_interested}% of "
                f"evidence comes from or quotes interested parties "
                f"({parties_str}). {reasoning}\n\nThese sources cannot "
                f"independently verify claims about themselves. Look for "
                f"truly INDEPENDENT corroboration."
            )

    return evidence_text
