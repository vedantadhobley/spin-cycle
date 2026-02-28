"""Temporal activities for claim verification.

Each activity is a single unit of work that the Temporal worker executes.
Activities can be retried independently if they fail.

The verification pipeline has 6 activities:
  0. create_claim      — creates a claim record in the DB (when started from Temporal UI)
  1. decompose_claim   — LLM extracts atomic facts from a claim (one flat pass)
  2. research_subclaim — LangGraph agent gathers evidence using tools
  3. judge_subclaim    — LLM evaluates evidence for/against a sub-claim
  4. synthesize_verdict — LLM combines sub-verdicts into a final verdict
  5. store_result       — writes result to Postgres

The pipeline is flat: decompose once → research+judge each fact → synthesize.
No recursion, no tree — follows the approach used by Google's SAFE and FActScore.
"""

import json
import re
import time
import uuid
from datetime import date, datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import select
from temporalio import activity

from src.db.session import async_session
from src.db.models import Claim, SubClaim, Evidence, Verdict
from src.llm import (
    get_llm,
    invoke_llm,
    create_fallback,
    LLMInvocationError,
    validate_decompose,
    validate_judge,
    validate_synthesize,
)
from src.prompts.verification import (
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
    SYNTHESIZE_SYSTEM,
    SYNTHESIZE_USER,
)
from src.prompts.linguistic_patterns import get_linguistic_patterns
from src.schemas.llm_outputs import (
    DecomposeOutput,
    JudgeOutput,
    SynthesizeOutput,
    InterestedParties,
)
from src.tools.source_ratings import get_source_rating_sync, format_source_tag, detect_claim_subject_quotes
from src.utils.logging import log
from src.utils.text_cleanup import cleanup_text


@activity.defn
async def create_claim(claim_text: str) -> str:
    """Create a claim record in the database and return its ID.

    This lets workflows be self-contained — you can start a VerifyClaimWorkflow
    from Temporal UI with just the claim text, and it will create the DB record
    automatically. When started via the API, the claim already exists so this
    activity is skipped.
    """
    log.info(activity.logger, "create", "start", "Creating claim record",
             claim=claim_text)

    async with async_session() as session:
        async with session.begin():
            claim = Claim(
                text=claim_text,
                status="pending",
            )
            session.add(claim)
            await session.flush()
            claim_id = str(claim.id)

    log.info(activity.logger, "create", "done", "Claim record created",
             claim_id=claim_id)
    return claim_id


@activity.defn
async def decompose_claim(claim_text: str) -> dict:
    """Extract atomic verifiable facts and thesis from a claim in one pass.

    Uses STRUCTURED extraction (entities + predicates + comparisons) and then
    programmatically expands entity × predicate combinations. This ensures
    completeness — when a claim says "both X and Y do Z", we're guaranteed
    to verify Z for both X and Y.

    Returns a dict with:
      - "facts": list of {"text": "..."} dicts, each an atomic fact
      - "thesis_info": {"thesis": "...", "structure": "...", "key_test": "..."}

    Falls back to a single item with no thesis if invocation fails after retries.
    """
    log.info(activity.logger, "decompose", "start", "Decomposing claim",
             claim=claim_text)

    # Build decompose system prompt with linguistic patterns
    decompose_system_with_patterns = DECOMPOSE_SYSTEM + "\n\n" + get_linguistic_patterns()

    try:
        # Use unified invoker with schema validation and automatic retry
        output = await invoke_llm(
            system_prompt=decompose_system_with_patterns,
            user_prompt=DECOMPOSE_USER.format(claim_text=claim_text),
            schema=DecomposeOutput,
            semantic_validator=validate_decompose,
            max_retries=2,
            activity_name="decompose",
        )
        
        # Convert facts to dict format
        facts = [{"text": f.strip()} for f in output.facts if f and f.strip()]
        
        # Build thesis_info from validated output
        interested_parties_obj = _normalize_interested_parties(output.interested_parties)
        thesis_info = {
            "thesis": output.thesis,
            "structure": output.structure,
            "key_test": output.key_test,
            "interested_parties": interested_parties_obj,
        }
        
        log.info(activity.logger, "decompose", "facts_extracted",
                 "Decomposition complete",
                 interested_parties=interested_parties_obj,
                 fact_count=len(facts))

    except LLMInvocationError as e:
        # All retries failed — use fallback
        log.warning(activity.logger, "decompose", "invocation_failed",
                    "LLM invocation failed after retries, using original claim as single fact",
                    error=str(e), attempts=e.attempts)
        facts = [{"text": claim_text}]
        thesis_info = {
            "thesis": None,
            "structure": "simple",
            "key_test": None,
            "interested_parties": _normalize_interested_parties([]),
        }

    log.info(activity.logger, "decompose", "done", "Claim decomposed",
             claim=claim_text, sub_count=len(facts),
             thesis=thesis_info.get("thesis"),
             structure=thesis_info.get("structure"))
    return {"facts": facts, "thesis_info": thesis_info}


def _normalize_interested_parties(raw) -> dict:
    """Normalize interested_parties to a consistent structure.
    
    The decomposition now returns interested_parties as an object with:
    - direct: Organizations immediately involved
    - institutional: Parent/governing bodies
    - affiliated_media: News outlets with ownership ties
    - reasoning: Brief explanation of relationships
    
    For backward compatibility, also handles legacy list format and InterestedParties objects.
    
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


# Common media outlet domain aliases for better URL matching
# Maps common name variations to their likely domain patterns
_MEDIA_DOMAIN_ALIASES = {
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


def _url_matches_media(url_lower: str, media_outlet: str) -> bool:
    """Check if a URL matches a media outlet name, handling common variations.
    
    This handles cases like:
    - "Washington Post" matching washingtonpost.com
    - "New York Times" matching nytimes.com
    - "Fox News" matching foxnews.com
    
    Args:
        url_lower: Lowercase URL to check
        media_outlet: Name of media outlet (e.g., "Washington Post")
    
    Returns:
        True if URL appears to be from this media outlet
    """
    media_lower = media_outlet.lower()
    
    # Check known aliases first
    for name, aliases in _MEDIA_DOMAIN_ALIASES.items():
        if name in media_lower or media_lower in name:
            for alias in aliases:
                if alias in url_lower:
                    return True
    
    # Fall back to generic normalization
    # "Washington Post" -> "washingtonpost"
    normalized = media_lower.replace(" ", "").replace("the", "")
    if normalized and len(normalized) > 3 and normalized in url_lower:
        return True
    
    # Also try hyphenated: "washington-post"
    hyphenated = media_lower.replace(" ", "-")
    if hyphenated in url_lower:
        return True
    
    return False


@activity.defn
async def research_subclaim(sub_claim: str) -> list[dict]:
    """Research evidence for a sub-claim using the LangGraph ReAct agent.

    This is where the "agentic AI" happens. Instead of a single LLM call,
    we run a research agent that:
      1. Decides what to search for (based on the claim)
      2. Calls web search and Wikipedia tools
      3. Reads the results
      4. Decides if it needs to search more
      5. Loops until it has enough evidence or gives up

    The agent runs inside this Temporal activity, which provides:
      - 120s timeout (set in the workflow)
      - 3 retry attempts
      - Crash recovery (if the worker dies, Temporal reruns this)

    Returns a list of evidence dicts for the judge step.
    """
    log.info(activity.logger, "research", "start", "Researching sub-claim",
             sub_claim=sub_claim)

    from src.agent.research import research_claim
    evidence = await research_claim(sub_claim)

    log.info(activity.logger, "research", "done", "Research complete",
             sub_claim=sub_claim, evidence_count=len(evidence))
    return evidence


@activity.defn
async def judge_subclaim(
    claim_text: str,
    sub_claim: str,
    evidence: list[dict],
    interested_parties: dict | list | None = None,
) -> dict:
    """Judge a sub-claim based on collected evidence.

    This is the critical evaluation step. The LLM looks at the evidence
    gathered by the research agent and determines:
      - Does the evidence SUPPORT the claim? → "true" or "mostly_true"
      - Does the evidence CONTRADICT the claim? → "false" or "mostly_false"
      - Is the picture mixed? → "mixed"
      - Is there not enough evidence to decide? → "unverifiable"

    The key constraint: the LLM must reason ONLY from the provided evidence,
    NOT from its own training data. This is what makes the verdict trustworthy
    — it's grounded in real, citable sources.

    The original claim_text is passed for context so the judge can interpret
    the sub-claim naturally (e.g., "has not been audited" in the context of
    a claim about promised audits means the promised audit hasn't happened).

    interested_parties: A dict containing:
      - all_parties: flat list of all interested orgs
      - direct: orgs immediately involved
      - institutional: parent/governing bodies
      - affiliated_media: news outlets with ownership ties
      - reasoning: explanation of relationships
    
    Evidence from interested parties (or their affiliated media) is flagged
    as self-serving and cannot independently verify or refute claims about them.

    If there's no evidence at all, we short-circuit to "unverifiable" without
    bothering the LLM.
    """
    # Normalize interested_parties to new structure
    if interested_parties is None:
        interested_parties = _normalize_interested_parties([])
    elif isinstance(interested_parties, list):
        interested_parties = _normalize_interested_parties(interested_parties)
    
    all_parties = interested_parties.get("all_parties", [])
    affiliated_media = interested_parties.get("affiliated_media", [])
    
    log.info(activity.logger, "judge", "start", "Judging sub-claim",
             sub_claim=sub_claim, evidence_count=len(evidence),
             interested_parties=interested_parties)

    # No evidence → unverifiable (no point asking the LLM)
    if not evidence:
        log.info(activity.logger, "judge", "no_evidence", "No evidence found, returning unverifiable",
                 sub_claim=sub_claim)
        return {
            "sub_claim": sub_claim,
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reasoning": "No evidence was found for this claim.",
            "evidence": [],
        }

    # Filter out agent_summary — it's the research agent's interpretation,
    # not primary evidence. The judge should reason from sources only.
    source_evidence = [
        ev for ev in evidence
        if ev.get("source_type") != "agent_summary"
    ]

    # Cap evidence to keep the judge prompt manageable.
    # The thinking model's chain-of-thought scales super-linearly with
    # evidence count — 34 items can take >180s while 20 takes ~60s.
    # 20 unique items is plenty for a well-grounded verdict.
    MAX_JUDGE_EVIDENCE = 20
    if len(source_evidence) > MAX_JUDGE_EVIDENCE:
        log.info(activity.logger, "judge", "evidence_capped",
                 "Capping evidence for judge prompt",
                 original=len(source_evidence), capped=MAX_JUDGE_EVIDENCE)
        source_evidence = source_evidence[:MAX_JUDGE_EVIDENCE]

    # Log which URLs the judge will see — critical for debugging verdicts
    evidence_urls = [
        ev.get("source_url") or "N/A"
        for ev in source_evidence
    ]
    source_types = {}
    for ev in source_evidence:
        st = ev.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1
    log.debug(activity.logger, "judge", "evidence_summary",
              "Evidence prepared for judge",
              sub_claim=sub_claim,
              evidence_count=len(source_evidence),
              source_types=source_types,
              urls=evidence_urls)

    # Format evidence for the LLM prompt with source bias/credibility ratings
    evidence_parts = []
    bias_distribution = {"left": 0, "center": 0, "right": 0, "unrated": 0}
    interested_party_count = 0  # Track evidence from interested parties
    
    for i, ev in enumerate(source_evidence, 1):
        source = ev.get("source_type", "unknown")
        title = ev.get("title", "")
        content = ev.get("content", "")
        url = ev.get("source_url") or "N/A"
        
        # Get source rating (from cache)
        rating = get_source_rating_sync(url) if url != "N/A" else None
        rating_tag = format_source_tag(rating)
        
        interest_warnings = []
        
        # Check 1: Is the source URL from affiliated media?
        # e.g., washingtonpost.com when claim is about Amazon (both owned by Bezos)
        if affiliated_media and url != "N/A":
            url_lower = url.lower()
            for media_outlet in affiliated_media:
                if _url_matches_media(url_lower, media_outlet):
                    interest_warnings.append(
                        f"⚠️ AFFILIATED MEDIA: Source is {media_outlet}, which has ownership ties to claim subject."
                    )
                    interested_party_count += 1
                    log.debug(activity.logger, "judge", "affiliated_media_detected",
                              "Source URL matches affiliated media",
                              media=media_outlet, url=url)
                    break
        
        # Check 2: Does the content quote statements from interested parties?
        # e.g., "FBI stated that..." when claim is about FBI conduct
        if all_parties:
            quoted_entities = detect_claim_subject_quotes(content, all_parties)
            if quoted_entities:
                interested_party_count += 1
                entities_str = ", ".join(quoted_entities)
                interest_warnings.append(
                    f"⚠️ QUOTES INTERESTED PARTY: {entities_str} — Self-serving statement, NOT independent verification."
                )
                log.debug(activity.logger, "judge", "interested_party_quoted",
                          "Evidence quotes interested party",
                          entities=quoted_entities, url=url)
        
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
    
    # Add bias distribution warning if evidence is skewed
    bias_warning = ""
    total_rated = bias_distribution["left"] + bias_distribution["center"] + bias_distribution["right"]
    if total_rated >= 3:
        left_pct = bias_distribution["left"] / total_rated
        right_pct = bias_distribution["right"] / total_rated
        if left_pct > 0.6:
            bias_warning = "\n\n⚠️ BIAS WARNING: Evidence skews LEFT ({}% of rated sources). Consider whether right-leaning sources have covered this differently.".format(int(left_pct * 100))
        elif right_pct > 0.6:
            bias_warning = "\n\n⚠️ BIAS WARNING: Evidence skews RIGHT ({}% of rated sources). Consider whether left-leaning sources have covered this differently.".format(int(right_pct * 100))
    
    if bias_warning:
        evidence_text += bias_warning
    
    # Add warning if significant portion of evidence is from interested parties
    if interested_party_count > 0 and all_parties:
        pct_interested = int(100 * interested_party_count / len(source_evidence))
        if pct_interested >= 30:
            parties_str = ", ".join(all_parties[:5])  # Limit display
            if len(all_parties) > 5:
                parties_str += f" (+{len(all_parties) - 5} more)"
            reasoning = interested_parties.get("reasoning") or "These parties have institutional or financial stake in the claim's outcome."
            evidence_text += f"\n\n⚠️ INTERESTED PARTY WARNING: {pct_interested}% of evidence comes from or quotes interested parties ({parties_str}). {reasoning}\n\nThese sources cannot independently verify claims about themselves. Look for truly INDEPENDENT corroboration."

    # Use non-thinking mode for judge - the structured prompt already guides
    # reasoning, and thinking mode generates 5000-9500 tokens of internal
    # monologue that takes 3-4 minutes without improving verdict quality.
    try:
        output = await invoke_llm(
            system_prompt=JUDGE_SYSTEM.format(current_date=date.today().isoformat()),
            user_prompt=JUDGE_USER.format(
                claim_text=claim_text,
                sub_claim=sub_claim,
                evidence_text=evidence_text,
            ),
            schema=JudgeOutput,
            semantic_validator=validate_judge,
            max_retries=2,
            activity_name="judge",
        )
        
        verdict = output.verdict
        confidence = output.confidence
        reasoning = output.reasoning
        nuance = output.nuance
        
    except LLMInvocationError as e:
        log.warning(activity.logger, "judge", "invocation_failed",
                    "LLM invocation failed after retries",
                    error=str(e), attempts=e.attempts)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to parse LLM judgment after {e.attempts} attempts"
        nuance = None

    # Clean up reasoning and nuance text using LanguageTool
    # This catches grammar oddities from quantized model outputs
    reasoning = cleanup_text(reasoning)
    nuance = cleanup_text(nuance)

    log.info(activity.logger, "judge", "done", "Sub-claim judged",
             sub_claim=sub_claim, verdict=verdict, confidence=confidence,
             nuance=nuance if nuance else None)

    return {
        "sub_claim": sub_claim,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "nuance": nuance,
        "evidence": evidence,
    }


@activity.defn
async def synthesize_verdict(
    claim_text: str,
    node_text: str,
    child_results: list[dict],
    is_final: bool = True,
    thesis_info: dict | None = None,
) -> dict:
    """Combine child verdicts into a single verdict for a tree node.

    Unified synthesis — works for both intermediate (one aspect of the claim)
    and final (overall verdict). The prompt adapts via context parameters:
      - Final: "This is the FINAL OVERALL verdict for the original claim."
      - Intermediate: "This is an INTERMEDIATE verdict for one aspect..."

    Both use the full 6-level verdict scale.
    """
    log.info(activity.logger, "synthesize", "start", "Synthesizing verdict",
             claim=claim_text, node=node_text, is_final=is_final,
             num_children=len(child_results))

    # Format sub-verdicts for the LLM prompt
    sub_verdict_parts = []
    for i, sub in enumerate(child_results, 1):
        part = (
            f"[{i}] Sub-claim: {sub['sub_claim']}\n"
            f"    Verdict: {sub['verdict']}\n"
            f"    Confidence: {sub['confidence']}\n"
            f"    Reasoning: {sub['reasoning']}"
        )
        if sub.get("nuance"):
            part += f"\n    Nuance: {sub['nuance']}"
        sub_verdict_parts.append(part)
    sub_verdicts_text = "\n\n".join(sub_verdict_parts)

    # Adapt prompt context based on whether this is final or intermediate
    if is_final:
        synthesis_context = (
            "This is the FINAL OVERALL verdict for the original claim. "
            "Your verdict is the definitive assessment."
        )
        # Build thesis context for the synthesizer
        thesis_block = ""
        if thesis_info and thesis_info.get("thesis"):
            thesis_block = (
                f"\n\nSPEAKER'S THESIS: {thesis_info['thesis']}\n"
                f"Claim structure: {thesis_info.get('structure', 'simple')}\n"
                f"Key test: {thesis_info.get('key_test', 'N/A')}\n"
                f"\nEvaluate whether THIS THESIS survives the sub-verdicts, "
                f"not just whether a majority of individual facts are true."
            )
        synthesis_framing = f"Original claim: {claim_text}{thesis_block}"
    else:
        synthesis_context = (
            f'This is an INTERMEDIATE verdict for one aspect of a larger claim: '
            f'"{node_text}". This verdict will be combined with other aspect '
            f'verdicts in a later synthesis step.'
        )
        synthesis_framing = (
            f"Aspect being synthesized: {node_text}\n"
            f"Original claim (for context): {claim_text}"
        )

    try:
        output = await invoke_llm(
            system_prompt=SYNTHESIZE_SYSTEM.format(
                current_date=date.today().isoformat(),
                synthesis_context=synthesis_context,
            ),
            user_prompt=SYNTHESIZE_USER.format(
                synthesis_framing=synthesis_framing,
                sub_verdicts_text=sub_verdicts_text,
            ),
            schema=SynthesizeOutput,
            semantic_validator=validate_synthesize,
            max_retries=2,
            activity_name="synthesize",
        )
        
        verdict = output.verdict
        confidence = output.confidence
        reasoning = output.reasoning
        nuance = output.nuance
        
    except LLMInvocationError as e:
        log.warning(activity.logger, "synthesize", "invocation_failed",
                    "LLM invocation failed after retries",
                    error=str(e), attempts=e.attempts)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to synthesize verdict after {e.attempts} attempts"
        nuance = None

    # Clean up reasoning and nuance text using LanguageTool
    # This catches grammar oddities from quantized model outputs
    reasoning = cleanup_text(reasoning)
    nuance = cleanup_text(nuance)

    log.info(activity.logger, "synthesize", "done", "Verdict synthesized",
             node=node_text, is_final=is_final, verdict=verdict,
             confidence=confidence)

    return {
        "sub_claim": node_text,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "nuance": nuance,
        "evidence": [],
        "child_results": child_results,
        "reasoning_chain": (
            [sub.get("reasoning", "") for sub in child_results]
            if is_final else None
        ),
    }


@activity.defn
async def store_result(claim_id: str, result: dict) -> None:
    """Store the full verification result tree in the database.

    The result dict is the output of the recursive _process function in the
    workflow. It's either:
      - A leaf: {"sub_claim": "...", "verdict": "...", "evidence": [...]}
      - A synthesis: {"sub_claim": "...", "verdict": "...", "child_results": [...]}

    The tree is stored recursively — synthesis nodes get is_leaf=False,
    leaf nodes get is_leaf=True, and parent_id links children to parents.
    """
    log.info(activity.logger, "store", "start", "Storing verification result",
             claim_id=claim_id, verdict=result.get("verdict"))

    async with async_session() as session:
        async with session.begin():
            # Fetch the claim
            claim_uuid = uuid.UUID(claim_id)
            db_result = await session.execute(
                select(Claim).where(Claim.id == claim_uuid)
            )
            claim = db_result.scalar_one_or_none()
            if not claim:
                log.error(activity.logger, "store", "claim_not_found",
                          "Claim not found in database",
                          error="claim_not_found", claim_id=claim_id)
                return

            # Recursive storage of the result tree
            async def _store_node(sub: dict, parent_id=None):
                """Store a sub-result node (leaf or synthesis) with evidence and children."""
                has_children = "child_results" in sub and sub["child_results"]
                sub_claim = SubClaim(
                    claim_id=claim_uuid,
                    parent_id=parent_id,
                    is_leaf=not has_children,
                    text=sub["sub_claim"],
                    verdict=sub.get("verdict"),
                    confidence=sub.get("confidence"),
                    reasoning=sub.get("reasoning"),
                    nuance=sub.get("nuance"),
                )
                session.add(sub_claim)
                await session.flush()  # get sub_claim.id

                # Store evidence (leaves have evidence, synthesis nodes don't)
                for ev in sub.get("evidence", []):
                    source_type = ev.get("source_type", "web")
                    if source_type not in ("web", "wikipedia", "news_api"):
                        continue
                    evidence = Evidence(
                        sub_claim_id=sub_claim.id,
                        source_type=source_type,
                        source_url=ev.get("source_url"),
                        content=ev.get("content"),
                        supports_claim=ev.get("supports_claim"),
                    )
                    session.add(evidence)

                # Recurse into children for synthesis nodes
                if has_children:
                    for child in sub["child_results"]:
                        await _store_node(child, parent_id=sub_claim.id)

            # Store sub-claims recursively
            child_results = result.get("child_results")
            if child_results:
                # Complex claim — top-level children are the first level of sub-claims
                for sub in child_results:
                    await _store_node(sub)
            else:
                # Simple atomic claim — result itself is the only sub-claim
                await _store_node(result)

            # Write verdict
            verdict_row = Verdict(
                claim_id=claim_uuid,
                verdict=result.get("verdict", "unverifiable"),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning"),
                reasoning_chain=result.get("reasoning_chain"),
                nuance=result.get("nuance"),
            )
            session.add(verdict_row)

            # Update claim status
            claim.status = "verified"
            claim.updated_at = datetime.now(timezone.utc)

        log.info(activity.logger, "store", "done", "Result stored in database",
                 claim_id=claim_id, verdict=result.get("verdict"))


@activity.defn
async def start_next_queued_claim() -> str | None:
    """Check for queued claims and start the next one.
    
    Called at the end of each workflow to trigger the next claim in queue.
    Returns the claim_id if a workflow was started, None otherwise.
    """
    import os
    from temporalio.client import Client as TemporalClient
    
    TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
    TASK_QUEUE = "spin-cycle-verify"
    
    async with async_session() as session:
        # Find oldest queued claim
        result = await session.execute(
            select(Claim)
            .where(Claim.status == "queued")
            .order_by(Claim.created_at.asc())
            .limit(1)
        )
        claim = result.scalar_one_or_none()
        
        if not claim:
            log.info(activity.logger, "queue", "empty", "No queued claims")
            return None
        
        claim_id = str(claim.id)
        claim_text = claim.text
        
        # Update status to pending before starting workflow
        claim.status = "pending"
        claim.updated_at = datetime.now(timezone.utc)
        await session.commit()
        
        log.info(activity.logger, "queue", "starting_next", 
                 "Starting next queued claim",
                 claim_id=claim_id)
    
    # Connect to Temporal and start the workflow
    # Import here to avoid circular dependency with workflow module
    from src.workflows.verify import VerifyClaimWorkflow
    
    temporal = await TemporalClient.connect(TEMPORAL_HOST)
    await temporal.start_workflow(
        VerifyClaimWorkflow.run,
        args=[claim_id, claim_text],
        id=f"verify-{claim_id}",
        task_queue=TASK_QUEUE,
    )
    
    log.info(activity.logger, "queue", "workflow_started",
             "Queued claim workflow started",
             claim_id=claim_id)
    
    return claim_id
