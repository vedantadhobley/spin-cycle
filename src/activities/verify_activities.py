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
from src.llm import get_llm
from src.prompts.verification import (
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
    SYNTHESIZE_SYSTEM,
    SYNTHESIZE_USER,
)
from src.tools.source_ratings import get_source_rating_sync, format_source_tag
from src.utils.logging import log


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

    Falls back to a single item with no thesis if JSON parsing fails.
    """
    log.info(activity.logger, "decompose", "start", "Decomposing claim",
             claim=claim_text)
    llm = get_llm()

    _t0 = time.monotonic()
    response = await llm.ainvoke([
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=DECOMPOSE_USER.format(claim_text=claim_text)),
    ])
    _latency = int((time.monotonic() - _t0) * 1000)

    raw = response.content.strip()
    log.info(activity.logger, "decompose", "llm_call", "Decompose LLM call complete",
             model="thinking=off", latency_ms=_latency)
    log.debug(activity.logger, "decompose", "llm_response", "LLM response received",
              raw=raw)

    # Parse the JSON response — now structured with entities/predicates/comparisons
    try:
        parsed = json.loads(raw)

        # Handle new structured format
        if isinstance(parsed, dict) and "predicates" in parsed:
            facts = _expand_structured(parsed)
            thesis_info = {
                "thesis": parsed.get("thesis"),
                "structure": parsed.get("structure", "simple"),
                "key_test": parsed.get("key_test"),
            }
            log.info(activity.logger, "decompose", "expanded",
                     "Structured extraction expanded to facts",
                     entities=parsed.get("entities", []),
                     predicate_count=len(parsed.get("predicates", [])),
                     comparison_count=len(parsed.get("comparisons", [])),
                     expanded_fact_count=len(facts))

        # Handle legacy format (bare list or {facts: [...]})
        elif isinstance(parsed, list):
            facts = _normalize_flat(parsed)
            thesis_info = {
                "thesis": None,
                "structure": "simple",
                "key_test": None,
            }
        elif isinstance(parsed, dict) and "facts" in parsed:
            facts = _normalize_flat(parsed["facts"])
            thesis_info = {
                "thesis": parsed.get("thesis"),
                "structure": parsed.get("structure", "simple"),
                "key_test": parsed.get("key_test"),
            }
        else:
            raise ValueError(f"Unexpected format: {type(parsed)}")

        if not facts:
            raise ValueError("LLM returned empty facts/predicates")

    except (json.JSONDecodeError, ValueError) as e:
        log.warning(activity.logger, "decompose", "parse_failed",
                    "Failed to parse LLM decomposition, using original claim",
                    error=str(e), raw=raw)
        facts = [{"text": claim_text}]
        thesis_info = {"thesis": None, "structure": "simple", "key_test": None}

    log.info(activity.logger, "decompose", "done", "Claim decomposed",
             claim=claim_text, sub_count=len(facts),
             thesis=thesis_info.get("thesis"),
             structure=thesis_info.get("structure"))
    return {"facts": facts, "thesis_info": thesis_info}


def _expand_structured(parsed: dict) -> list[dict]:
    """Expand structured extraction (entities × predicates) into flat facts.

    This is the key to ensuring completeness. Instead of relying on the LLM
    to list every fact, we:
    1. Get the entities and predicate templates
    2. Programmatically expand each predicate for each applicable entity
    3. Add comparisons directly (they're already complete facts)

    Example:
      entities: ["US", "China"]
      predicates: [{"claim": "{entity} is cutting aid", "applies_to": ["US", "China"]}]
    Expands to:
      ["US is cutting aid", "China is cutting aid"]
    """
    facts = []
    seen = set()  # Deduplicate

    # Expand predicates
    for pred in parsed.get("predicates", []):
        claim_template = pred.get("claim", "")
        applies_to = pred.get("applies_to", [])

        for item in applies_to:
            if isinstance(item, str):
                # Simple form: entity name directly
                entity = item.strip()
                fact_text = claim_template.replace("{entity}", entity)
                fact_text = fact_text.replace("{{entity}}", entity)
            elif isinstance(item, dict):
                # Detailed form: {"entity": "US", "value": "over $800B"}
                entity = item.get("entity", "").strip()
                value = item.get("value", "").strip()
                fact_text = claim_template.replace("{entity}", entity)
                fact_text = fact_text.replace("{{entity}}", entity)
                fact_text = fact_text.replace("{value}", value)
                fact_text = fact_text.replace("{{value}}", value)
            else:
                continue

            # Clean up any remaining braces the LLM might have added
            # e.g., "{US}" → "US", "{over $800 billion}" → "over $800 billion"
            fact_text = re.sub(r'\{([^}]+)\}', r'\1', fact_text)
            fact_text = fact_text.strip()
            
            if fact_text and fact_text not in seen:
                seen.add(fact_text)
                facts.append({"text": fact_text})

    # Add comparisons directly (they're already complete facts)
    for comp in parsed.get("comparisons", []):
        comp_text = comp.get("claim", "").strip()
        # Clean up any braces here too
        comp_text = re.sub(r'\{([^}]+)\}', r'\1', comp_text)
        if comp_text and comp_text not in seen:
            seen.add(comp_text)
            facts.append({"text": comp_text})

    return facts


def _normalize_flat(items: list) -> list[dict]:
    """Normalize LLM decompose output to a flat list of {"text": "..."} dicts.

    Handles:
    - Bare strings: "sub-claim" → {"text": "sub-claim"}
    - Dicts with text: {"text": "sub-claim"} → kept as-is
    - Groups (legacy): {"label": "...", "children": [...]} → flattened to children
    """
    result = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append({"text": text})
        elif isinstance(item, dict):
            if "text" in item:
                text = item["text"].strip() if isinstance(item["text"], str) else ""
                if text:
                    result.append({"text": text})
            elif "label" in item and "children" in item:
                # Legacy group output — flatten children
                result.extend(_normalize_flat(item["children"]))
    return result


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
async def judge_subclaim(claim_text: str, sub_claim: str, evidence: list[dict]) -> dict:
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

    If there's no evidence at all, we short-circuit to "unverifiable" without
    bothering the LLM.
    """
    log.info(activity.logger, "judge", "start", "Judging sub-claim",
             sub_claim=sub_claim, evidence_count=len(evidence))

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
    for i, ev in enumerate(source_evidence, 1):
        source = ev.get("source_type", "unknown")
        title = ev.get("title", "")
        content = ev.get("content", "")
        url = ev.get("source_url") or "N/A"
        
        # Get source rating (from cache)
        rating = get_source_rating_sync(url) if url != "N/A" else None
        rating_tag = format_source_tag(rating)
        
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

    # Use non-thinking mode for judge - the structured prompt already guides
    # reasoning, and thinking mode generates 5000-9500 tokens of internal
    # monologue that takes 3-4 minutes without improving verdict quality.
    llm = get_llm()
    _t0 = time.monotonic()
    response = await llm.ainvoke([
        SystemMessage(content=JUDGE_SYSTEM.format(current_date=date.today().isoformat())),
        HumanMessage(content=JUDGE_USER.format(
            claim_text=claim_text,
            sub_claim=sub_claim,
            evidence_text=evidence_text,
        )),
    ])
    _latency = int((time.monotonic() - _t0) * 1000)
    log.info(activity.logger, "judge", "llm_call", "Judge LLM call complete",
             model="thinking=off", latency_ms=_latency,
             sub_claim=sub_claim)

    raw = response.content.strip()
    original_raw = raw  # Keep original for fallback

    # Strip <think>...</think> tags from the thinking model's output.
    # The thinking model always produces chain-of-thought before its answer;
    # we capture it for debug logging, then extract just the JSON.
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        log.debug(activity.logger, "judge", "thinking", "Model reasoning captured",
                  thinking=think_match.group(1))
        raw = raw[think_match.end():].strip()

        # If nothing remains after stripping think tags, try to find JSON
        # anywhere in the original response (some models embed it in thinking)
        if not raw:
            log.warning(activity.logger, "judge", "empty_after_think",
                        "No content after </think>, searching for JSON in full response")
            # Try parsing from each '{' in the original until one works
            decoder = json.JSONDecoder()
            for i, char in enumerate(original_raw):
                if char == '{':
                    try:
                        # raw_decode handles trailing content after valid JSON
                        parsed_candidate, end_idx = decoder.raw_decode(original_raw[i:])
                        if isinstance(parsed_candidate, dict) and "verdict" in parsed_candidate:
                            raw = original_raw[i:i + end_idx]
                            log.debug(activity.logger, "judge", "json_recovered",
                                      "Found JSON in original response")
                            break
                    except json.JSONDecodeError:
                        continue

    log.debug(activity.logger, "judge", "llm_response", "LLM response received",
              raw=raw)

    # Parse the JSON verdict from the LLM
    try:
        result = json.loads(raw)
        verdict = result.get("verdict", "unverifiable")
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        nuance = result.get("nuance") or None

        # Validate verdict is one of our expected values (full 6-level scale)
        valid_verdicts = {
            "true", "mostly_true", "mixed",
            "mostly_false", "false", "unverifiable",
        }
        if verdict not in valid_verdicts:
            log.warning(activity.logger, "judge", "invalid_verdict",
                        "Invalid verdict from LLM, falling back to unverifiable",
                        verdict=verdict)
            verdict = "unverifiable"

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.warning(activity.logger, "judge", "parse_failed",
                    "Failed to parse LLM judgment",
                    error=str(e), raw=raw)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to parse LLM judgment: {raw}"
        nuance = None

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

    llm = get_llm()
    t0 = time.monotonic()
    response = await llm.ainvoke([
        SystemMessage(content=SYNTHESIZE_SYSTEM.format(
            current_date=date.today().isoformat(),
            synthesis_context=synthesis_context,
        )),
        HumanMessage(content=SYNTHESIZE_USER.format(
            synthesis_framing=synthesis_framing,
            sub_verdicts_text=sub_verdicts_text,
        )),
    ])
    latency_ms = round((time.monotonic() - t0) * 1000)
    log.info(activity.logger, "synthesize", "llm_call",
             "Synthesize LLM call completed",
             model="thinking=off", latency_ms=latency_ms, is_final=is_final)

    raw = response.content.strip()
    log.debug(activity.logger, "synthesize", "llm_response", "LLM response received",
              raw=raw)

    # Parse the JSON verdict
    try:
        result = json.loads(raw)
        verdict = result.get("verdict", "unverifiable")
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        nuance = result.get("nuance") or None

        # Validate verdict — full 6-level scale at every level
        valid_verdicts = {
            "true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"
        }
        if verdict not in valid_verdicts:
            log.warning(activity.logger, "synthesize", "invalid_verdict",
                        "Invalid verdict from LLM, falling back to unverifiable",
                        verdict=verdict)
            verdict = "unverifiable"

        confidence = max(0.0, min(1.0, confidence))

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.warning(activity.logger, "synthesize", "parse_failed",
                    "Failed to parse LLM synthesis",
                    error=str(e), raw=raw)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to parse LLM synthesis: {raw}"
        nuance = None

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
