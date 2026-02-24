"""Temporal activities for claim verification.

Each activity is a single unit of work that the Temporal worker executes.
Activities can be retried independently if they fail.

The verification pipeline has 6 activities:
  0. create_claim      — creates a claim record in the DB (when started from Temporal UI)
  1. decompose_claim   — LLM breaks a complex claim into atomic sub-claims
  2. research_subclaim — LangGraph agent gathers evidence using tools
  3. judge_subclaim    — LLM evaluates evidence for/against a sub-claim
  4. synthesize_verdict — LLM combines sub-verdicts into overall verdict
  5. store_result       — writes everything to Postgres

Each activity is independent and retryable. If the LLM times out during
decompose, Temporal will retry just that activity — not the whole pipeline.
"""

import json
import re
import uuid
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import select
from temporalio import activity

from src.db.session import async_session
from src.db.models import Claim, SubClaim, Evidence, Verdict
from src.llm import get_llm, get_reasoning_llm
from src.prompts.verification import (
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
    SYNTHESIZE_SYSTEM,
    SYNTHESIZE_USER,
    GROUP_SYNTHESIZE_SYSTEM,
    GROUP_SYNTHESIZE_USER,
)
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
async def decompose_claim(claim_text: str) -> list[dict]:
    """Break a claim into a tree of atomic, verifiable sub-claims using the LLM.

    Returns a tree structure as a list of nodes. Each node is either:
      - A LEAF: {"text": "verifiable assertion"}
      - A GROUP: {"label": "aspect name", "children": [...]}

    Example output for "Fort Knox not audited despite Trump + Musk promises":
      [
        {"text": "The gold in Fort Knox has not been audited recently"},
        {"label": "Audit promises", "children": [
          {"text": "Donald Trump promised to audit Fort Knox"},
          {"text": "Elon Musk or DOGE promised to audit Fort Knox"}
        ]}
      ]

    If the claim is atomic, returns a single leaf: [{"text": "the claim"}]

    Falls back to a single leaf if JSON parsing fails.
    """
    log.info(activity.logger, "decompose", "start", "Decomposing claim into sub-claim tree",
             claim=claim_text)
    llm = get_llm()

    response = await llm.ainvoke([
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=DECOMPOSE_USER.format(claim_text=claim_text)),
    ])

    raw = response.content.strip()
    log.debug(activity.logger, "decompose", "llm_response", "LLM response received",
              raw=raw)

    # Parse the JSON tree from the LLM response
    try:
        tree = json.loads(raw)
        if not isinstance(tree, list):
            raise ValueError(f"Expected list, got: {type(tree)}")
        tree = _normalize_tree(tree, claim_text)
        if not tree:
            raise ValueError("LLM returned empty tree")
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(activity.logger, "decompose", "parse_failed",
                    "Failed to parse LLM decomposition tree, using original claim",
                    error=str(e), raw=raw)
        tree = [{"text": claim_text}]

    leaf_count = _count_leaves(tree)
    log.info(activity.logger, "decompose", "done", "Claim decomposed into tree",
             claim=claim_text, leaf_count=leaf_count,
             top_level_nodes=len(tree))
    return tree


def _normalize_tree(nodes: list, claim_text: str) -> list[dict]:
    """Normalize and validate the LLM's tree output.

    Handles backwards compatibility with flat string arrays:
      ["sub-claim 1", "sub-claim 2"]  →  [{"text": "sub-claim 1"}, {"text": "sub-claim 2"}]

    Also validates structure — groups need label + children, leaves need text.
    """
    result = []
    for node in nodes:
        if isinstance(node, str):
            # Backwards compat: flat string → leaf
            if node.strip():
                result.append({"text": node.strip()})
        elif isinstance(node, dict):
            if "label" in node and "children" in node:
                # Group node — recurse into children
                children = _normalize_tree(node["children"], claim_text)
                if children:
                    result.append({"label": node["label"], "children": children})
                # Empty group → discard silently
            elif "text" in node:
                # Leaf node
                text = node["text"].strip() if isinstance(node["text"], str) else ""
                if text:
                    result.append({"text": text})
            else:
                # Unknown structure — skip
                pass
    return result


def _count_leaves(nodes: list[dict]) -> int:
    """Count leaf nodes in the tree."""
    count = 0
    for node in nodes:
        if "children" in node:
            count += _count_leaves(node["children"])
        else:
            count += 1
    return count


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
      - Does the evidence SUPPORT the claim? → "true"
      - Does the evidence CONTRADICT the claim? → "false"
      - Is the claim broadly correct but has inaccuracies? → "partially_true"
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

    # Format evidence for the LLM prompt
    evidence_parts = []
    for i, ev in enumerate(evidence, 1):
        source = ev.get("source_type", "unknown")
        content = ev.get("content", "")
        url = ev.get("source_url") or "N/A"
        evidence_parts.append(f"[{i}] Source: {source} | URL: {url}\n{content}")
    evidence_text = "\n\n".join(evidence_parts)

    llm = get_reasoning_llm()
    response = await llm.ainvoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=JUDGE_USER.format(
            claim_text=claim_text,
            sub_claim=sub_claim,
            evidence_text=evidence_text,
        )),
    ])

    raw = response.content.strip()

    # Strip <think>...</think> tags if the model used chain-of-thought
    # We let the judge think (no /no_think token) for better reasoning,
    # then extract just the JSON output after the thinking block.
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        log.debug(activity.logger, "judge", "thinking", "Model reasoning captured",
                  thinking=think_match.group(1))
        raw = raw[think_match.end():].strip()

    log.debug(activity.logger, "judge", "llm_response", "LLM response received",
              raw=raw)

    # Parse the JSON verdict from the LLM
    try:
        result = json.loads(raw)
        verdict = result.get("verdict", "unverifiable")
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        nuance = result.get("nuance") or None

        # Validate verdict is one of our expected values
        valid_verdicts = {"true", "false", "partially_true", "unverifiable"}
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
async def synthesize_verdict(claim_text: str, sub_results: list[dict]) -> dict:
    """Combine sub-claim verdicts into an overall claim verdict.

    This is the final reasoning step. The LLM looks at all the sub-claim
    verdicts and produces an overall assessment:
      - "true" if all sub-claims are well-supported
      - "mostly_true" if most are supported with minor issues
      - "mixed" if roughly half true, half false
      - "mostly_false" if most are contradicted
      - "false" if all sub-claims are clearly wrong
      - "unverifiable" if evidence is insufficient

    The LLM also explains HOW it arrived at the overall verdict by
    referencing the sub-claim verdicts. This reasoning chain is stored
    and can be displayed to users.
    """
    log.info(activity.logger, "synthesize", "start", "Synthesizing overall verdict",
             claim=claim_text, num_sub_results=len(sub_results))

    # Format sub-verdicts for the LLM prompt
    sub_verdict_parts = []
    for i, sub in enumerate(sub_results, 1):
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

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=SYNTHESIZE_SYSTEM),
        HumanMessage(content=SYNTHESIZE_USER.format(
            claim_text=claim_text,
            sub_verdicts_text=sub_verdicts_text,
        )),
    ])

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

        # Validate overall verdict
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

    log.info(activity.logger, "synthesize", "done", "Overall verdict synthesized",
             claim=claim_text, verdict=verdict, confidence=confidence,
             reasoning=reasoning if reasoning else None,
             nuance=nuance if nuance else None)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "reasoning_chain": [sub.get("reasoning", "") for sub in sub_results],
        "nuance": nuance,
        "sub_results": sub_results,
    }


@activity.defn
async def synthesize_group(claim_text: str, group_label: str, child_results: list[dict]) -> dict:
    """Synthesize child verdicts into an intermediate group verdict.

    This is like synthesize_verdict but for intermediate tree nodes.
    A group like "Audit promises" might contain:
      - "Trump promised audit" → true (0.9)
      - "Musk/DOGE promised audit" → true (0.85)
    And this activity produces: "Both promised" → true (0.85)

    The group result looks like a leaf result but also carries child_results
    so the final synthesis and storage can see the full tree.
    """
    log.info(activity.logger, "synthesize_group", "start",
             "Synthesizing group verdict",
             claim=claim_text, group_label=group_label,
             num_children=len(child_results))

    # Format child verdicts for the prompt
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

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=GROUP_SYNTHESIZE_SYSTEM.format(group_label=group_label)),
        HumanMessage(content=GROUP_SYNTHESIZE_USER.format(
            group_label=group_label,
            claim_text=claim_text,
            sub_verdicts_text=sub_verdicts_text,
        )),
    ])

    raw = response.content.strip()
    log.debug(activity.logger, "synthesize_group", "llm_response",
              "LLM response received", raw=raw)

    try:
        result = json.loads(raw)
        verdict = result.get("verdict", "unverifiable")
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "")
        nuance = result.get("nuance") or None

        valid_verdicts = {"true", "false", "partially_true", "unverifiable"}
        if verdict not in valid_verdicts:
            log.warning(activity.logger, "synthesize_group", "invalid_verdict",
                        "Invalid group verdict, falling back to unverifiable",
                        verdict=verdict)
            verdict = "unverifiable"

        confidence = max(0.0, min(1.0, confidence))

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.warning(activity.logger, "synthesize_group", "parse_failed",
                    "Failed to parse group synthesis",
                    error=str(e), raw=raw)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to parse group synthesis: {raw}"
        nuance = None

    log.info(activity.logger, "synthesize_group", "done",
             "Group verdict synthesized",
             group_label=group_label, verdict=verdict, confidence=confidence)

    return {
        "sub_claim": group_label,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "nuance": nuance,
        "evidence": [],
        "child_results": child_results,
    }


@activity.defn
async def store_result(claim_id: str, verdict: dict, sub_results: list[dict], tree: list[dict] | None = None) -> None:
    """Store verification result in the database.

    Handles both flat and tree-structured results. For tree structures,
    group nodes get parent_id set on their children.
    """
    log.info(activity.logger, "store", "start", "Storing verification result",
             claim_id=claim_id, verdict=verdict.get("verdict"))

    async with async_session() as session:
        async with session.begin():
            # Fetch the claim
            claim_uuid = uuid.UUID(claim_id)
            result = await session.execute(
                select(Claim).where(Claim.id == claim_uuid)
            )
            claim = result.scalar_one_or_none()
            if not claim:
                log.error(activity.logger, "store", "claim_not_found",
                          "Claim not found in database",
                          error="claim_not_found", claim_id=claim_id)
                return

            # Write sub-claims + evidence recursively
            async def _store_node(sub: dict, parent_id=None):
                """Store a sub-result node (leaf or group) with its evidence and children."""
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

                # Store evidence (leaves have evidence, groups don't)
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

                # Recurse into children for group nodes
                if has_children:
                    for child in sub["child_results"]:
                        await _store_node(child, parent_id=sub_claim.id)

            for sub in sub_results:
                await _store_node(sub)

            # Write verdict
            verdict_row = Verdict(
                claim_id=claim_uuid,
                verdict=verdict.get("verdict", "unverifiable"),
                confidence=verdict.get("confidence", 0.0),
                reasoning=verdict.get("reasoning"),
                reasoning_chain=verdict.get("reasoning_chain"),
                nuance=verdict.get("nuance"),
            )
            session.add(verdict_row)

            # Update claim status
            claim.status = "verified"
            claim.updated_at = datetime.now(timezone.utc)

        log.info(activity.logger, "store", "done", "Result stored in database",
                 claim_id=claim_id, verdict=verdict.get("verdict"),
                 sub_claims=len(sub_results))
