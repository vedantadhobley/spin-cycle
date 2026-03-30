"""Batch claim classification using LLM.

Classifies claims into categories (verifiable_fact, future_prediction, etc.)
and determines checkability. Operates on decontextualized claim text only —
no transcript context needed.
"""

from __future__ import annotations

from src.config import CLASSIFY_MAX_TOKENS
from src.llm import invoke_llm
from src.prompts.classification import CLASSIFY_CLAIMS_SYSTEM, CLASSIFY_CLAIMS_USER
from src.schemas.llm_outputs import ClassifyClaimsOutput
from src.utils.logging import log, get_logger

MODULE = "claim_classifier"
logger = get_logger()


async def classify_claims_batch(claims: list[dict]) -> list[dict]:
    """Classify a batch of claims. Returns same dicts with classification fields added.

    Args:
        claims: List of claim dicts (must have thesis_statement).

    Returns:
        Same list with classification, checkable, check_rationale added to each.
    """
    if not claims:
        return claims

    # Build numbered list for prompt
    lines = []
    for i, c in enumerate(claims):
        lines.append(f"{i}. {c['thesis_statement']}")
    claims_list = "\n".join(lines)

    log.info(logger, MODULE, "classify_start",
             f"Classifying {len(claims)} claims",
             count=len(claims))

    result = await invoke_llm(
        system_prompt=CLASSIFY_CLAIMS_SYSTEM,
        user_prompt=CLASSIFY_CLAIMS_USER.format(claims_list=claims_list),
        schema=ClassifyClaimsOutput,
        max_tokens=CLASSIFY_MAX_TOKENS,
        activity_name="classify_claims",
    )

    # Index results by position for fast lookup
    by_index: dict[int, dict] = {}
    for c in result.classifications:
        by_index[c.index] = {
            "factual_anchor": c.factual_anchor,
            "classification": c.classification,
            "checkable": c.checkable,
            "check_rationale": c.check_rationale,
        }

    # Apply classifications to claims, falling back to defaults
    classified = []
    matched = 0
    for i, claim in enumerate(claims):
        updated = {**claim}
        if i in by_index:
            updated.update(by_index[i])
            matched += 1
        else:
            updated.setdefault("factual_anchor", "")
            updated.setdefault("classification", "verifiable_fact")
            updated.setdefault("checkable", True)
            updated.setdefault("check_rationale", "")
        classified.append(updated)

    log.info(logger, MODULE, "classify_done",
             f"Classified {len(claims)} claims ({matched} matched)",
             count=len(claims), matched=matched)

    return classified
