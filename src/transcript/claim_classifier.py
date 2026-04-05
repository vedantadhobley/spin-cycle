"""Batch claim classification using LLM.

Classifies claims into categories (verifiable_fact, not_checkable)
and determines checkability. Operates on decontextualized claim text only —
no transcript context needed.

Uses targeted retry: if the LLM drops some claims from a batch, only the
missing claims are retried instead of the entire batch.
"""

from __future__ import annotations

from src.llm import invoke_llm, LLMInvocationError
from src.prompts.classification import CLASSIFY_CLAIMS_SYSTEM, CLASSIFY_CLAIMS_USER
from src.schemas.llm_outputs import ClassifyClaimsOutput
from src.utils.logging import log, get_logger

MODULE = "claim_classifier"
_default_logger = get_logger()


def _classification_to_dict(c) -> dict:
    """Convert a ClaimClassification to a dict for merging into claim dicts."""
    return {
        "factual_anchor": c.factual_anchor,
        "classification": c.classification,
        "checkable": c.checkable,
        "check_rationale": c.check_rationale,
        "topic": c.topic,
    }


async def classify_claims_batch(claims: list[dict], logger=None) -> list[dict]:
    """Classify a batch of claims. Returns same dicts with classification fields added.

    If the LLM omits some claims from its response, only the missing claims
    are retried in a targeted second call — the successfully classified claims
    are kept from the first call.

    Args:
        claims: List of claim dicts (must have thesis_statement).

    Returns:
        Same list with classification, checkable, check_rationale added to each.
    """
    logger = logger or _default_logger
    if not claims:
        return claims

    # Build numbered list for prompt
    lines = []
    for i, c in enumerate(claims):
        lines.append(f"{i}. {c['thesis_statement']}")
    claims_list = "\n".join(lines)

    log.info(logger, MODULE, "classify_start",
             "Classifying claims",
             count=len(claims))

    result = await invoke_llm(
        system_prompt=CLASSIFY_CLAIMS_SYSTEM,
        user_prompt=CLASSIFY_CLAIMS_USER.format(claims_list=claims_list),
        schema=ClassifyClaimsOutput,
        activity_name="classify_claims",
    )

    # Index results by position for fast lookup
    by_index: dict[int, dict] = {}
    for c in result.classifications:
        if 0 <= c.index < len(claims):
            by_index[c.index] = _classification_to_dict(c)

    # Check for missing claims — targeted retry instead of re-running the whole batch
    missing_indices = [i for i in range(len(claims)) if i not in by_index]

    if missing_indices:
        log.info(logger, MODULE, "classify_missing",
                 "Some claims missing from classification output, retrying",
                 total=len(claims),
                 returned=len(by_index),
                 missing_count=len(missing_indices),
                 missing_indices=missing_indices[:20])

        retry_lines = [
            f"{i}. {claims[i]['thesis_statement']}" for i in missing_indices
        ]
        try:
            retry_result = await invoke_llm(
                system_prompt=CLASSIFY_CLAIMS_SYSTEM,
                user_prompt=CLASSIFY_CLAIMS_USER.format(
                    claims_list="\n".join(retry_lines),
                ),
                schema=ClassifyClaimsOutput,
                max_retries=1,
                activity_name="classify_claims_retry",
            )
            for c in retry_result.classifications:
                if 0 <= c.index < len(claims) and c.index not in by_index:
                    by_index[c.index] = _classification_to_dict(c)
        except LLMInvocationError as e:
            log.warning(logger, MODULE, "classify_retry_failed",
                        "Targeted retry failed, using defaults for missing claims",
                        error=str(e),
                        missing_count=len(missing_indices))

    # Apply classifications to claims, falling back to defaults for any still missing
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
             "Classification complete",
             count=len(claims), matched=matched,
             retried=len(missing_indices) if missing_indices else 0)

    return classified
