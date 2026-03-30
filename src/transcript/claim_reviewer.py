"""Phase 2: Claim review — sequential batches with accumulating group context.

Each review_batch() call processes ~10 claims against accumulated groups.
The ReviewClaimsWorkflow orchestrates sequential calls, passing group state
between batches. Each call is a separate Temporal activity.
"""

from __future__ import annotations

from src.llm import invoke_llm
from src.llm.validators import validate_review_batch
from src.prompts.claim_review import (
    REVIEW_BATCH_SYSTEM,
    REVIEW_BATCH_USER,
    format_claims_for_review,
    format_existing_groups,
)
from src.schemas.llm_outputs import (
    ReviewBatchOutput, ExtractedThesis,
)
from src.utils.logging import log, get_logger

MODULE = "claim_reviewer"
logger = get_logger()

from src.config import REVIEW_BATCH_SIZE


async def review_batch(
    theses: list[ExtractedThesis],
    speaker: str,
    current_date: str,
    existing_groups: dict,
    existing_group_ids: list[str],
    batch_label: str,
) -> ReviewBatchOutput:
    """Review a single batch of ~10 claims via one LLM call.

    Args:
        theses: Claims in this batch (max ~10).
        speaker: Speaker name.
        current_date: ISO date string.
        existing_groups: Accumulated groups from prior batches for context.
        existing_group_ids: List of all existing group IDs for validation.
        batch_label: Label for logging/activity naming.

    Returns:
        ReviewBatchOutput with dispositions and new_groups.
    """
    claims_list = format_claims_for_review(theses, speaker)
    existing_groups_section = format_existing_groups(existing_groups)

    log.info(logger, MODULE, "review_batch_start",
             f"Reviewing batch of {len(theses)} claims for {speaker}",
             speaker=speaker, batch=batch_label,
             claim_count=len(theses),
             existing_groups=len(existing_groups))

    output = await invoke_llm(
        system_prompt=REVIEW_BATCH_SYSTEM.format(current_date=current_date),
        user_prompt=REVIEW_BATCH_USER.format(
            claim_count=len(theses),
            speaker_name=speaker,
            existing_groups_section=existing_groups_section,
            claims_list=claims_list,
        ),
        schema=ReviewBatchOutput,
        semantic_validator=lambda o: validate_review_batch(
            o, len(theses), existing_group_ids
        ),
        temperature=0,
        max_tokens=16384,
        activity_name=f"review_batch_{speaker}_{batch_label}",
    )

    log.info(logger, MODULE, "review_batch_done",
             f"Batch reviewed: {len(output.dispositions)} dispositions, "
             f"{len(output.new_groups)} new groups",
             speaker=speaker, batch=batch_label,
             dispositions=len(output.dispositions),
             new_groups=len(output.new_groups))

    return output


def make_trivial_review(thesis_dict: dict) -> dict:
    """Create trivial review result for a speaker with 1 claim (skip LLM).

    Returns a dict matching the ReviewClaimsWorkflow output format.
    """
    return {
        "dispositions": [
            {
                "claim_index": 0,
                "classification": "verifiable_fact",
                "action": "new_group",
                "group_id": "G1",
                "rationale": "Single claim — trivially classified as verifiable",
            }
        ],
        "groups": {
            "G1": {
                "topic": thesis_dict.get("topic", ""),
                "checkable": True,
                "checkability_rationale": "Single verifiable claim assumed checkable",
                "member_indices": [0],
                "representative_statement": thesis_dict.get("thesis_statement", ""),
            },
        },
    }
