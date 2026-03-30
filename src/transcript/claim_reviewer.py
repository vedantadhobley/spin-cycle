"""Phase 2: Claim review — classify, deduplicate, group, assess checkability.

One LLM call per speaker (batched if >MAX_CLAIMS_PER_BATCH). For speakers
with only 1 claim, a trivial group is created without calling the LLM.
"""

from __future__ import annotations

from src.llm import invoke_llm
from src.llm.validators import validate_claim_review
from src.prompts.claim_review import (
    CLAIM_REVIEW_SYSTEM,
    CLAIM_REVIEW_USER,
    format_claims_for_review,
)
from src.schemas.llm_outputs import (
    ClaimReviewOutput, ClaimGroup, ClaimClassification,
    ExtractedThesis, SupportingReference,
)
from src.utils.logging import log, get_logger

MODULE = "claim_reviewer"
logger = get_logger()

# Max claims per LLM review call — the LLM struggles to emit a classification
# for every index when the list is too long
MAX_CLAIMS_PER_BATCH = 50


async def _review_batch(
    theses: list[ExtractedThesis],
    speaker: str,
    current_date: str,
    batch_label: str,
) -> ClaimReviewOutput:
    """Review a single batch of claims via one LLM call."""
    claims_list = format_claims_for_review(theses, speaker)

    output = await invoke_llm(
        system_prompt=CLAIM_REVIEW_SYSTEM.format(current_date=current_date),
        user_prompt=CLAIM_REVIEW_USER.format(
            claim_count=len(theses),
            speaker_name=speaker,
            claims_list=claims_list,
        ),
        schema=ClaimReviewOutput,
        semantic_validator=lambda o: validate_claim_review(o, len(theses)),
        temperature=0,
        max_tokens=16384,
        activity_name=f"review_claims_{speaker}_{batch_label}",
    )
    return output


async def review_claims(
    theses: list[ExtractedThesis],
    speaker: str,
    current_date: str,
) -> ClaimReviewOutput:
    """Review and group claims for a single speaker via LLM.

    If there are more than MAX_CLAIMS_PER_BATCH claims, splits into batches,
    reviews each separately, then merges results. Cross-batch dedup and
    grouping are limited to within-batch (acceptable tradeoff — chunks
    already handle overlap dedup, and grouping within ~50 claims covers
    most related arguments).

    Args:
        theses: All Phase 1 claims for this speaker.
        speaker: Speaker name.
        current_date: ISO date string for the prompt.

    Returns:
        ClaimReviewOutput with classifications and groups.
    """
    if len(theses) <= MAX_CLAIMS_PER_BATCH:
        log.info(logger, MODULE, "reviewing",
                 f"Reviewing {len(theses)} claims for {speaker} (single batch)",
                 speaker=speaker, claim_count=len(theses))

        output = await _review_batch(theses, speaker, current_date, "all")

        log.info(logger, MODULE, "reviewed",
                 f"Review complete for {speaker}: "
                 f"{len(output.classifications)} classifications, "
                 f"{len(output.groups)} groups",
                 speaker=speaker, groups=len(output.groups))
        return output

    # Split into batches
    batches: list[list[ExtractedThesis]] = []
    for i in range(0, len(theses), MAX_CLAIMS_PER_BATCH):
        batches.append(theses[i:i + MAX_CLAIMS_PER_BATCH])

    log.info(logger, MODULE, "reviewing_batched",
             f"Reviewing {len(theses)} claims for {speaker} in {len(batches)} batches",
             speaker=speaker, claim_count=len(theses),
             batch_count=len(batches),
             batch_sizes=[len(b) for b in batches])

    # Review each batch and merge results with global indices
    all_classifications: list[ClaimClassification] = []
    all_groups: list[ClaimGroup] = []
    global_offset = 0

    for bi, batch in enumerate(batches):
        batch_output = await _review_batch(
            batch, speaker, current_date, f"batch{bi}"
        )

        # Remap local indices to global
        for cls in batch_output.classifications:
            all_classifications.append(ClaimClassification(
                claim_index=cls.claim_index + global_offset,
                classification=cls.classification,
                is_duplicate=cls.is_duplicate,
                duplicate_of=(cls.duplicate_of + global_offset
                              if cls.duplicate_of is not None else None),
                rationale=cls.rationale,
            ))

        for group in batch_output.groups:
            all_groups.append(ClaimGroup(
                member_indices=[idx + global_offset for idx in group.member_indices],
                group_rationale=group.group_rationale,
                checkable=group.checkable,
                checkability_rationale=group.checkability_rationale,
                topic=group.topic,
            ))

        log.info(logger, MODULE, "batch_reviewed",
                 f"Batch {bi} reviewed: {len(batch_output.classifications)} classifications, "
                 f"{len(batch_output.groups)} groups",
                 batch=bi, speaker=speaker)

        global_offset += len(batch)

    merged = ClaimReviewOutput(
        classifications=all_classifications,
        groups=all_groups,
    )

    log.info(logger, MODULE, "reviewed",
             f"Review complete for {speaker}: "
             f"{len(merged.classifications)} classifications, "
             f"{len(merged.groups)} groups (from {len(batches)} batches)",
             speaker=speaker, groups=len(merged.groups),
             batches=len(batches))

    return merged


def make_trivial_review(theses: list[ExtractedThesis]) -> ClaimReviewOutput:
    """Create a trivial review for a speaker with 1 claim (skip LLM).

    The single claim is classified as verifiable_fact and placed in a
    group of 1. Checkability defaults to True.
    """
    thesis = theses[0]
    return ClaimReviewOutput(
        classifications=[
            ClaimClassification(
                claim_index=0,
                classification="verifiable_fact",
                is_duplicate=False,
                rationale="Single claim — trivially classified as verifiable",
            )
        ],
        groups=[
            ClaimGroup(
                member_indices=[0],
                group_rationale="standalone claim",
                checkable=True,
                checkability_rationale="Single verifiable claim assumed checkable",
                topic=thesis.topic,
            )
        ],
    )


def build_group_text(group_claims: list[ExtractedThesis]) -> str:
    """Concatenate member thesis_statements. NOT paraphrased."""
    return "\n".join(c.thesis_statement for c in group_claims)


def build_group_references(
    group_claims: list[ExtractedThesis],
) -> list[SupportingReference]:
    """Union of all supporting references across group members, deduped by segment_index."""
    seen: set[int] = set()
    refs: list[SupportingReference] = []
    for c in group_claims:
        for ref in c.supporting_references:
            if ref.segment_index not in seen:
                refs.append(ref)
                seen.add(ref.segment_index)
    return sorted(refs, key=lambda r: r.segment_index)
