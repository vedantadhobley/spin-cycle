"""Phase 2b: Claim synthesis — generate overarching claim per group.

Each group's member thesis_statements are combined into a single clean
verifiable statement. Pronouns are resolved, context is added.
"""

from __future__ import annotations

from src.llm import invoke_llm
from src.llm.validators import validate_synthesized_claim
from src.prompts.claim_review import (
    SYNTHESIZE_CLAIM_SYSTEM,
    SYNTHESIZE_CLAIM_USER,
)
from src.schemas.llm_outputs import SynthesizedClaim
from src.utils.logging import log, get_logger

MODULE = "claim_synthesizer"
logger = get_logger()


async def synthesize_group_claim(
    member_statements: list[str],
    topic: str,
    speaker: str,
) -> SynthesizedClaim:
    """Generate an overarching claim for a group of related claims.

    Args:
        member_statements: Thesis statements from all group members.
        topic: Group topic area.
        speaker: Speaker name.

    Returns:
        SynthesizedClaim with overarching_claim and rationale.
    """
    member_claims_list = "\n".join(
        f"{i+1}. \"{stmt}\"" for i, stmt in enumerate(member_statements)
    )

    log.info(logger, MODULE, "synthesize_start",
             f"Synthesizing {len(member_statements)} claims for {speaker}",
             speaker=speaker, topic=topic,
             member_count=len(member_statements))

    output = await invoke_llm(
        system_prompt=SYNTHESIZE_CLAIM_SYSTEM,
        user_prompt=SYNTHESIZE_CLAIM_USER.format(
            speaker_name=speaker,
            topic=topic,
            member_claims_list=member_claims_list,
        ),
        schema=SynthesizedClaim,
        semantic_validator=validate_synthesized_claim,
        max_tokens=4096,
        activity_name=f"synthesize_{speaker}_{topic[:20]}",
    )

    log.info(logger, MODULE, "synthesize_done",
             f"Synthesized claim: {output.overarching_claim[:80]}...",
             speaker=speaker, topic=topic)

    return output
