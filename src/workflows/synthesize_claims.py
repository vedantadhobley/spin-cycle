"""Temporal workflow for claim synthesis — parallel overarching claim generation.

Takes checkable groups for one speaker, runs synthesis activities in
parallel pairs (matching 2 LLM slots), returns overarching claims.
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import synthesize_claim_activity
    from src.utils.logging import log

MODULE = "synthesize_claims_workflow"


@workflow.defn
class SynthesizeClaimsWorkflow:
    """Parallel synthesis of overarching claims per checkable group.

    Input: checkable groups with member statements.
    Output: dict of group_id → {overarching_claim, rationale}.
    """

    @workflow.run
    async def run(self, groups: list[dict], speaker: str) -> dict:
        """Synthesize overarching claims for all checkable groups.

        Args:
            groups: List of dicts with:
                - group_id: str
                - topic: str
                - member_statements: list[str]
            speaker: Speaker name.

        Returns:
            Dict of group_id → {overarching_claim, rationale}.
        """
        log.info(workflow.logger, MODULE, "started",
                 f"Synthesizing {len(groups)} groups for {speaker}",
                 speaker=speaker, group_count=len(groups),
                 group_ids=[g["group_id"] for g in groups])

        results: dict[str, dict] = {}

        # Process in parallel pairs (2 LLM slots)
        pair_count = (len(groups) + 1) // 2
        for i in range(0, len(groups), 2):
            pair_num = i // 2 + 1
            batch = groups[i:i + 2]

            log.info(workflow.logger, MODULE, "pair_starting",
                     f"Synthesis pair {pair_num}/{pair_count}: "
                     f"{[g['group_id'] for g in batch]}",
                     speaker=speaker, pair=pair_num,
                     group_ids=[g["group_id"] for g in batch],
                     member_counts=[len(g["member_statements"]) for g in batch])

            futures = []
            for g in batch:
                futures.append(
                    workflow.execute_activity(
                        synthesize_claim_activity,
                        args=[g["member_statements"], g["topic"], speaker],
                        start_to_close_timeout=timedelta(seconds=300),
                        retry_policy=RetryPolicy(maximum_attempts=2),
                    )
                )
            batch_results = await asyncio.gather(*futures)
            for g, result in zip(batch, batch_results):
                results[g["group_id"]] = result
                log.info(workflow.logger, MODULE, "group_synthesized",
                         f"Group {g['group_id']}: "
                         f"\"{result['overarching_claim'][:100]}\"",
                         speaker=speaker, group_id=g["group_id"],
                         topic=g["topic"],
                         member_count=len(g["member_statements"]))

        log.info(workflow.logger, MODULE, "complete",
                 f"Synthesis complete for {speaker}: "
                 f"{len(results)}/{len(groups)} groups synthesized",
                 speaker=speaker, synthesized=len(results),
                 total=len(groups))

        return results
