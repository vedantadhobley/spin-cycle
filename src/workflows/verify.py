"""Temporal workflow for claim verification."""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.verify_activities import (
        create_claim,
        decompose_claim,
        research_subclaim,
        judge_subclaim,
        synthesize_verdict,
        store_result,
    )
    from src.utils.logging import log

MODULE = "workflow"


@workflow.defn
class VerifyClaimWorkflow:
    """Orchestrates the full claim verification pipeline.

    Steps:
    1. Decompose claim into atomic sub-claims
    2. For each sub-claim: research evidence, then judge
    3. Synthesize overall verdict
    4. Store result in database
    """

    @workflow.run
    async def run(self, claim_id: str | None, claim_text: str) -> dict:
        """Run the verification pipeline.

        Args:
            claim_id: Existing claim UUID from the database, or None.
                      When None, the workflow creates the claim record itself.
                      This lets you start workflows from Temporal UI with just
                      the claim text — pass [null, "claim text here"].
            claim_text: The claim to verify.
        """
        # Step 0: Create claim record if we don't have one
        if not claim_id:
            claim_id = await workflow.execute_activity(
                create_claim,
                args=[claim_text],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            log.info(workflow.logger, MODULE, "claim_created", "Created claim record",
                     claim_id=claim_id)

        log.info(workflow.logger, MODULE, "started", "Starting verification pipeline",
                 claim_id=claim_id, claim=claim_text[:80])

        # Step 1: Decompose into sub-claims
        sub_claims = await workflow.execute_activity(
            decompose_claim,
            args=[claim_text],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        log.info(workflow.logger, MODULE, "decomposed", "Claim decomposed into sub-claims",
                 claim_id=claim_id, num_sub_claims=len(sub_claims))

        # Step 2: Research + judge each sub-claim (IN PARALLEL)
        # Sub-claims are independent — no reason to wait for one before
        # starting the next. This cuts total time from O(n * per_claim)
        # to O(max_per_claim).
        async def _process_subclaim(index: int, sub_claim: str) -> dict:
            log.info(workflow.logger, MODULE, "subclaim_start", "Processing sub-claim",
                     claim_id=claim_id, index=index + 1, total=len(sub_claims),
                     sub_claim=sub_claim[:80])

            evidence = await workflow.execute_activity(
                research_subclaim,
                args=[sub_claim],
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            result = await workflow.execute_activity(
                judge_subclaim,
                args=[claim_text, sub_claim, evidence],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            return result

        sub_results = list(await asyncio.gather(
            *[_process_subclaim(i, sc) for i, sc in enumerate(sub_claims)]
        ))

        # Step 3: Synthesize overall verdict
        verdict = await workflow.execute_activity(
            synthesize_verdict,
            args=[claim_text, sub_results],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 4: Store result
        await workflow.execute_activity(
            store_result,
            args=[claim_id, verdict, sub_results],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        log.info(workflow.logger, MODULE, "complete", "Verification complete",
                 claim_id=claim_id, verdict=verdict.get("verdict"),
                 confidence=verdict.get("confidence"))
        return verdict
