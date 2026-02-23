"""Temporal workflow for claim verification."""

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
            workflow.logger.info(f"Created claim record: {claim_id}")

        workflow.logger.info(f"Starting verification for claim: {claim_id}")

        # Step 1: Decompose into sub-claims
        sub_claims = await workflow.execute_activity(
            decompose_claim,
            args=[claim_text],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        workflow.logger.info(f"Decomposed into {len(sub_claims)} sub-claims")

        # Step 2: Research + judge each sub-claim
        sub_results = []
        for i, sub_claim in enumerate(sub_claims):
            workflow.logger.info(f"Processing sub-claim {i+1}/{len(sub_claims)}: {sub_claim}")

            # Research evidence (LangGraph agent)
            evidence = await workflow.execute_activity(
                research_subclaim,
                args=[sub_claim],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Judge based on evidence
            result = await workflow.execute_activity(
                judge_subclaim,
                args=[sub_claim, evidence],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            sub_results.append(result)

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

        workflow.logger.info(f"Verification complete for claim {claim_id}: {verdict.get('verdict')}")
        return verdict
