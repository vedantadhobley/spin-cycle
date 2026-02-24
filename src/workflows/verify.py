"""Temporal workflow for claim verification.

Processes claims as a tree of sub-claims:
  - Leaves are researched + judged in parallel
  - Group nodes synthesize their children's results
  - Final synthesis combines top-level results into overall verdict
"""

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
        synthesize_group,
        store_result,
    )
    from src.utils.logging import log

MODULE = "workflow"


@workflow.defn
class VerifyClaimWorkflow:
    """Orchestrates the full claim verification pipeline.

    Steps:
    1. Decompose claim into a tree of sub-claims
    2. Walk tree: leaves get researched+judged, groups synthesize children
    3. Synthesize overall verdict from top-level results
    4. Store result in database
    """

    @workflow.run
    async def run(self, claim_id: str | None, claim_text: str) -> dict:
        """Run the verification pipeline.

        Args:
            claim_id: Existing claim UUID from the database, or None.
                      When None, the workflow creates the claim record itself.
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
                 claim_id=claim_id, claim=claim_text)

        # Step 1: Decompose into sub-claim tree
        tree = await workflow.execute_activity(
            decompose_claim,
            args=[claim_text],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        log.info(workflow.logger, MODULE, "decomposed", "Claim decomposed into tree",
                 claim_id=claim_id, top_level_nodes=len(tree))

        # Step 2: Process tree recursively — leaves in parallel, groups synthesize
        async def _process_node(node: dict) -> dict:
            """Process a single tree node.

            Leaf: research + judge → returns sub-result dict
            Group: process children in parallel, then synthesize → returns sub-result dict
            """
            if "children" in node:
                # GROUP NODE — process children in parallel, then synthesize
                label = node.get("label", "Group")
                log.info(workflow.logger, MODULE, "group_start",
                         "Processing group node",
                         claim_id=claim_id, label=label,
                         num_children=len(node["children"]))

                child_results = list(await asyncio.gather(
                    *[_process_node(child) for child in node["children"]]
                ))

                # Synthesize children into intermediate verdict
                group_result = await workflow.execute_activity(
                    synthesize_group,
                    args=[claim_text, label, child_results],
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )

                log.info(workflow.logger, MODULE, "group_done",
                         "Group node synthesized",
                         claim_id=claim_id, label=label,
                         verdict=group_result.get("verdict"))

                return group_result
            else:
                # LEAF NODE — research + judge
                sub_claim = node["text"]
                log.info(workflow.logger, MODULE, "leaf_start",
                         "Processing leaf sub-claim",
                         claim_id=claim_id, sub_claim=sub_claim)

                evidence = await workflow.execute_activity(
                    research_subclaim,
                    args=[sub_claim],
                    start_to_close_timeout=timedelta(seconds=360),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )

                result = await workflow.execute_activity(
                    judge_subclaim,
                    args=[claim_text, sub_claim, evidence],
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                return result

        # Process all top-level nodes in parallel
        top_results = list(await asyncio.gather(
            *[_process_node(node) for node in tree]
        ))

        # Step 3: Synthesize overall verdict
        verdict = await workflow.execute_activity(
            synthesize_verdict,
            args=[claim_text, top_results],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 4: Store result (pass the tree for structure-aware storage)
        await workflow.execute_activity(
            store_result,
            args=[claim_id, verdict, top_results, tree],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        log.info(workflow.logger, MODULE, "complete", "Verification complete",
                 claim_id=claim_id, verdict=verdict.get("verdict"),
                 confidence=verdict.get("confidence"))
        return verdict
