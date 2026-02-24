"""Temporal workflow for claim verification.

Processes claims recursively:
  - Each text is decomposed into sub-parts (one level at a time)
  - Atomic parts (single items) are researched + judged
  - Compound parts are further decomposed, processed, and synthesized
  - The tree structure emerges naturally from recursion
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
        store_result,
    )
    from src.utils.logging import log

MODULE = "workflow"

# Maximum recursion depth for decomposition. At MAX_DEPTH, items are forced
# to be leaves (researched + judged directly) regardless of complexity.
# depth=0 is the original claim, depth=1 is first split, etc.
MAX_DEPTH = 3


@workflow.defn
class VerifyClaimWorkflow:
    """Orchestrates the full claim verification pipeline.

    The workflow is recursive:
    1. Decompose the text into sub-parts (one level)
    2. For each sub-part:
       - If atomic (1 item): research + judge as a leaf
       - If compound (2+ items): recurse into each, then synthesize
    3. The tree emerges from recursion — depth adapts to claim complexity
    4. Store the full result tree in the database
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

        # Recursive processing function — decomposes, researches, judges, synthesizes
        async def _process(node_text: str, depth: int) -> dict:
            """Recursively process a claim or sub-claim.

            At each level:
            1. Decompose the text into sub-parts
            2. If atomic (single item returned): research + judge
            3. If compound (multiple items): recurse on each, then synthesize
            """
            # Decompose this node (unless at max depth)
            if depth < MAX_DEPTH:
                sub_nodes = await workflow.execute_activity(
                    decompose_claim,
                    args=[node_text],
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
            else:
                # Max depth — force as atomic leaf
                sub_nodes = [{"text": node_text}]

            log.info(workflow.logger, MODULE, "decomposed",
                     "Node decomposed",
                     claim_id=claim_id, node=node_text,
                     depth=depth, sub_count=len(sub_nodes))

            if len(sub_nodes) <= 1:
                # ATOMIC — research + judge as a leaf
                leaf_text = sub_nodes[0]["text"] if sub_nodes else node_text

                log.info(workflow.logger, MODULE, "leaf_start",
                         "Processing leaf sub-claim",
                         claim_id=claim_id, sub_claim=leaf_text, depth=depth)

                evidence = await workflow.execute_activity(
                    research_subclaim,
                    args=[leaf_text],
                    start_to_close_timeout=timedelta(seconds=360),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )

                result = await workflow.execute_activity(
                    judge_subclaim,
                    args=[claim_text, leaf_text, evidence],
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                return result

            # COMPOUND — process children in parallel, then synthesize
            log.info(workflow.logger, MODULE, "branch",
                     "Processing compound node",
                     claim_id=claim_id, node=node_text,
                     depth=depth, num_children=len(sub_nodes))

            child_results = list(await asyncio.gather(
                *[_process(child["text"], depth + 1) for child in sub_nodes]
            ))

            # Synthesize children into a single verdict for this node
            is_final = (depth == 0)
            synthesis_result = await workflow.execute_activity(
                synthesize_verdict,
                args=[claim_text, node_text, child_results, is_final],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            log.info(workflow.logger, MODULE, "synthesized",
                     "Node synthesized",
                     claim_id=claim_id, node=node_text,
                     depth=depth, is_final=is_final,
                     verdict=synthesis_result.get("verdict"))

            return synthesis_result

        # Process the claim recursively starting at depth 0
        result = await _process(claim_text, depth=0)

        # Store the full result tree
        await workflow.execute_activity(
            store_result,
            args=[claim_id, result],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        log.info(workflow.logger, MODULE, "complete", "Verification complete",
                 claim_id=claim_id, verdict=result.get("verdict"),
                 confidence=result.get("confidence"))
        return result
