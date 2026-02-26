"""Temporal workflow for claim verification.

Flat pipeline — no recursion:
  1. Decompose the claim into atomic verifiable facts (one LLM call)
  2. Research all facts (batched, thinking=off — fast)
  3. Judge all facts (batched, thinking=on — slow but separate phase)
  4. Synthesize all verdicts into a final result
  5. Store the result

Research and judge are SEPARATE PHASES to prevent slow judge calls
(thinking=on, generating thousands of tokens) from starving the
faster research agents (thinking=off). This was causing research
timeouts when judge and research overlapped.

This follows the approach used by Google's SAFE (NeurIPS 2024) and FActScore:
extract all facts in one pass, verify each independently, aggregate.

Concurrency is tuned for vLLM on joi. With --parallel set on the unified
Qwen3.5 model, MAX_CONCURRENT=2 gives each research agent a dedicated slot — no
interleaving overhead.
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
        start_next_queued_claim,
    )
    from src.utils.logging import log

MODULE = "workflow"

# Maximum atomic facts to process. If decomposition returns more, we cap it.
# 6 facts × ~4 min each = ~12 min per batch of 2 = ~12 min total. Reasonable.
MAX_FACTS = 6

# Maximum concurrent research+judge pipelines. Matched to --parallel on
# joi's Qwen3.5 instance — each agent gets a dedicated inference slot.
MAX_CONCURRENT = 2


@workflow.defn
class VerifyClaimWorkflow:
    """Orchestrates the full claim verification pipeline.

    Flat pipeline:
    1. Decompose claim into atomic facts (1 LLM call, thinking=off)
    2. For each fact: research with ReAct agent + judge (thinking=on)
    3. Synthesize all sub-verdicts into final verdict (thinking=off)
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

        # Step 1: Decompose — extract atomic facts + thesis in one pass
        decomposition = await workflow.execute_activity(
            decompose_claim,
            args=[claim_text],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        atomic_facts = decomposition["facts"]
        thesis_info = decomposition.get("thesis_info", {})

        # Cap to prevent runaway decompositions
        if len(atomic_facts) > MAX_FACTS:
            log.warning(workflow.logger, MODULE, "facts_capped",
                        "Capping atomic facts",
                        original=len(atomic_facts), capped=MAX_FACTS)
            atomic_facts = atomic_facts[:MAX_FACTS]

        log.info(workflow.logger, MODULE, "decomposed",
                 "Claim decomposed into atomic facts",
                 claim_id=claim_id, fact_count=len(atomic_facts),
                 facts=[f["text"] for f in atomic_facts],
                 thesis=thesis_info.get("thesis"),
                 structure=thesis_info.get("structure"))

        # Step 2: Research all facts first (thinking=off, fast)
        # Then judge all facts (thinking=on, slow)
        # This prevents slow judge calls from starving faster research agents.
        
        async def _research(fact_text: str) -> tuple[str, list]:
            """Research a single fact, return (fact_text, evidence)."""
            log.info(workflow.logger, MODULE, "research_start",
                     "Researching fact",
                     claim_id=claim_id, fact=fact_text)
            evidence = await workflow.execute_activity(
                research_subclaim,
                args=[fact_text],
                start_to_close_timeout=timedelta(seconds=180),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            return (fact_text, evidence)
        
        async def _judge(fact_text: str, evidence: list) -> dict:
            """Judge a single fact given its evidence."""
            return await workflow.execute_activity(
                judge_subclaim,
                args=[claim_text, fact_text, evidence],
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        # Phase 1: Research all facts in batches
        all_evidence = []  # list of (fact_text, evidence)
        for i in range(0, len(atomic_facts), MAX_CONCURRENT):
            batch = atomic_facts[i:i + MAX_CONCURRENT]
            batch_num = i // MAX_CONCURRENT + 1
            log.info(workflow.logger, MODULE, "research_batch_start",
                     "Starting research batch",
                     claim_id=claim_id, batch_num=batch_num, batch_size=len(batch))

            _t0 = workflow.time()
            batch_evidence = list(await asyncio.gather(
                *[_research(fact["text"]) for fact in batch]
            ))
            _batch_ms = round((workflow.time() - _t0) * 1000)
            log.info(workflow.logger, MODULE, "research_batch_done",
                     "Research batch completed",
                     claim_id=claim_id, batch_num=batch_num, latency_ms=_batch_ms)
            all_evidence.extend(batch_evidence)

        # Phase 2: Judge all facts in batches (now no research to compete with)
        sub_results = []
        for i in range(0, len(all_evidence), MAX_CONCURRENT):
            batch = all_evidence[i:i + MAX_CONCURRENT]
            batch_num = i // MAX_CONCURRENT + 1
            log.info(workflow.logger, MODULE, "judge_batch_start",
                     "Starting judge batch",
                     claim_id=claim_id, batch_num=batch_num, batch_size=len(batch))

            _t0 = workflow.time()
            batch_results = list(await asyncio.gather(
                *[_judge(fact_text, evidence) for fact_text, evidence in batch]
            ))
            _batch_ms = round((workflow.time() - _t0) * 1000)
            log.info(workflow.logger, MODULE, "judge_batch_done",
                     "Judge batch completed",
                     claim_id=claim_id, batch_num=batch_num, latency_ms=_batch_ms)
            sub_results.extend(batch_results)

        # Step 3: Synthesize all verdicts into final result
        if len(sub_results) == 1:
            # Single fact — no synthesis needed, just use the verdict directly
            result = sub_results[0]
        else:
            result = await workflow.execute_activity(
                synthesize_verdict,
                args=[claim_text, claim_text, sub_results, True, thesis_info],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        log.info(workflow.logger, MODULE, "verdict",
                 "Final verdict reached",
                 claim_id=claim_id,
                 verdict=result.get("verdict"),
                 confidence=result.get("confidence"),
                 fact_count=len(atomic_facts))

        # Step 4: Store the result
        await workflow.execute_activity(
            store_result,
            args=[claim_id, result],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 5: Start next queued claim (if any)
        # This ensures claims process sequentially - one at a time
        await workflow.execute_activity(
            start_next_queued_claim,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        log.info(workflow.logger, MODULE, "complete", "Verification complete",
                 claim_id=claim_id, verdict=result.get("verdict"),
                 confidence=result.get("confidence"))
        return result
