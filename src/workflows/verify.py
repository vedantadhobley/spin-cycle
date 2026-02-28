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
MAX_FACTS = 10

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
        
        # interested_parties is now an object with all_parties, direct, institutional, affiliated_media
        interested_parties = thesis_info.get("interested_parties", {})
        if isinstance(interested_parties, list):
            # Legacy format - convert to new structure
            interested_parties = {
                "all_parties": interested_parties,
                "direct": interested_parties,
                "institutional": [],
                "affiliated_media": [],
                "reasoning": None,
            }

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
                 structure=thesis_info.get("structure"),
                 interested_parties=interested_parties)

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
                args=[claim_text, fact_text, evidence, interested_parties],
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        # Phase 1: Research all facts with sliding window concurrency
        # Uses semaphore to maintain MAX_CONCURRENT tasks at all times.
        # As soon as one finishes, the next starts immediately (no batch waiting).
        research_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def _research_with_limit(fact_text: str) -> tuple[str, list]:
            async with research_semaphore:
                return await _research(fact_text)
        
        log.info(workflow.logger, MODULE, "research_phase_start",
                 "Starting research phase with sliding window",
                 claim_id=claim_id, fact_count=len(atomic_facts), 
                 max_concurrent=MAX_CONCURRENT)
        
        _t0 = workflow.time()
        research_results = await asyncio.gather(
            *[_research_with_limit(fact["text"]) for fact in atomic_facts],
            return_exceptions=True
        )
        
        # Filter out failures
        all_evidence = []
        research_failures = 0
        for i, result in enumerate(research_results):
            if isinstance(result, Exception):
                log.warning(workflow.logger, MODULE, "research_failed",
                            "Research failed for fact, skipping",
                            claim_id=claim_id, fact=atomic_facts[i]["text"],
                            error=str(result))
                research_failures += 1
            else:
                all_evidence.append(result)
        
        _research_ms = round((workflow.time() - _t0) * 1000)
        log.info(workflow.logger, MODULE, "research_phase_done",
                 "Research phase completed",
                 claim_id=claim_id, latency_ms=_research_ms,
                 succeeded=len(all_evidence), failed=research_failures)

        # Phase 2: Judge all facts with sliding window concurrency
        judge_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def _judge_with_limit(fact_text: str, evidence: list) -> dict:
            async with judge_semaphore:
                return await _judge(fact_text, evidence)
        
        log.info(workflow.logger, MODULE, "judge_phase_start",
                 "Starting judge phase with sliding window",
                 claim_id=claim_id, evidence_count=len(all_evidence),
                 max_concurrent=MAX_CONCURRENT)
        
        _t0 = workflow.time()
        judge_results = await asyncio.gather(
            *[_judge_with_limit(fact_text, evidence) for fact_text, evidence in all_evidence],
            return_exceptions=True
        )
        
        # Filter out failures
        sub_results = []
        judge_failures = 0
        for i, result in enumerate(judge_results):
            if isinstance(result, Exception):
                fact_text, _ = all_evidence[i]
                log.warning(workflow.logger, MODULE, "judge_failed",
                            "Judge failed for fact, skipping",
                            claim_id=claim_id, fact=fact_text,
                            error=str(result))
                judge_failures += 1
            else:
                sub_results.append(result)
        
        _judge_ms = round((workflow.time() - _t0) * 1000)
        log.info(workflow.logger, MODULE, "judge_phase_done",
                 "Judge phase completed",
                 claim_id=claim_id, latency_ms=_judge_ms,
                 succeeded=len(sub_results), failed=judge_failures)

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
