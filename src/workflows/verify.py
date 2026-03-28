"""Temporal workflow for claim verification.

Flat pipeline — no recursion:
  0. Create claim record (if not already in DB)
  1. Decompose claim into atomic facts (2 LLM calls: normalize + extract)
     — each fact gets categories, seed_queries from the LLM
     — interested parties expanded via Wikidata (programmatic)
  2. Research all facts (separate phase)
     — full interested_parties dict passed for conflict detection
     — MBFC ownership → Wikidata enrichment (overlap-gated)
     — evidence NER → Wikidata enrichment (overlap-gated)
     — enriched parties merged across sub-claims for judge phase
  3. Judge all facts (separate phase)
     — receives merged interested parties from all research sub-claims
  4. Synthesize all verdicts into a final result
  5. Store result + start next queued claim

Research and judge are SEPARATE PHASES to prevent longer judge calls
(structured rubric evaluation) from starving the faster research agents.

Follows Google's SAFE (NeurIPS 2024) and FActScore:
extract all facts in one pass, verify each independently, aggregate.

Concurrency is tuned for the local LLM server. MAX_CONCURRENT=2 gives
each research agent a dedicated inference slot (2 × 65K context).
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey

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
    from src.activities.transcript_activities import (
        finish_transcript_and_start_next,
        notify_frontend_refresh,
    )
    from src.utils.logging import log

MODULE = "workflow"

# Maximum atomic facts to process. If decomposition returns more, we cap it.
# 10 facts × ~4 min each ÷ 2 concurrent = ~20 min total.
MAX_FACTS = 10

# Maximum concurrent research+judge pipelines. Matched to --parallel 2 on
# the LLM server — 2 slots × 65K context each (131K total ctx-size).
MAX_CONCURRENT = 2

# Search attribute keys for Temporal UI visibility
SA_PHASE = SearchAttributeKey.for_keyword("Phase")
SA_FACT_COUNT = SearchAttributeKey.for_int("FactCount")
SA_RESEARCH_PROGRESS = SearchAttributeKey.for_keyword("ResearchProgress")
SA_JUDGE_PROGRESS = SearchAttributeKey.for_keyword("JudgeProgress")
SA_VERDICT = SearchAttributeKey.for_keyword("Verdict")
SA_CONFIDENCE = SearchAttributeKey.for_float("Confidence")


@workflow.defn
class VerifyClaimWorkflow:
    """Orchestrates the full claim verification pipeline.

    Flat pipeline with separate research and judge phases:
    1. Decompose claim into atomic facts (2 LLM calls + Wikidata expansion)
    2. Research all facts (Phase 1: seed search + rank, Phase 2: ReAct agent)
    3. Judge all facts (evidence annotation + LLM verdict)
    4. Synthesize all sub-verdicts into final verdict
    5. Store result in database + start next queued claim
    """

    def __init__(self) -> None:
        # Workflow state — exposed via @workflow.query for Temporal UI
        self._phase = "initializing"
        self._claim_text = ""
        self._fact_count = 0
        self._facts: list[str] = []
        self._thesis = ""
        self._interested_parties_count = 0
        self._research_done = 0
        self._research_failed = 0
        self._evidence_counts: dict[str, int] = {}
        self._judge_done = 0
        self._judge_failed = 0
        self._sub_verdicts: list[dict] = []
        self._verdict = ""
        self._confidence = 0.0

    @workflow.query
    def status(self) -> dict:
        """Current workflow state — queryable from Temporal UI."""
        return {
            "phase": self._phase,
            "claim": self._claim_text,
            "fact_count": self._fact_count,
            "facts": self._facts,
            "thesis": self._thesis,
            "interested_parties": self._interested_parties_count,
            "research": {
                "done": self._research_done,
                "failed": self._research_failed,
                "total": self._fact_count,
                "evidence_per_fact": self._evidence_counts,
            },
            "judge": {
                "done": self._judge_done,
                "failed": self._judge_failed,
                "total": self._fact_count,
                "sub_verdicts": self._sub_verdicts,
            },
            "verdict": self._verdict,
            "confidence": self._confidence,
        }

    def _set_phase(self, phase: str) -> None:
        """Update phase in state and search attributes."""
        self._phase = phase
        workflow.upsert_search_attributes([SA_PHASE.value_set(phase)])

    @workflow.run
    async def run(self, claim_id: str | None, claim_text: str,
                  speaker: str | None = None,
                  claim_date: str | None = None,
                  is_child: bool = False,
                  transcript_title: str | None = None,
                  speaker_description: str = "",
                  supporting_quotes: list[str] | None = None) -> dict:
        """Run the verification pipeline.

        Args:
            claim_id: Existing claim UUID from the database, or None.
                      When None, the workflow creates the claim record itself.
            claim_text: The claim to verify.
            speaker: Optional name of the person making the claim.
                     Automatically added as an interested party.
            claim_date: When the claim was made (e.g. transcript date).
                       Used to anchor temporal references like "36 hours ago".
            is_child: True when started as a child of ExtractTranscriptWorkflow.
                      Skips queue chaining (parent orchestrates sequentially).
            transcript_title: Title of the source transcript (e.g. "Iran
                            conflict update"). Provides topic context for
                            research and judgment.
            speaker_description: Wikidata description of the speaker (e.g.
                               "45th and 47th president of the United States").
                               Passed from transcript extraction to avoid
                               redundant Wikidata lookups.
            supporting_quotes: Optional list of supporting quote texts from
                             transcript segments (thesis extraction v2).
                             Passed to decompose for richer argument context.
        """
        self._claim_text = claim_text

        # Step 0: Create claim record if we don't have one
        if not claim_id:
            self._set_phase("creating_claim")
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

        # Step 1: Normalize + Decompose
        self._set_phase("decomposing")

        decomposition = await workflow.execute_activity(
            decompose_claim,
            args=[claim_text, speaker, claim_date, transcript_title,
                  speaker_description, supporting_quotes],
            start_to_close_timeout=timedelta(seconds=180),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        atomic_facts = decomposition["facts"]
        thesis_info = decomposition.get("thesis_info", {})

        # Build enriched speaker line for downstream prompts.
        # Prefer description from transcript extraction (already resolved),
        # fall back to decompose's Wikidata lookup.
        speaker_desc = speaker_description or decomposition.get("speaker_description", "")
        if speaker and speaker_desc:
            speaker_context = f"{speaker} ({speaker_desc})"
        else:
            speaker_context = speaker

        interested_parties = thesis_info.get("interested_parties", {})
        if isinstance(interested_parties, list):
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

        # Update state after decompose
        self._fact_count = len(atomic_facts)
        self._facts = [f["text"] for f in atomic_facts]
        self._thesis = thesis_info.get("thesis", "")
        self._interested_parties_count = len(
            interested_parties.get("all_parties", [])
        )
        workflow.upsert_search_attributes([
            SA_FACT_COUNT.value_set(self._fact_count),
        ])

        log.info(workflow.logger, MODULE, "decomposed",
                 "Claim decomposed into atomic facts",
                 claim_id=claim_id, fact_count=len(atomic_facts),
                 facts=[f["text"] for f in atomic_facts],
                 thesis=thesis_info.get("thesis"),
                 structure=thesis_info.get("structure"),
                 interested_parties=interested_parties)

        # Build verification_target lookup for judge phase
        verification_targets = {
            fact["text"]: fact.get("verification_target", "")
            for fact in atomic_facts
        }

        # Key test: the overarching "what must be true" for the claim.
        # Passed to every judge call so each sub-judgment is anchored
        # to the core question (especially critical for single-fact claims
        # that skip synthesis entirely).
        key_test = thesis_info.get("key_test", "")

        # Step 2: Research all facts (thinking=off, fast)
        self._set_phase("researching")

        async def _research(fact: dict) -> tuple[str, list, dict]:
            """Research a single fact, return (fact_text, evidence, enriched_parties)."""
            fact_text = fact["text"]
            fact_categories = fact.get("categories", ["GENERAL"])
            fact_seed_queries = fact.get("seed_queries", [])
            log.info(workflow.logger, MODULE, "research_start",
                     "Researching fact",
                     claim_id=claim_id, fact=fact_text,
                     categories=fact_categories,
                     seed_query_count=len(fact_seed_queries))
            result = await workflow.execute_activity(
                research_subclaim,
                args=[fact_text, interested_parties,
                      fact_categories, fact_seed_queries, speaker_context,
                      claim_date, claim_text, transcript_title],
                start_to_close_timeout=timedelta(seconds=540),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            # Update progress
            ev_count = len(result.get("evidence", []))
            self._research_done += 1
            self._evidence_counts[fact_text] = ev_count
            workflow.upsert_search_attributes([
                SA_RESEARCH_PROGRESS.value_set(
                    f"{self._research_done}/{self._fact_count}",
                ),
            ])
            return (fact_text, result["evidence"], result.get("enriched_parties", {}))

        # Sliding window concurrency
        research_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def _research_with_limit(fact: dict) -> tuple[str, list, dict]:
            async with research_semaphore:
                return await _research(fact)

        log.info(workflow.logger, MODULE, "research_phase_start",
                 "Starting research phase with sliding window",
                 claim_id=claim_id, fact_count=len(atomic_facts),
                 max_concurrent=MAX_CONCURRENT)

        _t0 = workflow.time()
        research_results = await asyncio.gather(
            *[_research_with_limit(fact) for fact in atomic_facts],
            return_exceptions=True
        )

        # Filter out failures and merge enriched parties
        all_evidence = []
        merged_all_parties = set(interested_parties.get("all_parties", []))
        merged_affiliated_media = set(interested_parties.get("affiliated_media", []))

        for i, result in enumerate(research_results):
            if isinstance(result, Exception):
                log.warning(workflow.logger, MODULE, "research_failed",
                            "Research failed for fact, skipping",
                            claim_id=claim_id, fact=atomic_facts[i]["text"],
                            error=str(result))
                self._research_failed += 1
            else:
                fact_text, evidence, enriched = result
                all_evidence.append((fact_text, evidence))
                if enriched:
                    merged_all_parties.update(enriched.get("all_parties", []))
                    merged_affiliated_media.update(enriched.get("affiliated_media", []))

        # Build merged interested parties for the judge phase
        MAX_ALL_PARTIES = 40
        merged_parties_list = list(merged_all_parties)
        if len(merged_parties_list) > MAX_ALL_PARTIES:
            log.warning(workflow.logger, MODULE, "parties_capped",
                        "Capping merged all_parties to prevent explosion",
                        before=len(merged_parties_list), after=MAX_ALL_PARTIES)
            merged_parties_list = merged_parties_list[:MAX_ALL_PARTIES]
        merged_parties = dict(interested_parties)
        merged_parties["all_parties"] = merged_parties_list
        merged_parties["affiliated_media"] = list(merged_affiliated_media)

        self._interested_parties_count = len(merged_parties_list)

        _research_ms = round((workflow.time() - _t0) * 1000)
        log.info(workflow.logger, MODULE, "research_phase_done",
                 "Research phase completed",
                 claim_id=claim_id, latency_ms=_research_ms,
                 succeeded=len(all_evidence), failed=self._research_failed,
                 merged_parties=len(merged_all_parties),
                 merged_media=len(merged_affiliated_media))

        # Step 3: Judge all facts (thinking=on, slow)
        self._set_phase("judging")

        async def _judge(fact_text: str, evidence: list,
                         merged_p: dict) -> dict:
            """Judge a single fact given its evidence."""
            vt = verification_targets.get(fact_text, "")
            result = await workflow.execute_activity(
                judge_subclaim,
                args=[claim_text, fact_text, evidence, merged_p, speaker_context,
                      claim_date, vt, transcript_title, key_test],
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            # Update progress
            self._judge_done += 1
            self._sub_verdicts.append({
                "fact": fact_text,
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
                "evidence_count": len(result.get("evidence", [])),
                "citations": len(result.get("citations", [])),
            })
            workflow.upsert_search_attributes([
                SA_JUDGE_PROGRESS.value_set(
                    f"{self._judge_done}/{self._fact_count}",
                ),
            ])
            return result

        judge_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def _judge_with_limit(fact_text: str, evidence: list) -> dict:
            async with judge_semaphore:
                return await _judge(fact_text, evidence, merged_parties)

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
        for i, result in enumerate(judge_results):
            if isinstance(result, Exception):
                fact_text, _ = all_evidence[i]
                log.warning(workflow.logger, MODULE, "judge_failed",
                            "Judge failed for fact, skipping",
                            claim_id=claim_id, fact=fact_text,
                            error=str(result))
                self._judge_failed += 1
            else:
                sub_results.append(result)

        _judge_ms = round((workflow.time() - _t0) * 1000)
        log.info(workflow.logger, MODULE, "judge_phase_done",
                 "Judge phase completed",
                 claim_id=claim_id, latency_ms=_judge_ms,
                 succeeded=len(sub_results), failed=self._judge_failed)

        # Step 4: Synthesize all verdicts into final result
        self._set_phase("synthesizing")

        if len(sub_results) == 1:
            log.info(workflow.logger, MODULE, "single_fact_skip",
                     "Single fact — skipping synthesis, using judge result directly",
                     claim_id=claim_id)
            result = sub_results[0]
        else:
            result = await workflow.execute_activity(
                synthesize_verdict,
                args=[claim_text, sub_results, thesis_info, claim_date,
                      transcript_title],
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        self._verdict = result.get("verdict", "")
        self._confidence = result.get("confidence", 0.0)

        log.info(workflow.logger, MODULE, "verdict",
                 "Final verdict reached",
                 claim_id=claim_id,
                 verdict=result.get("verdict"),
                 confidence=result.get("confidence"),
                 fact_count=len(atomic_facts))

        # Step 5: Store the result
        self._set_phase("storing")

        await workflow.execute_activity(
            store_result,
            args=[claim_id, result, thesis_info, atomic_facts],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 6: Queue chaining (only for standalone workflows, not children)
        if not is_child:
            next_claim = await workflow.execute_activity(
                start_next_queued_claim,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

            # No more queued claims — check if transcript is done
            if next_claim is None:
                await workflow.execute_activity(
                    finish_transcript_and_start_next,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )

        # Notify frontend (fire-and-forget, don't fail workflow)
        await workflow.execute_activity(
            notify_frontend_refresh,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        self._set_phase("complete")
        workflow.upsert_search_attributes([
            SA_VERDICT.value_set(self._verdict),
            SA_CONFIDENCE.value_set(self._confidence),
        ])

        log.info(workflow.logger, MODULE, "complete", "Verification complete",
                 claim_id=claim_id, verdict=result.get("verdict"),
                 confidence=result.get("confidence"))
        return result
