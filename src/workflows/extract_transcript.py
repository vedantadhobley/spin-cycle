"""Temporal workflow for transcript claim extraction.

Takes a transcript URL, fetches it, extracts verifiable claims via per-batch
activities, and optionally submits them to the verification pipeline.

Each batch of ~30 segments is a separate Temporal activity — visible in the
UI with its own timing, retries, and status.  Temporal's max_concurrent_activities
on the worker naturally limits GPU contention (no internal semaphore needed).

Phases visible in Temporal UI:
  1. fetching    — downloading and parsing the transcript
  2. extracting  — N batch activities running (with overlap context)
  3. finalizing  — dedup + filter across batches
  4. submitting  — creating claim records for verification (if auto_verify=True)
  5. complete    — done
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        fetch_transcript,
        extract_transcript_batch,
        finalize_extraction,
        store_transcript,
        store_transcript_claims,
        create_claims_for_transcript,
        update_transcript_status,
        finish_transcript_and_start_next,
    )
    from src.activities.verify_activities import (
        start_next_queued_claim,
    )
    from src.transcript.extractor import build_batches
    from src.utils.logging import log

MODULE = "transcript_workflow"

# Search attribute keys for Temporal UI visibility
SA_PHASE = SearchAttributeKey.for_keyword("Phase")
SA_CLAIM_COUNT = SearchAttributeKey.for_int("ClaimCount")
SA_TRANSCRIPT_TITLE = SearchAttributeKey.for_keyword("TranscriptTitle")


@workflow.defn
class ExtractTranscriptWorkflow:
    """Extract verifiable claims from a transcript URL.

    Phases:
    1. Fetch and parse the transcript
    2. Extract claims in parallel batches (each a visible Temporal activity)
    3. Finalize — dedup + filter across batch boundaries
    4. Optionally submit claims to verification pipeline
    """

    def __init__(self) -> None:
        self._phase = "initializing"
        self._url = ""
        self._title = ""
        self._word_count = 0
        self._segment_count = 0
        self._speakers: list[str] = []
        self._batch_count = 0
        self._batches_done = 0
        self._batches_failed = 0
        self._claim_count = 0
        self._claims: list[dict] = []
        self._transcript_id = ""
        self._verification_submitted = 0

    @workflow.query
    def status(self) -> dict:
        """Current workflow state — queryable from Temporal UI."""
        return {
            "phase": self._phase,
            "url": self._url,
            "title": self._title,
            "word_count": self._word_count,
            "segment_count": self._segment_count,
            "speakers": self._speakers,
            "batches": {
                "total": self._batch_count,
                "done": self._batches_done,
                "failed": self._batches_failed,
            },
            "claim_count": self._claim_count,
            "claims": self._claims,
            "transcript_id": self._transcript_id,
            "verification_submitted": self._verification_submitted,
        }

    def _set_phase(self, phase: str) -> None:
        """Update phase in state and search attributes."""
        self._phase = phase
        workflow.upsert_search_attributes([SA_PHASE.value_set(phase)])

    @workflow.run
    async def run(self, url: str) -> dict:
        """Run the transcript extraction pipeline.

        After extraction, batch-creates Claim records, links FKs, and kicks
        off the verification pipeline. Always verifies — no opt-out.

        Args:
            url: Rev.com transcript URL.

        Returns:
            Dict with transcript metadata and extracted claims.
        """
        self._url = url

        log.info(workflow.logger, MODULE, "started",
                 "Starting transcript extraction",
                 url=url)

        # Step 1: Fetch and parse transcript
        self._set_phase("fetching")

        transcript_data = await workflow.execute_activity(
            fetch_transcript,
            args=[url],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        self._title = transcript_data["title"]
        self._word_count = transcript_data["word_count"]
        self._segment_count = len(transcript_data["segments"])
        self._speakers = transcript_data["speakers"]

        workflow.upsert_search_attributes([
            SA_TRANSCRIPT_TITLE.value_set(self._title),
        ])

        log.info(workflow.logger, MODULE, "fetched",
                 "Transcript fetched",
                 title=self._title,
                 word_count=self._word_count,
                 segment_count=self._segment_count,
                 speakers=self._speakers)

        # Step 2: Build batch specs and extract in parallel
        self._set_phase("extracting")

        segment_word_counts = [
            len(seg["text"].split()) for seg in transcript_data["segments"]
        ]
        batches = build_batches(segment_word_counts)
        self._batch_count = len(batches)

        log.info(workflow.logger, MODULE, "batching",
                 "Extraction batches planned",
                 batch_count=self._batch_count,
                 segment_count=self._segment_count)

        # Sliding window: 2 concurrent batches to match LLM server parallelism
        MAX_CONCURRENT_BATCHES = 2
        batch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

        async def _run_batch(i: int, batch: dict) -> list[dict]:
            async with batch_semaphore:
                label = f"Batch {i+1} of {self._batch_count}"
                result = await workflow.execute_activity(
                    extract_transcript_batch,
                    args=[
                        transcript_data,
                        batch["target_start"],
                        batch["target_end"],
                        batch["text_start"],
                        batch["text_end"],
                        label,
                    ],
                    # Structured output for 30 segments can take 15+ min on local LLM
                    start_to_close_timeout=timedelta(seconds=1200),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
                self._batches_done += 1
                return result

        # Store cleaned transcript in DB (runs in parallel with extraction)
        store_task = asyncio.ensure_future(
            workflow.execute_activity(
                store_transcript,
                args=[transcript_data],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        )

        batch_results = await asyncio.gather(
            *[_run_batch(i, b) for i, b in enumerate(batches)],
            return_exceptions=True,
        )

        # Await store (should be done long before batches finish)
        self._transcript_id = await store_task

        # Collect results, log failures
        all_batch_claims: list[list[dict]] = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self._batches_failed += 1
                log.warning(workflow.logger, MODULE, "batch_failed",
                            f"Batch {i+1} failed",
                            error=str(result))
            else:
                all_batch_claims.append(result)

        log.info(workflow.logger, MODULE, "extraction_done",
                 "All batches complete",
                 succeeded=len(all_batch_claims),
                 failed=self._batches_failed)

        # Step 3: Finalize — dedup + filter across batches
        self._set_phase("finalizing")

        claims = await workflow.execute_activity(
            finalize_extraction,
            args=[transcript_data, all_batch_claims],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        self._claim_count = len(claims)
        self._claims = claims

        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        log.info(workflow.logger, MODULE, "finalized",
                 "Claims finalized",
                 claim_count=self._claim_count)

        # Step 3b: Store extracted claims in DB (linked to transcript)
        if self._transcript_id and claims:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, claims],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Step 4: Batch-create Claim records and link FKs
            self._set_phase("submitting")

            log.info(workflow.logger, MODULE, "creating_claims",
                     "Batch-creating Claim records for verification",
                     claim_count=len(claims))

            claim_ids = await workflow.execute_activity(
                create_claims_for_transcript,
                args=[self._transcript_id, tc_ids, claims],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            self._verification_submitted = len(claim_ids)

            # Step 5: Transition to verifying and start first claim
            await workflow.execute_activity(
                update_transcript_status,
                args=[self._transcript_id, "verifying"],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            await workflow.execute_activity(
                start_next_queued_claim,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

            log.info(workflow.logger, MODULE, "verification_started",
                     "Verification pipeline started",
                     submitted=self._verification_submitted)

        elif self._transcript_id:
            # No claims extracted — mark complete and try next queued transcript
            await workflow.execute_activity(
                update_transcript_status,
                args=[self._transcript_id, "complete"],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            await workflow.execute_activity(
                finish_transcript_and_start_next,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

            log.info(workflow.logger, MODULE, "no_claims",
                     "No claims extracted, transcript marked complete")

        # Done
        self._set_phase("complete")

        result = {
            "url": url,
            "title": self._title,
            "word_count": self._word_count,
            "segment_count": self._segment_count,
            "speakers": self._speakers,
            "claim_count": self._claim_count,
            "claims": claims,
            "transcript_id": self._transcript_id,
            "verification_submitted": self._verification_submitted,
        }

        log.info(workflow.logger, MODULE, "complete",
                 "Transcript extraction complete",
                 title=self._title,
                 claim_count=self._claim_count)

        return result
