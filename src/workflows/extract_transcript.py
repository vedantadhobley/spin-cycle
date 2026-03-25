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
        notify_frontend_refresh,
    )
    from src.workflows.verify import VerifyClaimWorkflow
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

        # 2 concurrent batches to match LLM server parallelism
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
                        self._title,
                    ],
                    # Structured output for 30-40 segments can take 15-25 min on local LLM
                    start_to_close_timeout=timedelta(seconds=1800),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                self._batches_done += 1
                return result

        # Store transcript metadata + speaker enrichment BEFORE batches start
        store_result = await workflow.execute_activity(
            store_transcript,
            args=[transcript_data],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        batch_results = await asyncio.gather(
            *[_run_batch(i, b) for i, b in enumerate(batches)],
            return_exceptions=True,
        )
        self._transcript_id = store_result["transcript_id"]

        # Build speaker → description lookup from Wikidata-enriched speakers
        speaker_descriptions = {}
        for s in store_result.get("speakers", []):
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

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

        finalize_result = await workflow.execute_activity(
            finalize_extraction,
            args=[transcript_data, all_batch_claims],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        claims = finalize_result["worth_checking"]
        all_claims_for_storage = finalize_result["all_claims"]
        # Indices into all_claims_for_storage that survived finalization
        # (filtering + dedup). These map 1:1 to `claims` (worth_checking).
        surviving_indices: list[int] = finalize_result.get("surviving_indices", [])

        self._claim_count = len(claims)
        self._claims = claims

        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        log.info(workflow.logger, MODULE, "finalized",
                 "Claims finalized",
                 worth_checking=len(claims),
                 total_stored=len(all_claims_for_storage))

        # Step 3b: Store ALL extracted claims in DB (linked to transcript)
        if self._transcript_id and all_claims_for_storage:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, all_claims_for_storage],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Use surviving_indices to pick the tc_ids that correspond
            # exactly to the worth_checking claims (post-filter, post-dedup).
            # This avoids the FK cross-wiring bug where dedup removes a claim
            # from worth_checking but not from tc_ids.
            if surviving_indices:
                worth_checking_tc_ids = [tc_ids[i] for i in surviving_indices]
            else:
                # Fallback for older finalize_extraction without surviving_indices
                worth_checking_tc_ids = [
                    tc_id for tc_id, c in zip(tc_ids, all_claims_for_storage)
                    if c.get("worth_checking", True)
                ]

            # Step 4: Batch-create Claim records and link FKs (only for worth_checking)
            self._set_phase("submitting")

            log.info(workflow.logger, MODULE, "creating_claims",
                     "Batch-creating Claim records for verification",
                     claim_count=len(claims),
                     tc_id_count=len(worth_checking_tc_ids))

            claim_ids = await workflow.execute_activity(
                create_claims_for_transcript,
                args=[self._transcript_id, worth_checking_tc_ids, claims,
                      transcript_data.get("date"), self._title,
                      speaker_descriptions],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            self._verification_submitted = len(claim_ids)

            # Step 5: Transition to verifying, run child verification workflows
            await workflow.execute_activity(
                update_transcript_status,
                args=[self._transcript_id, "verifying"],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Notify frontend that extraction is done and claims are ready
            await workflow.execute_activity(
                notify_frontend_refresh,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            self._set_phase("verifying")
            transcript_date = transcript_data.get("date")

            log.info(workflow.logger, MODULE, "verification_started",
                     "Starting sequential child verification workflows",
                     claim_count=len(claim_ids))

            verified = 0
            failed = 0
            for claim_id_str, claim_data in zip(claim_ids, claims):
                try:
                    speaker_name = claim_data.get("speaker")
                    speaker_desc = speaker_descriptions.get(speaker_name, "") if speaker_name else ""
                    await workflow.execute_child_workflow(
                        VerifyClaimWorkflow.run,
                        args=[claim_id_str, claim_data["claim_text"],
                              speaker_name, transcript_date,
                              True,  # is_child
                              self._title,  # transcript_title
                              speaker_desc],  # speaker_description
                        id=f"verify-{claim_id_str}",
                        task_queue="spin-cycle-verify",
                    )
                    verified += 1
                except Exception as e:
                    failed += 1
                    log.warning(workflow.logger, MODULE,
                                "child_verify_failed",
                                "Child verification workflow failed",
                                claim_id=claim_id_str,
                                error=str(e))

            log.info(workflow.logger, MODULE, "verification_done",
                     "All child verifications complete",
                     verified=verified, failed=failed)

            # Mark transcript complete and start next queued transcript
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

        elif self._transcript_id and not claims and all_claims_for_storage:
            # Skipped claims exist but none worth checking — store them, then mark complete
            await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, all_claims_for_storage],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
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

            log.info(workflow.logger, MODULE, "no_worth_checking",
                     "No worth-checking claims, stored skipped claims",
                     stored=len(all_claims_for_storage))

        elif self._transcript_id:
            # No claims at all — mark complete and try next queued transcript
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

        # Notify frontend (fire-and-forget, don't fail workflow)
        await workflow.execute_activity(
            notify_frontend_refresh,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

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
