"""Temporal workflow for transcript claim extraction.

Supports two extraction modes:
  - v2 (thesis): Single LLM pass over full transcript → 15-30 theses with
    segment references → decompose each → verify. No batching, no semaphore.
  - v1 (batch): Legacy per-segment batch extraction (still available, not
    default). Kept for backward compatibility.

Phases visible in Temporal UI (v2):
  1. fetching           — downloading and parsing the transcript
  2. extracting_theses  — single LLM call over full transcript
  3. storing            — persisting transcript + theses to DB
  4. submitting         — creating claim records for verification
  5. verifying          — running child verification workflows
  6. complete           — done
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        fetch_transcript,
        fetch_raw_transcript,
        extract_theses_activity,
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
    from src.utils.logging import log

MODULE = "transcript_workflow"

# Search attribute keys for Temporal UI visibility
SA_PHASE = SearchAttributeKey.for_keyword("Phase")
SA_CLAIM_COUNT = SearchAttributeKey.for_int("ClaimCount")
SA_TRANSCRIPT_TITLE = SearchAttributeKey.for_keyword("TranscriptTitle")


@workflow.defn
class ExtractTranscriptWorkflow:
    """Extract verifiable claims from a transcript URL or raw text.

    Default mode (v2 thesis extraction):
    1. Fetch/parse the transcript
    2. Single LLM call → 15-30 theses with segment references
    3. Reference verification (programmatic)
    4. Store transcript + all theses
    5. Create Claim records for checkable theses
    6. Spawn child VerifyClaimWorkflow per thesis (sequential)
    """

    def __init__(self) -> None:
        self._phase = "initializing"
        self._url = ""
        self._title = ""
        self._word_count = 0
        self._segment_count = 0
        self._speakers: list[str] = []
        self._thesis_count = 0
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
            "thesis_count": self._thesis_count,
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
    async def run(self, url: str, raw_text: str | None = None,
                  title: str | None = None, date: str | None = None) -> dict:
        """Run the transcript extraction pipeline.

        Args:
            url: Transcript URL (for Rev.com fetching or as identifier for raw text).
            raw_text: If provided, parse this text instead of fetching from URL.
            title: Override title (used with raw_text).
            date: Override date (used with raw_text).

        Returns:
            Dict with transcript metadata and extracted theses.
        """
        self._url = url

        log.info(workflow.logger, MODULE, "started",
                 "Starting transcript extraction",
                 url=url, has_raw_text=bool(raw_text))

        # Step 1: Fetch/parse transcript
        self._set_phase("fetching")

        if raw_text:
            transcript_data = await workflow.execute_activity(
                fetch_raw_transcript,
                args=[raw_text, url, title or "", date],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        else:
            transcript_data = await workflow.execute_activity(
                fetch_transcript,
                args=[url],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            # Convert v1 fetch result to include v2 fields
            if "source_format" not in transcript_data:
                transcript_data["source_format"] = "revcom"
                # Add segment indices for revcom transcripts
                for i, seg in enumerate(transcript_data["segments"]):
                    if "index" not in seg:
                        seg["index"] = i

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

        # Step 2: Store transcript metadata + enrich speakers
        self._set_phase("storing")

        store_result = await workflow.execute_activity(
            store_transcript,
            args=[transcript_data],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        self._transcript_id = store_result["transcript_id"]

        # Build speaker descriptions
        speaker_descriptions = {}
        enriched_speakers = store_result.get("speakers", [])
        for s in enriched_speakers:
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

        # Step 3: Extract theses (single LLM call)
        self._set_phase("extracting_theses")

        theses = await workflow.execute_activity(
            extract_theses_activity,
            args=[transcript_data, enriched_speakers],
            # Single LLM call over full transcript — can take 20-40 min on local LLM
            start_to_close_timeout=timedelta(seconds=2700),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        self._thesis_count = len(theses)

        # Tag all theses as v2 and separate checkable from skipped
        for t in theses:
            t["thesis_version"] = 2
            t["claim_text"] = t["thesis_statement"]

        checkable_theses = [t for t in theses if t.get("worth_checking", False)]
        self._claim_count = len(checkable_theses)
        self._claims = checkable_theses

        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        log.info(workflow.logger, MODULE, "theses_extracted",
                 "Theses extracted",
                 total=len(theses),
                 checkable=len(checkable_theses))

        # Step 4: Store ALL theses as transcript_claims
        if self._transcript_id and theses:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, theses],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Build index mapping: only checkable theses get Claim records
            checkable_tc_ids = []
            for i, t in enumerate(theses):
                if t.get("worth_checking", False):
                    checkable_tc_ids.append(tc_ids[i])

            # Step 5: Create Claim records for checkable theses
            if checkable_theses:
                self._set_phase("submitting")

                # Build claim dicts for create_claims_for_transcript
                claim_dicts = []
                for t in checkable_theses:
                    claim_dicts.append({
                        "claim_text": t["thesis_statement"],
                        "speaker": t["speakers"][0] if t.get("speakers") else "",
                        "source_url": url,
                    })

                claim_ids = await workflow.execute_activity(
                    create_claims_for_transcript,
                    args=[self._transcript_id, checkable_tc_ids, claim_dicts,
                          transcript_data.get("date"), self._title,
                          speaker_descriptions],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                self._verification_submitted = len(claim_ids)

                # Step 6: Transition to verifying
                await workflow.execute_activity(
                    update_transcript_status,
                    args=[self._transcript_id, "verifying"],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )

                # Notify frontend that extraction is done
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

                # Run child verification workflows sequentially
                verified = 0
                failed = 0
                for claim_id_str, thesis in zip(claim_ids, checkable_theses):
                    try:
                        speaker_name = thesis["speakers"][0] if thesis.get("speakers") else None
                        speaker_desc = speaker_descriptions.get(speaker_name, "") if speaker_name else ""

                        # Collect supporting quote texts for decompose context
                        supporting_quotes = []
                        for ref in thesis.get("supporting_references", []):
                            seg_idx = ref.get("segment_index")
                            if seg_idx is not None:
                                for seg in transcript_data["segments"]:
                                    if seg.get("index") == seg_idx:
                                        supporting_quotes.append(seg["text"])
                                        break

                        await workflow.execute_child_workflow(
                            VerifyClaimWorkflow.run,
                            args=[claim_id_str, thesis["thesis_statement"],
                                  speaker_name, transcript_date,
                                  True,  # is_child
                                  self._title,  # transcript_title
                                  speaker_desc,  # speaker_description
                                  supporting_quotes],  # supporting_quotes
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

                # Mark transcript complete
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

            else:
                # No checkable theses — mark complete
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
                log.info(workflow.logger, MODULE, "no_checkable",
                         "No checkable theses, transcript marked complete")

        elif self._transcript_id:
            # No theses extracted at all
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
            log.info(workflow.logger, MODULE, "no_theses",
                     "No theses extracted, transcript marked complete")

        # Final frontend notification
        await workflow.execute_activity(
            notify_frontend_refresh,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        self._set_phase("complete")

        result = {
            "url": url,
            "title": self._title,
            "word_count": self._word_count,
            "segment_count": self._segment_count,
            "speakers": self._speakers,
            "thesis_count": self._thesis_count,
            "claim_count": self._claim_count,
            "claims": checkable_theses,
            "transcript_id": self._transcript_id,
            "verification_submitted": self._verification_submitted,
        }

        log.info(workflow.logger, MODULE, "complete",
                 "Transcript extraction complete",
                 title=self._title,
                 thesis_count=self._thesis_count,
                 claim_count=self._claim_count)

        return result
