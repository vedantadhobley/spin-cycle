"""Orchestrator workflow for the transcript extraction pipeline.

Calls child workflows sequentially, passing data through return values.
Each child owns its activities and DB storage. The orchestrator's only
direct activity is finish_transcript_and_start_next for queue management.

Phases visible in Temporal UI:
  1. fetching        — FetchAndStoreWorkflow
  2. extracting      — ExtractClaimsWorkflow
  3. classifying     — ClassifyAndDedupWorkflow (phase 1)
  4. deduplicating   — ClassifyAndDedupWorkflow (phase 2)
  5. synthesizing    — SynthesizeClaimsWorkflow
  6. verifying       — VerifyClaimWorkflow × N
  7. complete        — done
"""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        update_transcript_status,
        finish_transcript_and_start_next,
        notify_frontend_refresh,
    )
    from src.workflows.fetch_and_store import FetchAndStoreWorkflow
    from src.workflows.extract_claims import ExtractClaimsWorkflow
    from src.workflows.classify_and_dedup import ClassifyAndDedupWorkflow
    from src.workflows.synthesize_claims import SynthesizeClaimsWorkflow
    from src.workflows.verify_all_claims import VerifyAllClaimsWorkflow
    from src.utils.logging import log
    from src.config import (
        TASK_QUEUE,
        TIMEOUT_UPDATE_STATUS,
        TIMEOUT_NOTIFY_FRONTEND,
        TIMEOUT_FINISH_TRANSCRIPT,
    )

MODULE = "transcript_pipeline"

SA_PHASE = SearchAttributeKey.for_keyword("Phase")
SA_CLAIM_COUNT = SearchAttributeKey.for_int("ClaimCount")
SA_TRANSCRIPT_TITLE = SearchAttributeKey.for_keyword("TranscriptTitle")


@workflow.defn
class TranscriptPipelineWorkflow:
    """Orchestrate transcript extraction through child workflows."""

    def __init__(self) -> None:
        self._phase = "initializing"
        self._url = ""
        self._title = ""
        self._transcript_id = ""
        self._thesis_count = 0
        self._claim_count = 0

    @workflow.query
    def status(self) -> dict:
        return {
            "phase": self._phase,
            "url": self._url,
            "title": self._title,
            "transcript_id": self._transcript_id,
            "thesis_count": self._thesis_count,
            "claim_count": self._claim_count,
        }

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        workflow.upsert_search_attributes([SA_PHASE.value_set(phase)])

    @workflow.run
    async def run(
        self,
        url: str,
        raw_text: str | None = None,
        title: str | None = None,
        date: str | None = None,
        stop_after: str | None = None,
    ) -> dict:
        self._url = url

        log.info(workflow.logger, MODULE, "started",
                 "Starting transcript pipeline",
                 url=url, stop_after=stop_after)

        # --- Phase 1: Fetch and Store ---
        self._set_phase("fetching")
        fetch_result = await workflow.execute_child_workflow(
            FetchAndStoreWorkflow.run,
            args=[url, raw_text, title, date],
            id=f"fetch-{workflow.info().workflow_id}",
            task_queue=TASK_QUEUE,
        )
        self._transcript_id = fetch_result["transcript_id"]
        self._title = fetch_result["transcript_meta"]["title"]
        workflow.upsert_search_attributes([
            SA_TRANSCRIPT_TITLE.value_set(self._title),
        ])

        if stop_after == "fetch":
            self._set_phase("complete")
            return {**fetch_result, "stopped_after": "fetch"}

        # --- Phase 2: Extract Claims ---
        self._set_phase("extracting")
        extract_result = await workflow.execute_child_workflow(
            ExtractClaimsWorkflow.run,
            args=[
                fetch_result["transcript_id"],
                fetch_result["transcript_meta"],
                fetch_result["enriched_speakers"],
                fetch_result["turns"],
            ],
            id=f"extract-claims-{workflow.info().workflow_id}",
            task_queue=TASK_QUEUE,
        )
        self._thesis_count = len(extract_result["all_theses"])

        if stop_after == "extract":
            await self._mark_complete()
            return {**extract_result, "stopped_after": "extract"}

        # --- Phase 3: Classify and Dedup ---
        self._set_phase("classifying")
        dedup_result = await workflow.execute_child_workflow(
            ClassifyAndDedupWorkflow.run,
            args=[
                fetch_result["transcript_id"],
                extract_result["tc_ids"],
                extract_result["all_theses"],
                fetch_result["enriched_speakers"],
            ],
            id=f"classify-dedup-{workflow.info().workflow_id}",
            task_queue=TASK_QUEUE,
        )

        if stop_after == "dedup":
            await self._mark_complete()
            return {**dedup_result, "stopped_after": "dedup"}

        # --- Phase 4: Synthesize ---
        self._set_phase("synthesizing")

        # Build speaker descriptions from enriched speakers
        speaker_descriptions = {}
        for s in fetch_result["enriched_speakers"]:
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

        synth_result = await workflow.execute_child_workflow(
            SynthesizeClaimsWorkflow.run,
            args=[
                fetch_result["transcript_id"],
                dedup_result["dedup_groups"],
                dedup_result["classified_theses"],
                fetch_result["enriched_speakers"],
                dedup_result["tc_ids"],
                url,
                fetch_result["transcript_meta"].get("date"),
                self._title,
                speaker_descriptions,
            ],
            id=f"synthesize-{workflow.info().workflow_id}",
            task_queue=TASK_QUEUE,
        )
        self._claim_count = len(synth_result.get("claim_ids", []))
        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        if stop_after == "synthesize":
            await self._mark_complete()
            return {**synth_result, "stopped_after": "synthesize"}

        # --- Phase 5: Verify ---
        if synth_result.get("claim_ids"):
            await workflow.execute_activity(
                update_transcript_status,
                args=[self._transcript_id, "verifying"],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_UPDATE_STATUS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                notify_frontend_refresh,
                start_to_close_timeout=timedelta(seconds=TIMEOUT_NOTIFY_FRONTEND),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            self._set_phase("verifying")
            transcript_date = fetch_result["transcript_meta"].get("date") or "unknown"

            # Build claim dicts for VerifyAllClaimsWorkflow
            verify_claims = []
            for claim_id_str, group in zip(
                synth_result["claim_ids"],
                synth_result["checkable_groups"],
            ):
                speaker_name = group["speaker"]
                verify_claims.append({
                    "claim_id": claim_id_str,
                    "claim_text": group["claim_text"],
                    "speaker": speaker_name,
                    "speaker_description": speaker_descriptions.get(
                        speaker_name, ""
                    ),
                    "transcript_date": transcript_date,
                    "transcript_title": self._title,
                    "supporting_quotes": group.get("original_quotes", []),
                })

            verify_result = await workflow.execute_child_workflow(
                VerifyAllClaimsWorkflow.run,
                args=[verify_claims],
                id=f"verify-all-{workflow.info().workflow_id}",
                task_queue=TASK_QUEUE,
            )

            log.info(workflow.logger, MODULE, "verification_done",
                     "All verifications complete",
                     verified=verify_result["verified"],
                     failed=verify_result["failed"])

        # --- Finish ---
        await self._mark_complete()
        await workflow.execute_activity(
            finish_transcript_and_start_next,
            start_to_close_timeout=timedelta(seconds=TIMEOUT_FINISH_TRANSCRIPT),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        await workflow.execute_activity(
            notify_frontend_refresh,
            start_to_close_timeout=timedelta(seconds=TIMEOUT_NOTIFY_FRONTEND),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        self._set_phase("complete")
        return {
            "transcript_id": self._transcript_id,
            "title": self._title,
            "thesis_count": self._thesis_count,
            "claim_count": self._claim_count,
        }

    async def _mark_complete(self) -> None:
        if self._transcript_id:
            await workflow.execute_activity(
                update_transcript_status,
                args=[self._transcript_id, "complete"],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_UPDATE_STATUS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
