"""Fetch and store a transcript — Phase 1 of the pipeline.

Fetches/parses the transcript, stores it in DB, enriches speakers.
Returns transcript_id, enriched_speakers, transcript_meta, and turns.
"""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        fetch_transcript,
        fetch_raw_transcript,
        store_transcript,
    )
    from src.utils.logging import log

MODULE = "fetch_and_store"


@workflow.defn
class FetchAndStoreWorkflow:
    """Fetch, parse, and store a transcript with speaker enrichment."""

    @workflow.run
    async def run(
        self,
        url: str,
        raw_text: str | None = None,
        title: str | None = None,
        date: str | None = None,
    ) -> dict:
        log.info(workflow.logger, MODULE, "started",
                 "Fetching transcript", url=url)

        # Step 1: Fetch/parse
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
            if "source_format" not in transcript_data:
                transcript_data["source_format"] = "revcom"

        # Step 2: Store + enrich speakers
        store_result = await workflow.execute_activity(
            store_transcript,
            args=[transcript_data],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        transcript_id = store_result["transcript_id"]
        enriched_speakers = store_result.get("speakers", [])

        # Build slim transcript_meta (7 fields — no turns, no display_text)
        transcript_meta = {
            "url": transcript_data["url"],
            "title": transcript_data["title"],
            "date": transcript_data.get("date"),
            "speakers": transcript_data["speakers"],
            "word_count": transcript_data["word_count"],
            "turn_count": len(transcript_data["turns"]),
            "source_format": transcript_data.get("source_format", "revcom"),
        }

        log.info(workflow.logger, MODULE, "complete",
                 "Transcript fetched and stored",
                 transcript_id=transcript_id,
                 title=transcript_meta["title"],
                 word_count=transcript_meta["word_count"])

        return {
            "transcript_id": transcript_id,
            "enriched_speakers": enriched_speakers,
            "transcript_meta": transcript_meta,
            "turns": transcript_data["turns"],
        }
