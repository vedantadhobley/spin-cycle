"""Chunked claim extraction — Phase 2 of the pipeline.

Builds chunks deterministically, extracts claims in parallel pairs
(2 LLM slots), stores all claims once via INSERT.
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        extract_chunk_activity,
        store_transcript_claims,
    )
    from src.utils.logging import log
    from src.transcript.thesis_extractor import build_chunks
    from src.transcript.parsers import SpeakerTurn
    from src.config import MAX_CONCURRENT

MODULE = "extract_claims"


@workflow.defn
class ExtractClaimsWorkflow:
    """Extract claims from transcript chunks in parallel."""

    @workflow.run
    async def run(
        self,
        transcript_id: str,
        transcript_meta: dict,
        enriched_speakers: list[dict],
        turns: list[dict],
    ) -> dict:
        log.info(workflow.logger, MODULE, "started",
                 "Starting chunked extraction",
                 transcript_id=transcript_id,
                 title=transcript_meta["title"])

        # Build chunks (deterministic, pure function — safe in workflow)
        speaker_turns = [
            SpeakerTurn(
                speaker=t["speaker"], text=t["text"],
                section_header=t.get("section_header"),
            )
            for t in turns
        ]
        chunks = build_chunks(speaker_turns)

        log.info(workflow.logger, MODULE, "chunks_planned",
                 f"Planned {len(chunks)} chunks",
                 chunk_count=len(chunks))

        # Execute chunks with semaphore — keeps both LLM slots busy
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        chunk_results: list[list[dict]] = [[] for _ in chunks]

        async def extract_with_sem(idx: int, chunk):
            async with sem:
                chunk_dict = {
                    "target_text": chunk.target_text,
                    "context_before": chunk.context_before,
                    "context_after": chunk.context_after,
                    "full_text": chunk.full_text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }
                result = await workflow.execute_activity(
                    extract_chunk_activity,
                    args=[transcript_meta, chunk_dict, enriched_speakers],
                    start_to_close_timeout=timedelta(seconds=2700),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
                chunk_results[idx] = result

        await asyncio.gather(*(
            extract_with_sem(i, chunk) for i, chunk in enumerate(chunks)
        ))

        all_theses: list[dict] = []
        for theses in chunk_results:
            all_theses.extend(theses)

        log.info(workflow.logger, MODULE, "extraction_done",
                 f"Extracted {len(all_theses)} claims",
                 total=len(all_theses))

        # Tag theses for storage
        tagged = _tag_theses_for_storage(all_theses)

        # Store once (INSERT)
        tc_ids: list[str] = []
        if transcript_id and tagged:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[transcript_id, tagged],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            log.info(workflow.logger, MODULE, "stored",
                     f"Stored {len(tc_ids)} claims",
                     transcript_id=transcript_id)

        return {
            "all_theses": all_theses,
            "tc_ids": tc_ids,
        }


def _tag_theses_for_storage(all_theses: list[dict]) -> list[dict]:
    """Tag raw theses with storage fields. Returns new list of dicts."""
    tagged = []
    for t in all_theses:
        tagged.append({
            "claim_text": t["thesis_statement"],
            "original_quote": t.get("original_quote", ""),
            "speaker": (
                t["speakers"][0] if t.get("speakers") else "Unknown"
            ),
            "topic": t.get("topic"),
            "worth_checking": True,
            "is_duplicate": False,
        })
    return tagged
