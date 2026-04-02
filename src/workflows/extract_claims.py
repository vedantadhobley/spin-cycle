"""Chunked claim extraction — Phase 2 of the pipeline.

Sentencizes the transcript, builds sentence-based chunks, extracts claims
in parallel pairs (2 LLM slots), stores all claims once via INSERT.
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        sentencize_and_chunk_activity,
        extract_chunk_activity,
        store_transcript_claims,
        load_extract_inputs,
    )
    from src.utils.logging import log
    from src.config import MAX_CONCURRENT, TIMEOUT_EXTRACT_CHUNK, TIMEOUT_STORE_CLAIMS

MODULE = "extract_claims"


@workflow.defn
class ExtractClaimsWorkflow:
    """Extract claims from transcript chunks in parallel."""

    @workflow.run
    async def run(
        self,
        transcript_id: str,
        transcript_meta: dict | None = None,
        enriched_speakers: list[dict] | None = None,
        turns: list[dict] | None = None,
    ) -> dict:
        # Load from DB if inputs not provided (standalone mode)
        if transcript_meta is None or enriched_speakers is None or turns is None:
            loaded = await workflow.execute_activity(
                load_extract_inputs,
                args=[transcript_id],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_STORE_CLAIMS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            transcript_meta = transcript_meta or loaded["transcript_meta"]
            enriched_speakers = enriched_speakers or loaded["enriched_speakers"]
            turns = turns or loaded["turns"]

        log.info(workflow.logger, MODULE, "started",
                 "Starting sentence-level extraction",
                 transcript_id=transcript_id,
                 title=transcript_meta["title"])

        # Sentencize and build chunks (activity — SpaCy can't run in workflow sandbox)
        sentencize_result = await workflow.execute_activity(
            sentencize_and_chunk_activity,
            args=[turns],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        chunk_dicts = sentencize_result["chunk_dicts"]
        sentences_dict = sentencize_result["sentences_dict"]

        log.info(workflow.logger, MODULE, "chunks_planned",
                 "Sentence chunk plan ready",
                 transcript_id=transcript_id,
                 sentence_count=sentencize_result["sentence_count"],
                 chunk_count=len(chunk_dicts))

        # Execute chunks with semaphore — keeps both LLM slots busy
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        chunk_results: list[list[dict]] = [[] for _ in chunk_dicts]

        async def extract_with_sem(idx: int, chunk_dict: dict):
            async with sem:
                result = await workflow.execute_activity(
                    extract_chunk_activity,
                    args=[transcript_meta, chunk_dict, enriched_speakers, sentences_dict],
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_EXTRACT_CHUNK),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
                chunk_results[idx] = result

        await asyncio.gather(*(
            extract_with_sem(i, cd) for i, cd in enumerate(chunk_dicts)
        ))

        all_theses: list[dict] = []
        for theses in chunk_results:
            all_theses.extend(theses)

        log.info(workflow.logger, MODULE, "extraction_done",
                 "Extraction complete",
                 transcript_id=transcript_id,
                 thesis_count=len(all_theses))

        # Tag theses for storage (deduplicates exact cross-chunk duplicates)
        tagged = _tag_theses_for_storage(all_theses)
        dupes_dropped = len(all_theses) - len(tagged)
        if dupes_dropped:
            log.info(workflow.logger, MODULE, "cross_chunk_dedup",
                     "Exact cross-chunk duplicates dropped before storage",
                     transcript_id=transcript_id,
                     raw_count=len(all_theses),
                     unique_count=len(tagged),
                     dropped=dupes_dropped)

        # Store once (INSERT)
        tc_ids: list[str] = []
        if transcript_id and tagged:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[transcript_id, tagged],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_STORE_CLAIMS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            log.info(workflow.logger, MODULE, "stored",
                     "Claims stored",
                     transcript_id=transcript_id,
                     claim_count=len(tc_ids))

        return {
            "all_theses": all_theses,
            "tc_ids": tc_ids,
        }


def _tag_theses_for_storage(all_theses: list[dict]) -> list[dict]:
    """Tag raw theses with storage fields. Returns new list of dicts.

    Deduplicates by exact claim_text — overlapping chunks can cause the LLM
    to extract the same claim from context regions of adjacent chunks.
    First occurrence (earlier chunk) wins.
    """
    tagged = []
    seen_texts: set[str] = set()
    for t in all_theses:
        text = t["thesis_statement"]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        tagged.append({
            "claim_text": text,
            "original_quote": t.get("original_quote", ""),
            "speaker": (
                t["speakers"][0] if t.get("speakers") else "Unknown"
            ),
            "topic": t.get("topic"),
            "worth_checking": True,
            "is_duplicate": False,
        })
    return tagged
