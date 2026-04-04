"""Full claim extraction pipeline — extract, classify, dedup, synthesize.

Sentencizes the transcript, builds sentence-based chunks, then runs two
sequential LLM passes per chunk (parallel pairs within each pass):

  Pass 1: Group sentences + label claim/not_claim (no thesis writing)
  Pass 2: Context-inject each claim group (resolve pronouns/references)

Then calls ClassifyAndDedupWorkflow and SynthesizeClaimsWorkflow as child
workflows to classify, dedup, and synthesize the extracted claims.

Pipeline phases:
  1. Sentencize + chunk
  2. Pass 1: Group + disposition (parallel pairs, sem=2)
  3. Pass 2: Context injection (parallel pairs, skip chunks with 0 claims)
  4. Store transcript_claims
  5. Classify + dedup (child workflow)
  6. Synthesize (child workflow)
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        sentencize_and_chunk_activity,
        extract_dispositions_activity,
        inject_context_activity,
        store_transcript_claims,
        load_extract_inputs,
    )
    from src.workflows.classify_and_dedup import ClassifyAndDedupWorkflow
    from src.workflows.synthesize_claims import SynthesizeClaimsWorkflow
    from src.utils.logging import log
    from src.config import (
        MAX_CONCURRENT, TIMEOUT_EXTRACT_CHUNK, TIMEOUT_INJECT_CONTEXT,
        TIMEOUT_STORE_CLAIMS, TASK_QUEUE,
    )

MODULE = "extract_claims"


@workflow.defn
class ExtractClaimsWorkflow:
    """Extract, classify, dedup, and synthesize claims from a transcript."""

    @workflow.run
    async def run(
        self,
        transcript_id: str,
        transcript_meta: dict | None = None,
        enriched_speakers: list[dict] | None = None,
        turns: list[dict] | None = None,
        source_url: str | None = None,
        stop_after: str | None = None,
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
                 "Starting extraction pipeline",
                 transcript_id=transcript_id,
                 title=transcript_meta["title"],
                 stop_after=stop_after)

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

        # ---------------------------------------------------------------
        # Pass 1: Group + disposition (parallel pairs, sem=2)
        # ---------------------------------------------------------------
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        grouping_results: list[dict] = [None] * len(chunk_dicts)

        async def group_with_sem(idx: int, chunk_dict: dict):
            async with sem:
                result = await workflow.execute_activity(
                    extract_dispositions_activity,
                    args=[transcript_meta, chunk_dict, enriched_speakers],
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_EXTRACT_CHUNK),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
                grouping_results[idx] = result

        await asyncio.gather(*(
            group_with_sem(i, cd) for i, cd in enumerate(chunk_dicts)
        ))

        total_claim_groups = sum(
            sum(1 for g in gr["groups"] if g["disposition"] == "claim")
            for gr in grouping_results
        )
        log.info(workflow.logger, MODULE, "pass1_done",
                 "Pass 1 complete — all chunks grouped",
                 transcript_id=transcript_id,
                 total_claim_groups=total_claim_groups)

        # ---------------------------------------------------------------
        # Pass 2: Context injection (parallel pairs, skip chunks with 0 claims)
        # ---------------------------------------------------------------
        chunk_theses: list[list[dict]] = [[] for _ in chunk_dicts]

        async def inject_with_sem(idx: int, chunk_dict: dict, grp_result: dict):
            claim_groups = [g for g in grp_result.get("groups", [])
                            if g["disposition"] == "claim"]
            if not claim_groups:
                return  # no claims in this chunk
            async with sem:
                result = await workflow.execute_activity(
                    inject_context_activity,
                    args=[transcript_meta, chunk_dict, grp_result,
                          enriched_speakers, sentences_dict],
                    start_to_close_timeout=timedelta(seconds=TIMEOUT_INJECT_CONTEXT),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
                chunk_theses[idx] = result

        await asyncio.gather(*(
            inject_with_sem(i, cd, grouping_results[i])
            for i, cd in enumerate(chunk_dicts)
        ))

        # Flatten
        all_theses: list[dict] = []
        for theses in chunk_theses:
            all_theses.extend(theses)

        log.info(workflow.logger, MODULE, "pass2_done",
                 "Pass 2 complete — all claims decontextualized",
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

        # Return deduped theses to keep indices in sync with tc_ids
        if dupes_dropped:
            seen_texts: set[str] = set()
            deduped_theses = []
            for t in all_theses:
                text = t["thesis_statement"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    deduped_theses.append(t)
            all_theses = deduped_theses

        if stop_after == "extract":
            return {
                "all_theses": all_theses,
                "tc_ids": tc_ids,
                "stopped_after": "extract",
            }

        # ---------------------------------------------------------------
        # Phase 3: Classify + Dedup (child workflow)
        # ---------------------------------------------------------------
        dedup_result = await workflow.execute_child_workflow(
            ClassifyAndDedupWorkflow.run,
            args=[
                transcript_id,
                tc_ids,
                all_theses,
                enriched_speakers,
            ],
            id=f"classify-dedup-{workflow.info().workflow_id}",
            task_queue=TASK_QUEUE,
        )

        if stop_after == "dedup":
            return {
                **dedup_result,
                "stopped_after": "dedup",
            }

        # ---------------------------------------------------------------
        # Phase 4: Synthesize (child workflow)
        # ---------------------------------------------------------------
        speaker_descriptions = {}
        for s in enriched_speakers:
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

        synth_result = await workflow.execute_child_workflow(
            SynthesizeClaimsWorkflow.run,
            args=[
                transcript_id,
                dedup_result["dedup_groups"],
                dedup_result["classified_theses"],
                enriched_speakers,
                dedup_result["tc_ids"],
                source_url or transcript_meta.get("url", ""),
                transcript_meta.get("date"),
                transcript_meta["title"],
                transcript_meta.get("description") or "",
                speaker_descriptions,
            ],
            id=f"synthesize-{workflow.info().workflow_id}",
            task_queue=TASK_QUEUE,
        )

        log.info(workflow.logger, MODULE, "complete",
                 "Extraction pipeline complete",
                 transcript_id=transcript_id,
                 thesis_count=len(all_theses),
                 claim_count=len(synth_result.get("claim_ids", [])))

        return {
            "all_theses": all_theses,
            "tc_ids": tc_ids,
            "checkable_groups": synth_result.get("checkable_groups", []),
            "claim_ids": synth_result.get("claim_ids", []),
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
            "worth_checking": True,
            "is_duplicate": False,
        })
    return tagged
