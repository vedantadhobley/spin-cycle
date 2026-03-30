"""Temporal workflow for transcript claim extraction.

Multi-phase extraction pipeline:
  Phase 1: Chunked extraction — split transcript into overlapping chunks,
    extract claims from each chunk in parallel (up to 2 at a time).
    Produces 4-field claims (thesis_statement, speakers, original_quote, topic).
  Phase 1b: Batch classification — classifies claims by checkability
    (verifiable_fact, future_prediction, etc.) in cheap batch LLM calls.
  Phase 2: Embedding-based dedup — per-speaker cosine similarity clustering.
    Fast, deterministic, no context growth.
  Phase 2b: Synthesis — per-group overarching claim generation
    (SynthesizeClaimsWorkflow child). Only multi-member clusters need LLM.

After dedup+synthesis, one Claim record is created per checkable group.
All member TranscriptClaims point to their group's Claim. Decompose
receives the synthesized overarching claim text.

Phases visible in Temporal UI:
  1. fetching           — downloading and parsing the transcript
  2. storing            — persisting transcript + enriching speakers
  3. extracting_claims  — chunked Phase 1 extraction (lean 4-field)
  4. classifying        — batch LLM classification
  5. deduplicating      — embedding-based dedup per speaker
  6. synthesizing       — Phase 2b overarching claims per group
  7. submitting         — creating claim records for verification
  8. verifying          — running child verification workflows
  9. complete           — done
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        fetch_transcript,
        fetch_raw_transcript,
        extract_chunk_activity,
        classify_claims_activity,
        dedup_claims_activity,
        store_transcript,
        store_transcript_claims,
        create_claims_for_transcript,
        update_transcript_status,
        finish_transcript_and_start_next,
        notify_frontend_refresh,
    )
    from src.workflows.verify import VerifyClaimWorkflow
    from src.workflows.synthesize_claims import SynthesizeClaimsWorkflow
    from src.utils.logging import log
    from src.transcript.thesis_extractor import build_chunks
    from src.transcript.parsers import SpeakerTurn
    from src.config import (
        TIMEOUT_DEDUP_CLAIMS,
        TIMEOUT_CLASSIFY_CLAIMS,
        CLASSIFY_BATCH_SIZE,
    )

MODULE = "transcript_workflow"

# Search attribute keys for Temporal UI visibility
SA_PHASE = SearchAttributeKey.for_keyword("Phase")
SA_CLAIM_COUNT = SearchAttributeKey.for_int("ClaimCount")
SA_TRANSCRIPT_TITLE = SearchAttributeKey.for_keyword("TranscriptTitle")


@workflow.defn
class ExtractTranscriptWorkflow:
    """Extract verifiable claims from a transcript URL or raw text.

    Pipeline:
    1. Fetch/parse the transcript
    2. Store transcript metadata + enrich speakers
    3. Chunked extraction (Phase 1) — parallel chunks, lean 4-field output
    4. Batch classification — checkability assessment in cheap LLM calls
    5. Embedding-based dedup (Phase 2) — per-speaker cosine similarity clustering
    6. Synthesis (Phase 2b) — per-speaker child SynthesizeClaimsWorkflow
    7. Store all claims + create Claim records per checkable group
    8. Spawn child VerifyClaimWorkflow per checkable group (sequential)
    """

    def __init__(self) -> None:
        self._phase = "initializing"
        self._url = ""
        self._title = ""
        self._word_count = 0
        self._turn_count = 0
        self._speakers: list[str] = []
        self._thesis_count = 0
        self._claim_count = 0
        self._group_count = 0
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
            "turn_count": self._turn_count,
            "speakers": self._speakers,
            "thesis_count": self._thesis_count,
            "claim_count": self._claim_count,
            "group_count": self._group_count,
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
                  title: str | None = None, date: str | None = None,
                  stop_after: str | None = None) -> dict:
        """Run the transcript extraction pipeline.

        Args:
            url: Transcript URL (C-SPAN program URL or identifier for raw text).
            raw_text: If provided, parse this text instead of fetching from URL.
            title: Override title (used with raw_text).
            date: Override date (used with raw_text).
            stop_after: Early exit point for testing. One of:
                - "fetch"     — fetch + parse only, return transcript data (no DB)
                - "store"     — fetch + store to DB with speaker enrichment
                - "phase1"    — Phase 1 extraction only, store raw theses
                - "classify"  — Phase 1 + classification, before dedup
                - "review"    — Phase 1 + classify + dedup, skip synthesis
                - "extract"   — full extraction + synthesis, skip verification
                - None        — full pipeline (default)

        Returns:
            Dict with transcript metadata and extracted claims/groups.
        """
        self._url = url

        log.info(workflow.logger, MODULE, "started",
                 "Starting transcript extraction",
                 url=url, has_raw_text=bool(raw_text),
                 stop_after=stop_after)

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
            if "source_format" not in transcript_data:
                transcript_data["source_format"] = "revcom"

        self._title = transcript_data["title"]
        self._word_count = transcript_data["word_count"]
        self._turn_count = len(transcript_data["turns"])
        self._speakers = transcript_data["speakers"]

        workflow.upsert_search_attributes([
            SA_TRANSCRIPT_TITLE.value_set(self._title),
        ])

        log.info(workflow.logger, MODULE, "fetched",
                 "Transcript fetched",
                 title=self._title,
                 word_count=self._word_count,
                 turn_count=self._turn_count,
                 speakers=self._speakers)

        if stop_after == "fetch":
            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "turn_count": self._turn_count,
                "speakers": self._speakers,
                "transcript_data": transcript_data,
                "stopped_after": "fetch",
            }

        # Step 2: Store transcript metadata + enrich speakers
        self._set_phase("storing")

        store_result = await workflow.execute_activity(
            store_transcript,
            args=[transcript_data],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        self._transcript_id = store_result["transcript_id"]

        speaker_descriptions = {}
        enriched_speakers = store_result.get("speakers", [])
        for s in enriched_speakers:
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

        if stop_after == "store":
            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "turn_count": self._turn_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "enriched_speakers": enriched_speakers,
                "stopped_after": "store",
            }

        # Step 3: Chunked extraction (Phase 1)
        self._set_phase("extracting_claims")

        # Build chunks (deterministic, pure function — safe in workflow)
        turns = [
            SpeakerTurn(
                speaker=t["speaker"], text=t["text"],
                section_header=t.get("section_header"),
            )
            for t in transcript_data["turns"]
        ]
        chunks = build_chunks(turns)

        log.info(workflow.logger, MODULE, "chunks_planned",
                 f"Planned {len(chunks)} chunks for extraction",
                 chunk_count=len(chunks))

        # Execute chunks with semaphore — keeps both LLM slots busy.
        # As soon as one chunk finishes, the next starts immediately.
        from src.config import MAX_CONCURRENT
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        # Results indexed by chunk position to preserve ordering
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
                    args=[transcript_data, chunk_dict, enriched_speakers],
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

        self._thesis_count = len(all_theses)

        log.info(workflow.logger, MODULE, "extraction_done",
                 f"Phase 1 complete: {len(all_theses)} claims extracted",
                 total=len(all_theses))

        # Step 3b: Store ALL raw theses as transcript_claims
        # (stored before dedup so they exist even if dedup fails)
        all_theses_with_meta = _tag_theses_for_storage(all_theses)

        if self._transcript_id and all_theses_with_meta:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, all_theses_with_meta],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            log.info(workflow.logger, MODULE, "theses_stored",
                     f"Stored {len(tc_ids)} raw theses to transcript_claims",
                     transcript_id=self._transcript_id,
                     tc_count=len(tc_ids))
        else:
            tc_ids = []

        if stop_after == "phase1":
            if self._transcript_id:
                await workflow.execute_activity(
                    update_transcript_status,
                    args=[self._transcript_id, "complete"],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "turn_count": self._turn_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "all_theses": all_theses,
                "stopped_after": "phase1",
            }

        # Step 3c: Batch classification
        self._set_phase("classifying")

        log.info(workflow.logger, MODULE, "classify_start",
                 f"Classifying {len(all_theses)} claims in batches "
                 f"of {CLASSIFY_BATCH_SIZE}",
                 total=len(all_theses),
                 batch_size=CLASSIFY_BATCH_SIZE)

        for batch_start in range(0, len(all_theses), CLASSIFY_BATCH_SIZE):
            batch = all_theses[batch_start:batch_start + CLASSIFY_BATCH_SIZE]
            classified = await workflow.execute_activity(
                classify_claims_activity,
                args=[batch],
                start_to_close_timeout=timedelta(
                    seconds=TIMEOUT_CLASSIFY_CLAIMS
                ),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            # Merge classification fields back into all_theses
            for i, c in enumerate(classified):
                idx = batch_start + i
                all_theses[idx]["classification"] = c.get(
                    "classification", "verifiable_fact"
                )
                all_theses[idx]["checkable"] = c.get("checkable", True)
                all_theses[idx]["check_rationale"] = c.get(
                    "check_rationale", ""
                )

        # Update stored theses with classification data
        all_theses_with_meta = _tag_theses_for_storage(all_theses)

        if self._transcript_id and all_theses_with_meta:
            await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, all_theses_with_meta],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        log.info(workflow.logger, MODULE, "classify_done",
                 f"Classification complete for {len(all_theses)} claims",
                 total=len(all_theses))

        if stop_after == "classify":
            if self._transcript_id:
                await workflow.execute_activity(
                    update_transcript_status,
                    args=[self._transcript_id, "complete"],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "turn_count": self._turn_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "all_theses": all_theses,
                "stopped_after": "classify",
            }

        # Step 4: Embedding-based dedup (Phase 2)
        self._set_phase("deduplicating")

        # Group theses by primary speaker
        speaker_theses: dict[str, list[tuple[int, dict]]] = {}
        for global_i, t in enumerate(all_theses):
            speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
            speaker_theses.setdefault(speaker, []).append((global_i, t))

        # Log speaker breakdown
        speaker_counts = {
            sp: len(theses) for sp, theses in speaker_theses.items()
        }
        log.info(workflow.logger, MODULE, "speaker_breakdown",
                 f"Claims by speaker: {speaker_counts}",
                 speaker_counts=speaker_counts,
                 speaker_count=len(speaker_theses))

        # dedup_results: speaker → { clusters: [...] }
        dedup_results: dict[str, dict] = {}

        for speaker, indexed_theses in speaker_theses.items():
            sp_theses = [t for _, t in indexed_theses]
            sp_global_indices = [gi for gi, _ in indexed_theses]

            if len(sp_theses) <= 1:
                # Trivial: single claim = single cluster
                dedup_results[speaker] = {
                    "clusters": [{
                        "representative": sp_theses[0],
                        "member_indices": [0],
                        "checkable": sp_theses[0].get("checkable", True),
                        "topic": sp_theses[0].get("topic", ""),
                    }],
                    "global_indices": sp_global_indices,
                }
                log.info(workflow.logger, MODULE, "trivial_dedup",
                         f"Trivial dedup for {speaker} (1 claim)",
                         speaker=speaker)
            else:
                log.info(workflow.logger, MODULE, "dedup_speaker_start",
                         f"Starting dedup for {speaker} "
                         f"({len(sp_theses)} claims)",
                         speaker=speaker, claim_count=len(sp_theses))

                result = await workflow.execute_activity(
                    dedup_claims_activity,
                    args=[sp_theses, speaker],
                    start_to_close_timeout=timedelta(
                        seconds=TIMEOUT_DEDUP_CLAIMS
                    ),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )

                result["global_indices"] = sp_global_indices
                dedup_results[speaker] = result

                cluster_count = len(result["clusters"])
                checkable_count = sum(
                    1 for c in result["clusters"] if c["checkable"]
                )
                log.info(workflow.logger, MODULE, "dedup_speaker_done",
                         f"Dedup done for {speaker}: "
                         f"{cluster_count} clusters "
                         f"({checkable_count} checkable)",
                         speaker=speaker,
                         clusters=cluster_count,
                         checkable=checkable_count)

        # Build consolidated group structures from dedup clusters
        all_groups = _collect_groups_from_dedup(dedup_results, all_theses)

        # Tag theses with dedup metadata (is_duplicate, worth_checking)
        _apply_dedup_metadata(
            all_theses_with_meta, dedup_results, speaker_theses
        )

        checkable_groups = [g for g in all_groups if g["checkable"]]
        self._group_count = len(all_groups)

        log.info(workflow.logger, MODULE, "dedup_done",
                 f"Phase 2 complete: {len(all_groups)} groups "
                 f"({len(checkable_groups)} checkable)",
                 total_groups=len(all_groups),
                 checkable_groups=len(checkable_groups))

        if stop_after == "review":
            if self._transcript_id:
                # Re-store theses with updated metadata
                await workflow.execute_activity(
                    store_transcript_claims,
                    args=[self._transcript_id, all_theses_with_meta],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                await workflow.execute_activity(
                    update_transcript_status,
                    args=[self._transcript_id, "complete"],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "turn_count": self._turn_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "group_count": self._group_count,
                "groups": all_groups,
                "stopped_after": "review",
            }

        # Step 5: Synthesis (Phase 2b) — per speaker child workflows
        # Only multi-member clusters need LLM synthesis.
        # Single-member clusters use thesis_statement directly.
        self._set_phase("synthesizing")

        for speaker, indexed_theses in speaker_theses.items():
            sp_checkable = [
                g for g in checkable_groups if g["speaker"] == speaker
            ]
            if not sp_checkable:
                log.info(workflow.logger, MODULE, "synthesis_skip_speaker",
                         f"No checkable groups for {speaker}, skipping",
                         speaker=speaker)
                continue

            # Split into single-member (no LLM needed) and multi-member
            multi_member = [
                g for g in sp_checkable
                if len(g["member_global_indices"]) > 1
            ]
            single_member = [
                g for g in sp_checkable
                if len(g["member_global_indices"]) == 1
            ]

            # Single-member: use thesis_statement directly
            for g in single_member:
                gi = g["member_global_indices"][0]
                g["claim_text"] = all_theses[gi]["thesis_statement"]

            if not multi_member:
                log.info(workflow.logger, MODULE, "synthesis_skip_speaker",
                         f"All {len(single_member)} groups for {speaker} "
                         f"are single-member, no synthesis needed",
                         speaker=speaker)
                continue

            synth_groups = []
            for g in multi_member:
                member_stmts = [
                    all_theses[gi]["thesis_statement"]
                    for gi in g["member_global_indices"]
                ]
                synth_groups.append({
                    "group_id": g["local_group_id"],
                    "topic": g["topic"],
                    "member_statements": member_stmts,
                })

            log.info(workflow.logger, MODULE, "synthesis_speaker_start",
                     f"Synthesizing {len(synth_groups)} multi-member groups "
                     f"for {speaker} ({len(single_member)} single-member "
                     f"skipped)",
                     speaker=speaker,
                     multi=len(synth_groups),
                     single=len(single_member))

            synth_results = await workflow.execute_child_workflow(
                SynthesizeClaimsWorkflow.run,
                args=[synth_groups, speaker],
                id=f"synthesize-{self._transcript_id}-{speaker[:30]}",
                task_queue="spin-cycle-verify",
            )

            # Apply synthesized claims to groups
            applied = 0
            for g in multi_member:
                synth = synth_results.get(g["local_group_id"])
                if synth:
                    g["claim_text"] = synth["overarching_claim"]
                    applied += 1
                else:
                    log.warning(workflow.logger, MODULE,
                                "synthesis_missing",
                                f"No synthesis result for group "
                                f"{g['local_group_id']} ({g['topic']})",
                                speaker=speaker,
                                group_id=g["local_group_id"],
                                topic=g["topic"])

            log.info(workflow.logger, MODULE, "synthesis_speaker_done",
                     f"Synthesis done for {speaker}: "
                     f"{applied}/{len(multi_member)} multi-member groups",
                     speaker=speaker,
                     applied=applied,
                     total=len(multi_member))

        self._claim_count = len(checkable_groups)
        self._claims = checkable_groups

        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        log.info(workflow.logger, MODULE, "synthesis_done",
                 f"Phase 2b complete: {len(checkable_groups)} overarching "
                 f"claims",
                 checkable_groups=len(checkable_groups))

        if stop_after == "extract":
            if self._transcript_id:
                # Re-store theses with updated metadata
                await workflow.execute_activity(
                    store_transcript_claims,
                    args=[self._transcript_id, all_theses_with_meta],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                await workflow.execute_activity(
                    update_transcript_status,
                    args=[self._transcript_id, "complete"],
                    start_to_close_timeout=timedelta(seconds=15),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )

            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "turn_count": self._turn_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "claim_count": self._claim_count,
                "group_count": self._group_count,
                "groups": all_groups,
                "stopped_after": "extract",
            }

        # Step 6: Create Claim records for checkable groups
        transcript_date = transcript_data.get("date") or "unknown"

        if self._transcript_id and checkable_groups and tc_ids:
            self._set_phase("submitting")

            # Re-store theses with updated metadata
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, all_theses_with_meta],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Build group dicts with member_tc_ids
            group_dicts = []
            for g in checkable_groups:
                member_tc_ids = [tc_ids[i] for i in g["member_global_indices"]]
                group_dicts.append({
                    "claim_text": g["claim_text"],
                    "speaker": g["speaker"],
                    "member_tc_ids": member_tc_ids,
                })

            log.info(workflow.logger, MODULE, "creating_claims",
                     f"Creating {len(group_dicts)} Claim records "
                     f"for verification",
                     group_count=len(group_dicts))

            claim_ids = await workflow.execute_activity(
                create_claims_for_transcript,
                args=[self._transcript_id, tc_ids, group_dicts,
                      transcript_data.get("date"), self._title,
                      speaker_descriptions, url],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            self._verification_submitted = len(claim_ids)

            log.info(workflow.logger, MODULE, "claims_created",
                     f"Created {len(claim_ids)} Claim records, "
                     f"starting verification",
                     claim_count=len(claim_ids))

            # Step 7: Verification
            await workflow.execute_activity(
                update_transcript_status,
                args=[self._transcript_id, "verifying"],
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            await workflow.execute_activity(
                notify_frontend_refresh,
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            self._set_phase("verifying")

            log.info(workflow.logger, MODULE, "verification_started",
                     "Starting sequential child verification workflows",
                     claim_count=len(claim_ids))

            verified = 0
            failed = 0
            for claim_id_str, group in zip(claim_ids, checkable_groups):
                try:
                    speaker_name = group["speaker"]
                    speaker_desc = speaker_descriptions.get(speaker_name, "")

                    # Collect original quotes from all member theses
                    supporting_quotes = group.get("original_quotes", [])

                    await workflow.execute_child_workflow(
                        VerifyClaimWorkflow.run,
                        args=[claim_id_str, group["claim_text"],
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

        elif self._transcript_id:
            # No checkable groups or no theses — mark complete
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
                     "No checkable groups, transcript marked complete")

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
            "turn_count": self._turn_count,
            "speakers": self._speakers,
            "thesis_count": self._thesis_count,
            "claim_count": self._claim_count,
            "group_count": self._group_count,
            "claims": checkable_groups,
            "transcript_id": self._transcript_id,
            "verification_submitted": self._verification_submitted,
        }

        log.info(workflow.logger, MODULE, "complete",
                 "Transcript extraction complete",
                 title=self._title,
                 thesis_count=self._thesis_count,
                 claim_count=self._claim_count,
                 group_count=self._group_count)

        return result


# ---------------------------------------------------------------------------
# Helper functions (deterministic, safe for workflow code)
# ---------------------------------------------------------------------------

def _tag_theses_for_storage(all_theses: list[dict]) -> list[dict]:
    """Tag raw theses with storage metadata. Returns new list of dicts."""
    tagged = []
    for t in all_theses:
        entry = {**t}
        entry["thesis_version"] = 3
        entry["claim_text"] = t["thesis_statement"]
        entry["speaker"] = (
            t["speakers"][0] if t.get("speakers") else "Unknown"
        )
        # Use classification from extraction (LLM-assigned)
        entry["classification"] = t.get("classification", "verifiable_fact")
        entry["claim_type"] = entry["classification"]
        entry["checkable"] = t.get("checkable", True)
        entry["checkability_rationale"] = t.get("check_rationale", "")
        entry["is_duplicate"] = False
        entry["worth_checking"] = False  # updated after dedup
        tagged.append(entry)
    return tagged


def _collect_groups_from_dedup(
    dedup_results: dict[str, dict],
    all_theses: list[dict],
) -> list[dict]:
    """Build flat group list from per-speaker dedup clusters.

    Each cluster becomes one group. Group IDs are namespaced by speaker.
    """
    all_groups = []

    for speaker, result in dedup_results.items():
        global_indices = result["global_indices"]

        for ci, cluster in enumerate(result["clusters"]):
            local_indices = cluster["member_indices"]
            member_global = [global_indices[li] for li in local_indices]

            # Collect original quotes from all member theses (deduped)
            original_quotes = []
            seen_quotes: set[str] = set()
            for gi in member_global:
                quote = all_theses[gi].get("original_quote", "")
                if quote and quote not in seen_quotes:
                    original_quotes.append(quote)
                    seen_quotes.add(quote)

            # Default claim_text — replaced by synthesis for multi-member
            member_stmts = [
                all_theses[gi]["thesis_statement"] for gi in member_global
            ]

            local_group_id = f"C{ci}"

            all_groups.append({
                "group_id": f"{speaker}_{local_group_id}",
                "local_group_id": local_group_id,
                "speaker": speaker,
                "topic": cluster.get("topic", ""),
                "checkable": cluster.get("checkable", False),
                "member_global_indices": member_global,
                "claim_text": "\n".join(member_stmts),
                "original_quotes": original_quotes,
            })

    return all_groups


def _apply_dedup_metadata(
    all_theses_with_meta: list[dict],
    dedup_results: dict[str, dict],
    speaker_theses: dict[str, list[tuple[int, dict]]],
) -> None:
    """Tag stored theses with dedup info (is_duplicate, worth_checking)."""
    for speaker, result in dedup_results.items():
        global_indices = result["global_indices"]

        # Find which global indices are in checkable clusters
        # and which are non-representative members (duplicates)
        checkable_indices: set[int] = set()
        duplicate_indices: set[int] = set()

        for cluster in result["clusters"]:
            local_indices = cluster["member_indices"]
            member_global = [global_indices[li] for li in local_indices]

            if cluster.get("checkable", False):
                checkable_indices.update(member_global)

            # If cluster has >1 member, non-representative ones are duplicates
            if len(member_global) > 1:
                rep = cluster["representative"]
                rep_stmt = rep.get("thesis_statement", "")
                for gi in member_global:
                    if all_theses_with_meta[gi]["thesis_statement"] != rep_stmt:
                        duplicate_indices.add(gi)

        for gi, _ in speaker_theses[speaker]:
            all_theses_with_meta[gi]["is_duplicate"] = (
                gi in duplicate_indices
            )
            all_theses_with_meta[gi]["worth_checking"] = (
                gi in checkable_indices
            )
