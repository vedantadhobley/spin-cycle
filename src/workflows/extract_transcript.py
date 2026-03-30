"""Temporal workflow for transcript claim extraction.

Two-phase extraction pipeline:
  Phase 1: Chunked extraction — split transcript into overlapping chunks,
    extract claims from each chunk in parallel (up to 2 at a time).
  Phase 2: Claim review — per-speaker sequential batch review with
    accumulating group context (ReviewClaimsWorkflow child).
  Phase 2b: Synthesis — per-group overarching claim generation
    (SynthesizeClaimsWorkflow child).

After review+synthesis, one Claim record is created per checkable group.
All member TranscriptClaims point to their group's Claim. Decompose
receives the synthesized overarching claim text.

Phases visible in Temporal UI:
  1. fetching           — downloading and parsing the transcript
  2. storing            — persisting transcript + enriching speakers
  3. extracting_claims  — chunked Phase 1 extraction
  4. reviewing_claims   — Phase 2 classify/group per speaker (child workflows)
  5. synthesizing       — Phase 2b overarching claims per group
  6. submitting         — creating claim records for verification
  7. verifying          — running child verification workflows
  8. complete           — done
"""

import asyncio
from collections import Counter
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        fetch_transcript,
        fetch_raw_transcript,
        extract_chunk_activity,
        store_transcript,
        store_transcript_claims,
        create_claims_for_transcript,
        update_transcript_status,
        finish_transcript_and_start_next,
        notify_frontend_refresh,
    )
    from src.workflows.verify import VerifyClaimWorkflow
    from src.workflows.review_claims import ReviewClaimsWorkflow
    from src.workflows.synthesize_claims import SynthesizeClaimsWorkflow
    from src.utils.logging import log
    from src.transcript.thesis_extractor import build_chunks
    from src.transcript.parsers import NumberedSegment
    from src.transcript.claim_reviewer import make_trivial_review

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
    3. Chunked extraction (Phase 1) — parallel chunks, up to 2 at a time
    4. Claim review (Phase 2) — per-speaker child ReviewClaimsWorkflow
    5. Synthesis (Phase 2b) — per-speaker child SynthesizeClaimsWorkflow
    6. Store all claims + create Claim records per checkable group
    7. Spawn child VerifyClaimWorkflow per checkable group (sequential)
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
            "segment_count": self._segment_count,
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
                - "fetch"   — fetch + parse only, return transcript data (no DB)
                - "store"   — fetch + store to DB with speaker enrichment
                - "phase1"  — Phase 1 extraction only, store raw theses
                - "review"  — Phase 1 + Phase 2 review, skip synthesis
                - "extract" — Phase 1 + Phase 2 + Phase 2b synthesis, skip verification
                - None      — full pipeline (default)

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

        if stop_after == "fetch":
            self._set_phase("complete")
            return {
                "url": url, "title": self._title,
                "word_count": self._word_count,
                "segment_count": self._segment_count,
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
                "segment_count": self._segment_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "enriched_speakers": enriched_speakers,
                "stopped_after": "store",
            }

        # Step 3: Chunked extraction (Phase 1)
        self._set_phase("extracting_claims")

        # Build chunks (deterministic, pure function — safe in workflow)
        segments = [
            NumberedSegment(
                index=s["index"], speaker=s["speaker"], text=s["text"],
                timestamp=s.get("timestamp"), section_header=s.get("section_header"),
            )
            for s in transcript_data["segments"]
        ]
        chunks = build_chunks(segments)

        log.info(workflow.logger, MODULE, "chunks_planned",
                 f"Planned {len(chunks)} chunks for extraction",
                 chunk_count=len(chunks))

        # Execute chunks in parallel pairs (up to 2 at a time = MAX_CONCURRENT)
        all_theses: list[dict] = []
        for i in range(0, len(chunks), 2):
            batch = chunks[i:i + 2]
            chunk_futures = []
            for chunk in batch:
                chunk_dict = {
                    "target_start": chunk.target_start,
                    "target_end": chunk.target_end,
                    "context_start": chunk.context_start,
                    "context_end": chunk.context_end,
                }
                chunk_futures.append(
                    workflow.execute_activity(
                        extract_chunk_activity,
                        args=[transcript_data, chunk_dict, enriched_speakers],
                        start_to_close_timeout=timedelta(seconds=2700),
                        retry_policy=RetryPolicy(maximum_attempts=2),
                    )
                )
            results = await asyncio.gather(*chunk_futures)
            for chunk_theses in results:
                all_theses.extend(chunk_theses)

        self._thesis_count = len(all_theses)

        log.info(workflow.logger, MODULE, "extraction_done",
                 f"Phase 1 complete: {len(all_theses)} claims extracted",
                 total=len(all_theses))

        # Step 3b: Store ALL raw theses as transcript_claims
        # (stored before review so they exist even if review fails)
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
                "segment_count": self._segment_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "all_theses": all_theses,
                "stopped_after": "phase1",
            }

        # Step 4: Claim review (Phase 2) — per speaker child workflows
        self._set_phase("reviewing_claims")

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

        transcript_date = transcript_data.get("date") or "unknown"

        # review_results: speaker → { dispositions, groups }
        review_results: dict[str, dict] = {}

        for speaker, indexed_theses in speaker_theses.items():
            sp_theses = [t for _, t in indexed_theses]
            sp_global_indices = [gi for gi, _ in indexed_theses]

            if len(sp_theses) <= 1:
                # Trivial: single claim, skip LLM
                review_result = make_trivial_review(sp_theses[0])
                log.info(workflow.logger, MODULE, "trivial_review",
                         f"Trivial review for {speaker} (1 claim, skipping LLM)",
                         speaker=speaker)
            else:
                log.info(workflow.logger, MODULE, "review_speaker_start",
                         f"Starting review for {speaker} "
                         f"({len(sp_theses)} claims)",
                         speaker=speaker, claim_count=len(sp_theses))

                review_result = await workflow.execute_child_workflow(
                    ReviewClaimsWorkflow.run,
                    args=[sp_theses, speaker, transcript_date],
                    id=f"review-{self._transcript_id}-{speaker[:30]}",
                    task_queue="spin-cycle-verify",
                )

                # Log per-speaker review summary
                sp_groups = review_result.get("groups", {})
                sp_checkable = sum(
                    1 for g in sp_groups.values() if g.get("checkable")
                )
                sp_action_counts = Counter(
                    d["action"] for d in review_result.get("dispositions", [])
                )
                log.info(workflow.logger, MODULE, "review_speaker_done",
                         f"Review done for {speaker}: "
                         f"{len(sp_groups)} groups ({sp_checkable} checkable), "
                         f"actions: {dict(sp_action_counts)}",
                         speaker=speaker,
                         groups=len(sp_groups),
                         checkable=sp_checkable,
                         action_counts=dict(sp_action_counts))

            # Remap speaker-local indices to global indices
            remapped = _remap_review_to_global(
                review_result, sp_global_indices
            )
            review_results[speaker] = remapped

        # Build consolidated group structures
        all_groups = _collect_all_groups(review_results, all_theses)

        # Tag theses with review metadata (classification, group membership)
        _apply_review_metadata(
            all_theses_with_meta, review_results, speaker_theses
        )

        checkable_groups = [g for g in all_groups if g["checkable"]]
        self._group_count = len(all_groups)

        # Log classification breakdown across all speakers
        all_disps = []
        for r in review_results.values():
            all_disps.extend(r.get("dispositions", []))
        class_counts = Counter(d["classification"] for d in all_disps)
        action_counts = Counter(d["action"] for d in all_disps)

        log.info(workflow.logger, MODULE, "review_done",
                 f"Phase 2 complete: {len(all_groups)} groups "
                 f"({len(checkable_groups)} checkable), "
                 f"classifications: {dict(class_counts)}, "
                 f"actions: {dict(action_counts)}",
                 total_groups=len(all_groups),
                 checkable_groups=len(checkable_groups),
                 classification_counts=dict(class_counts),
                 action_counts=dict(action_counts))

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
                "segment_count": self._segment_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "group_count": self._group_count,
                "groups": all_groups,
                "review_results": review_results,
                "stopped_after": "review",
            }

        # Step 5: Synthesis (Phase 2b) — per speaker child workflows
        self._set_phase("synthesizing")

        # Build synthesis input per speaker
        for speaker, indexed_theses in speaker_theses.items():
            sp_checkable = [
                g for g in checkable_groups if g["speaker"] == speaker
            ]
            if not sp_checkable:
                log.info(workflow.logger, MODULE, "synthesis_skip_speaker",
                         f"No checkable groups for {speaker}, skipping synthesis",
                         speaker=speaker)
                continue

            synth_groups = []
            for g in sp_checkable:
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
                     f"Synthesizing {len(synth_groups)} groups for {speaker}",
                     speaker=speaker, group_count=len(synth_groups),
                     group_ids=[g["local_group_id"] for g in sp_checkable])

            synth_results = await workflow.execute_child_workflow(
                SynthesizeClaimsWorkflow.run,
                args=[synth_groups, speaker],
                id=f"synthesize-{self._transcript_id}-{speaker[:30]}",
                task_queue="spin-cycle-verify",
            )

            # Apply synthesized claims to groups
            applied = 0
            for g in sp_checkable:
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
                     f"{applied}/{len(sp_checkable)} groups",
                     speaker=speaker,
                     applied=applied,
                     total=len(sp_checkable))

        self._claim_count = len(checkable_groups)
        self._claims = checkable_groups

        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        log.info(workflow.logger, MODULE, "synthesis_done",
                 f"Phase 2b complete: {len(checkable_groups)} overarching claims",
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
                "segment_count": self._segment_count,
                "speakers": self._speakers,
                "transcript_id": self._transcript_id,
                "thesis_count": self._thesis_count,
                "claim_count": self._claim_count,
                "group_count": self._group_count,
                "groups": all_groups,
                "stopped_after": "extract",
            }

        # Step 6: Create Claim records for checkable groups
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

                    # Collect supporting quote texts from ALL member refs
                    supporting_quotes = []
                    for ref in group.get("supporting_references", []):
                        seg_idx = ref.get("segment_index")
                        if seg_idx is not None:
                            for seg in transcript_data["segments"]:
                                if seg.get("index") == seg_idx:
                                    if seg["text"] not in supporting_quotes:
                                        supporting_quotes.append(seg["text"])
                                    break

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
            "segment_count": self._segment_count,
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
        entry["classification"] = "verifiable_fact"  # default, updated later
        entry["is_duplicate"] = False
        entry["worth_checking"] = False  # updated after review
        tagged.append(entry)
    return tagged


def _remap_review_to_global(
    review_result: dict,
    global_indices: list[int],
) -> dict:
    """Remap speaker-local indices in review result to global thesis indices.

    Args:
        review_result: Output from ReviewClaimsWorkflow or make_trivial_review.
            Has 'dispositions' (list) and 'groups' (dict).
        global_indices: Mapping from speaker-local index to global index.

    Returns:
        Same structure with indices remapped to global.
    """
    remapped_disps = []
    for d in review_result["dispositions"]:
        local_idx = d["claim_index"]
        remapped_disps.append({
            **d,
            "claim_index": global_indices[local_idx],
        })

    remapped_groups = {}
    for gid, g in review_result["groups"].items():
        remapped_groups[gid] = {
            **g,
            "member_indices": [
                global_indices[li] for li in g["member_indices"]
            ],
        }

    return {
        "dispositions": remapped_disps,
        "groups": remapped_groups,
    }


def _collect_all_groups(
    review_results: dict[str, dict],
    all_theses: list[dict],
) -> list[dict]:
    """Collect all groups from all speakers into a flat list.

    Group IDs are namespaced by speaker to avoid collisions (each
    speaker's ReviewClaimsWorkflow independently assigns G1, G2, etc.).
    The local_group_id is preserved so synthesis lookups still work.
    """
    all_groups = []

    for speaker, result in review_results.items():
        for gid, g in result["groups"].items():
            member_indices = g["member_indices"]

            # Collect supporting references from all members
            member_refs = []
            for gi in member_indices:
                t = all_theses[gi]
                for ref in t.get("supporting_references", []):
                    seg_idx = ref.get("segment_index")
                    if seg_idx is not None and not any(
                        r.get("segment_index") == seg_idx for r in member_refs
                    ):
                        member_refs.append(ref)
            member_refs.sort(key=lambda r: r.get("segment_index", 0))

            # Default claim_text is concatenation (replaced by synthesis later)
            member_stmts = [
                all_theses[gi]["thesis_statement"] for gi in member_indices
            ]

            all_groups.append({
                "group_id": f"{speaker}_{gid}",
                "local_group_id": gid,
                "speaker": speaker,
                "topic": g.get("topic", ""),
                "checkable": g.get("checkable", False),
                "checkability_rationale": g.get("checkability_rationale", ""),
                "member_global_indices": member_indices,
                "claim_text": "\n".join(member_stmts),
                "supporting_references": member_refs,
            })

    return all_groups


def _apply_review_metadata(
    all_theses_with_meta: list[dict],
    review_results: dict[str, dict],
    speaker_theses: dict[str, list[tuple[int, dict]]],
) -> None:
    """Tag stored theses with classification and worth_checking from review."""
    for speaker, result in review_results.items():
        # Build disposition lookup by global index
        disp_by_idx = {
            d["claim_index"]: d for d in result["dispositions"]
        }

        # Find which global indices are in checkable groups
        checkable_indices: set[int] = set()
        for gid, g in result["groups"].items():
            if g.get("checkable", False):
                checkable_indices.update(g["member_indices"])

        for gi, _ in speaker_theses[speaker]:
            disp = disp_by_idx.get(gi)
            if disp:
                all_theses_with_meta[gi]["classification"] = disp[
                    "classification"
                ]
                all_theses_with_meta[gi]["is_duplicate"] = (
                    disp["action"] == "duplicate"
                )
            all_theses_with_meta[gi]["worth_checking"] = (
                gi in checkable_indices
            )
