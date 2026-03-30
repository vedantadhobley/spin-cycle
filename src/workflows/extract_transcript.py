"""Temporal workflow for transcript claim extraction.

Two-phase extraction pipeline:
  Phase 1: Chunked extraction — split transcript into overlapping chunks,
    extract claims from each chunk in parallel (up to 2 at a time).
  Phase 2: Claim review — per-speaker LLM call to classify, deduplicate,
    group related claims, and assess checkability.

After review, one Claim record is created per checkable group. All member
TranscriptClaims point to their group's Claim. Decompose receives the
concatenated group text + union of supporting quotes.

Transcript sources:
  - C-SPAN: Fetched via Playwright (CloudFront WAF), structured JSON segments
  - Raw text: Copy-pasted transcripts parsed by the raw_text parser

Phases visible in Temporal UI:
  1. fetching           — downloading and parsing the transcript
  2. storing            — persisting transcript + enriching speakers
  3. extracting_claims  — chunked Phase 1 extraction
  4. reviewing_claims   — Phase 2 classify/dedup/group per speaker
  5. submitting         — creating claim records for verification
  6. verifying          — running child verification workflows
  7. complete           — done
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
        review_claims_activity,
        store_transcript,
        store_transcript_claims,
        create_claims_for_transcript,
        update_transcript_status,
        finish_transcript_and_start_next,
        notify_frontend_refresh,
    )
    from src.workflows.verify import VerifyClaimWorkflow
    from src.utils.logging import log
    from src.transcript.thesis_extractor import build_chunks
    from src.transcript.parsers import NumberedSegment
    from src.transcript.claim_reviewer import make_trivial_review
    from src.schemas.llm_outputs import ExtractedThesis, SupportingReference

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
    4. Claim review (Phase 2) — per-speaker classify/dedup/group
    5. Store all claims + create Claim records per checkable group
    6. Spawn child VerifyClaimWorkflow per checkable group (sequential)
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
                - "extract" — fetch + store + extract + review, skip verification
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

        # Step 3b: Claim review (Phase 2) — per speaker
        self._set_phase("reviewing_claims")

        # Group theses by primary speaker
        speaker_theses: dict[str, list[dict]] = {}
        for t in all_theses:
            speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
            speaker_theses.setdefault(speaker, []).append(t)

        # review_results: list of (speaker, theses_for_speaker, review_dict)
        review_results: list[tuple[str, list[dict], dict]] = []

        for speaker, sp_theses in speaker_theses.items():
            if len(sp_theses) <= 1:
                # Trivial: single claim, skip LLM
                extracted = [ExtractedThesis(
                    thesis_statement=sp_theses[0]["thesis_statement"],
                    speakers=sp_theses[0].get("speakers", [speaker]),
                    supporting_references=[
                        SupportingReference(**r) for r in sp_theses[0].get("supporting_references", [])
                    ],
                    topic=sp_theses[0].get("topic", ""),
                )]
                trivial = make_trivial_review(extracted)
                review_results.append((speaker, sp_theses, trivial.model_dump()))
            else:
                transcript_date = transcript_data.get("date") or "unknown"
                review_dict = await workflow.execute_activity(
                    review_claims_activity,
                    args=[sp_theses, speaker, transcript_date],
                    start_to_close_timeout=timedelta(seconds=1800),
                    retry_policy=RetryPolicy(maximum_attempts=2),
                )
                review_results.append((speaker, sp_theses, review_dict))

        # Build final group structures from review results
        # Each group becomes one Claim for verification
        checkable_groups: list[dict] = []  # groups to verify
        all_theses_with_meta: list[dict] = []  # all theses for storage

        # Global thesis index → tc_id mapping will be built after storing
        global_thesis_idx = 0
        # thesis_to_group: maps global thesis index → group index (or -1 if not grouped)
        thesis_to_group: dict[int, int] = {}
        group_idx = 0

        for speaker, sp_theses, review_dict in review_results:
            classifications = review_dict.get("classifications", [])
            groups = review_dict.get("groups", [])

            # Build classification lookup (local index → classification)
            class_by_idx = {}
            for cls in classifications:
                class_by_idx[cls["claim_index"]] = cls

            # Tag all theses with metadata for storage
            for local_i, t in enumerate(sp_theses):
                cls = class_by_idx.get(local_i, {})
                t["thesis_version"] = 3
                t["claim_text"] = t["thesis_statement"]
                t["speaker"] = speaker
                t["classification"] = cls.get("classification", "verifiable_fact")
                t["is_duplicate"] = cls.get("is_duplicate", False)
                t["worth_checking"] = False  # default, set True below for grouped
                all_theses_with_meta.append(t)

            # Process groups
            for g in groups:
                member_local_indices = g.get("member_indices", [])
                is_checkable = g.get("checkable", False)

                # Map local indices to global and build group text
                member_global_indices = []
                member_statements = []
                member_refs = []
                for local_i in member_local_indices:
                    gi = global_thesis_idx + local_i
                    member_global_indices.append(gi)
                    thesis_to_group[gi] = group_idx

                    t = sp_theses[local_i]
                    member_statements.append(t["thesis_statement"])
                    # Mark as worth checking if group is checkable
                    if is_checkable:
                        all_theses_with_meta[gi]["worth_checking"] = True

                    # Collect references
                    for ref in t.get("supporting_references", []):
                        seg_idx = ref.get("segment_index")
                        if seg_idx is not None and not any(
                            r.get("segment_index") == seg_idx for r in member_refs
                        ):
                            member_refs.append(ref)

                member_refs.sort(key=lambda r: r.get("segment_index", 0))

                group_text = "\n".join(member_statements)

                checkable_groups.append({
                    "group_idx": group_idx,
                    "member_global_indices": member_global_indices,
                    "claim_text": group_text,
                    "speaker": speaker,
                    "topic": g.get("topic", ""),
                    "checkable": is_checkable,
                    "checkability_rationale": g.get("checkability_rationale", ""),
                    "group_rationale": g.get("group_rationale", ""),
                    "supporting_references": member_refs,
                })
                group_idx += 1

            global_thesis_idx += len(sp_theses)

        # Filter to only checkable groups for verification
        groups_to_verify = [g for g in checkable_groups if g["checkable"]]
        self._claim_count = len(groups_to_verify)
        self._group_count = len(checkable_groups)
        self._claims = groups_to_verify

        workflow.upsert_search_attributes([
            SA_CLAIM_COUNT.value_set(self._claim_count),
        ])

        log.info(workflow.logger, MODULE, "review_done",
                 f"Phase 2 complete: {len(checkable_groups)} groups "
                 f"({len(groups_to_verify)} checkable)",
                 total_groups=len(checkable_groups),
                 checkable_groups=len(groups_to_verify),
                 total_theses=len(all_theses_with_meta))

        if stop_after == "extract":
            # Store claims but skip verification
            if self._transcript_id and all_theses_with_meta:
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
                "groups": checkable_groups,
                "stopped_after": "extract",
            }

        # Step 4: Store ALL theses as transcript_claims
        if self._transcript_id and all_theses_with_meta:
            tc_ids = await workflow.execute_activity(
                store_transcript_claims,
                args=[self._transcript_id, all_theses_with_meta],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            # Step 5: Create Claim records for checkable groups
            if groups_to_verify:
                self._set_phase("submitting")

                # Build group dicts with member_tc_ids for create_claims_for_transcript
                group_dicts = []
                for g in groups_to_verify:
                    member_tc_ids = [tc_ids[i] for i in g["member_global_indices"]]
                    group_dicts.append({
                        "claim_text": g["claim_text"],
                        "speaker": g["speaker"],
                        "member_tc_ids": member_tc_ids,
                    })

                claim_ids = await workflow.execute_activity(
                    create_claims_for_transcript,
                    args=[self._transcript_id, tc_ids, group_dicts,
                          transcript_data.get("date"), self._title,
                          speaker_descriptions, url],
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
                for claim_id_str, group in zip(claim_ids, groups_to_verify):
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
                # No checkable groups — mark complete
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
            "group_count": self._group_count,
            "claims": groups_to_verify,
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
