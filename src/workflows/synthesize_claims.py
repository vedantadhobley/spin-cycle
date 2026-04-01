"""Synthesize claims and create Claim records — Phase 4 of the pipeline.

Takes ALL dedup groups (all speakers), synthesizes multi-member clusters
in parallel pairs (2 LLM slots), passes through singletons. Then creates
Claim records and links TranscriptClaim FKs.
"""

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        synthesize_claim_activity,
        create_claims_for_transcript,
        load_synthesize_inputs,
    )
    from src.utils.logging import log
    from src.config import MAX_CONCURRENT, TIMEOUT_SYNTHESIZE_CLAIM, TIMEOUT_STORE_CLAIMS

MODULE = "synthesize_claims"


@workflow.defn
class SynthesizeClaimsWorkflow:
    """Synthesize overarching claims for all speakers, create Claim records."""

    @workflow.run
    async def run(
        self,
        transcript_id: str,
        dedup_groups: list[dict] | None = None,
        classified_theses: list[dict] | None = None,
        enriched_speakers: list[dict] | None = None,
        tc_ids: list[str] | None = None,
        source_url: str | None = None,
        transcript_date: str | None = None,
        transcript_title: str | None = None,
        transcript_description: str | None = None,
        speaker_descriptions: dict | None = None,
    ) -> dict:
        # Load from DB if inputs not provided (standalone mode)
        if dedup_groups is None or classified_theses is None:
            loaded = await workflow.execute_activity(
                load_synthesize_inputs,
                args=[transcript_id],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_STORE_CLAIMS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            dedup_groups = dedup_groups or loaded["dedup_groups"]
            classified_theses = classified_theses or loaded["classified_theses"]
            enriched_speakers = enriched_speakers or loaded["enriched_speakers"]
            tc_ids = tc_ids or loaded["tc_ids"]
            source_url = source_url or loaded.get("source_url")
            transcript_date = transcript_date or loaded.get("transcript_date")
            transcript_title = transcript_title or loaded.get("transcript_title")
            transcript_description = transcript_description or loaded.get("transcript_description")
            speaker_descriptions = speaker_descriptions or loaded.get("speaker_descriptions", {})

        speaker_descriptions = speaker_descriptions or {}
        checkable_groups = [g for g in dedup_groups if g["checkable"]]

        log.info(workflow.logger, MODULE, "started",
                 "Starting synthesis",
                 transcript_id=transcript_id,
                 checkable_groups=len(checkable_groups),
                 total_groups=len(dedup_groups))

        if not checkable_groups:
            log.info(workflow.logger, MODULE, "no_checkable",
                     "No checkable groups to synthesize")
            return {"checkable_groups": [], "claim_ids": []}

        # Separate single-member (no LLM needed) from multi-member
        single_member = [
            g for g in checkable_groups
            if len(g["member_global_indices"]) == 1
        ]
        multi_member = [
            g for g in checkable_groups
            if len(g["member_global_indices"]) > 1
        ]

        # Single-member: use thesis_statement directly
        for g in single_member:
            gi = g["member_global_indices"][0]
            g["claim_text"] = classified_theses[gi]["thesis_statement"]

        # Multi-member: synthesize in parallel pairs (2 LLM slots)
        if multi_member:
            synth_groups = []
            for g in multi_member:
                member_stmts = [
                    classified_theses[gi]["thesis_statement"]
                    for gi in g["member_global_indices"]
                ]
                synth_groups.append({
                    "group_id": g["local_group_id"],
                    "topic": g["topic"],
                    "speaker": g["speaker"],
                    "member_statements": member_stmts,
                })

            log.info(workflow.logger, MODULE, "synth_multi",
                     "Synthesizing multi-member groups",
                     transcript_id=transcript_id,
                     multi=len(synth_groups),
                     single=len(single_member))

            sem = asyncio.Semaphore(MAX_CONCURRENT)
            synth_results: dict[str, dict] = {}

            async def synth_with_sem(sg: dict) -> tuple[str, dict]:
                async with sem:
                    result = await workflow.execute_activity(
                        synthesize_claim_activity,
                        args=[
                            sg["member_statements"],
                            sg["topic"],
                            sg["speaker"],
                            transcript_title or "",
                            transcript_description or "",
                            transcript_date or "",
                        ],
                        start_to_close_timeout=timedelta(seconds=TIMEOUT_SYNTHESIZE_CLAIM),
                        retry_policy=RetryPolicy(maximum_attempts=2),
                    )
                    return sg["group_id"], result

            results = await asyncio.gather(*(
                synth_with_sem(sg) for sg in synth_groups
            ))
            for group_id, result in results:
                synth_results[group_id] = result

            for g in multi_member:
                synth = synth_results.get(g["local_group_id"])
                if synth:
                    g["claim_text"] = synth["overarching_claim"]
                else:
                    log.warning(workflow.logger, MODULE, "synthesis_missing",
                                f"No synthesis for group {g['local_group_id']}",
                                group_id=g["local_group_id"])

        log.info(workflow.logger, MODULE, "synthesis_done",
                 "Synthesis complete",
                 transcript_id=transcript_id,
                 checkable=len(checkable_groups))

        # Create Claim records and link TranscriptClaim FKs
        claim_ids: list[str] = []
        if tc_ids and checkable_groups:
            group_dicts = []
            for g in checkable_groups:
                member_tc_ids = [
                    tc_ids[i] for i in g["member_global_indices"]
                ]
                group_dicts.append({
                    "claim_text": g["claim_text"],
                    "speaker": g["speaker"],
                    "member_tc_ids": member_tc_ids,
                    "original_quotes": g.get("original_quotes", []),
                })

            claim_ids = await workflow.execute_activity(
                create_claims_for_transcript,
                args=[
                    transcript_id, tc_ids, group_dicts,
                    transcript_date, transcript_title,
                    speaker_descriptions, source_url,
                ],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_STORE_CLAIMS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            log.info(workflow.logger, MODULE, "claims_created",
                     "Claim records created",
                     transcript_id=transcript_id,
                     claim_count=len(claim_ids))

        return {
            "checkable_groups": checkable_groups,
            "claim_ids": claim_ids,
        }
