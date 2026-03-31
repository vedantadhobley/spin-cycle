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
    )
    from src.utils.logging import log
    from src.config import MAX_CONCURRENT, TIMEOUT_SYNTHESIZE_CLAIM, TIMEOUT_STORE_CLAIMS

MODULE = "synthesize_claims_workflow"


@workflow.defn
class SynthesizeClaimsWorkflow:
    """Synthesize overarching claims for all speakers, create Claim records."""

    @workflow.run
    async def run(
        self,
        transcript_id: str,
        dedup_groups: list[dict],
        classified_theses: list[dict],
        enriched_speakers: list[dict],
        tc_ids: list[str],
        source_url: str | None = None,
        transcript_date: str | None = None,
        transcript_title: str | None = None,
        speaker_descriptions: dict | None = None,
    ) -> dict:
        speaker_descriptions = speaker_descriptions or {}
        checkable_groups = [g for g in dedup_groups if g["checkable"]]

        log.info(workflow.logger, MODULE, "started",
                 f"Synthesizing {len(checkable_groups)} checkable groups",
                 transcript_id=transcript_id,
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
                     f"Synthesizing {len(synth_groups)} multi-member groups "
                     f"({len(single_member)} single-member skipped)",
                     multi=len(synth_groups), single=len(single_member))

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
                 f"Synthesis complete: {len(checkable_groups)} claims",
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
                     f"Created {len(claim_ids)} Claim records",
                     claim_count=len(claim_ids))

        return {
            "checkable_groups": checkable_groups,
            "claim_ids": claim_ids,
        }
