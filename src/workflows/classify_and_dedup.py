"""Classify and dedup extracted claims — Phase 3 of the pipeline.

Phase 1: Batch classification → UPDATE classification fields on transcript_claims.
Phase 2: Per-speaker embedding dedup → UPDATE dedup flags on transcript_claims.
"""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import (
        classify_claims_activity,
        dedup_claims_activity,
        update_transcript_claims_classification,
        update_transcript_claims_dedup,
    )
    from src.utils.logging import log
    from src.config import (
        TIMEOUT_CLASSIFY_CLAIMS,
        TIMEOUT_DEDUP_CLAIMS,
        CLASSIFY_BATCH_SIZE,
    )

MODULE = "classify_and_dedup"


@workflow.defn
class ClassifyAndDedupWorkflow:
    """Classify claims by checkability, then dedup per speaker."""

    @workflow.run
    async def run(
        self,
        transcript_id: str,
        tc_ids: list[str],
        all_theses: list[dict],
        enriched_speakers: list[dict],
    ) -> dict:
        log.info(workflow.logger, MODULE, "started",
                 f"Classifying {len(all_theses)} claims",
                 transcript_id=transcript_id)

        # --- Phase 1: Batch classification ---
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
            # Merge classification fields back
            for i, c in enumerate(classified):
                idx = batch_start + i
                all_theses[idx]["classification"] = c.get(
                    "classification", "verifiable_fact"
                )
                all_theses[idx]["checkable"] = c.get("checkable", True)
                all_theses[idx]["check_rationale"] = c.get(
                    "check_rationale", ""
                )
                if "factual_anchor" in c:
                    all_theses[idx]["factual_anchor"] = c["factual_anchor"]

        # UPDATE classification in DB
        if tc_ids:
            class_updates = []
            for i, t in enumerate(all_theses):
                if i < len(tc_ids):
                    class_updates.append({
                        "tc_id": tc_ids[i],
                        "classification": t.get("classification"),
                        "checkable": t.get("checkable"),
                        "checkability_rationale": t.get("check_rationale", ""),
                        "factual_anchor": t.get("factual_anchor"),
                    })
            await workflow.execute_activity(
                update_transcript_claims_classification,
                args=[transcript_id, class_updates],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        log.info(workflow.logger, MODULE, "classify_done",
                 f"Classification complete for {len(all_theses)} claims")

        # --- Phase 2: Per-speaker embedding dedup ---
        speaker_theses: dict[str, list[tuple[int, dict]]] = {}
        for global_i, t in enumerate(all_theses):
            speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
            speaker_theses.setdefault(speaker, []).append((global_i, t))

        dedup_results: dict[str, dict] = {}
        for speaker, indexed_theses in speaker_theses.items():
            sp_theses = [t for _, t in indexed_theses]
            sp_global_indices = [gi for gi, _ in indexed_theses]

            if len(sp_theses) <= 1:
                dedup_results[speaker] = {
                    "clusters": [{
                        "representative": sp_theses[0],
                        "member_indices": [0],
                        "checkable": sp_theses[0].get("checkable", True),
                        "topic": sp_theses[0].get("topic", ""),
                    }],
                    "global_indices": sp_global_indices,
                }
            else:
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

        # Build group structures
        all_groups = _collect_groups_from_dedup(dedup_results, all_theses)
        checkable_groups = [g for g in all_groups if g["checkable"]]

        # Compute dedup metadata and UPDATE DB
        dedup_updates = _build_dedup_updates(
            tc_ids, dedup_results, speaker_theses, all_theses
        )
        if dedup_updates:
            await workflow.execute_activity(
                update_transcript_claims_dedup,
                args=[transcript_id, dedup_updates],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        log.info(workflow.logger, MODULE, "complete",
                 f"Classify+dedup complete: {len(all_groups)} groups "
                 f"({len(checkable_groups)} checkable)",
                 total_groups=len(all_groups),
                 checkable=len(checkable_groups))

        return {
            "classified_theses": all_theses,
            "dedup_groups": all_groups,
            "tc_ids": tc_ids,
        }


# ---------------------------------------------------------------------------
# Helpers (deterministic, safe for workflow code)
# ---------------------------------------------------------------------------

def _collect_groups_from_dedup(
    dedup_results: dict[str, dict],
    all_theses: list[dict],
) -> list[dict]:
    """Build flat group list from per-speaker dedup clusters."""
    all_groups = []

    for speaker, result in dedup_results.items():
        global_indices = result["global_indices"]

        for ci, cluster in enumerate(result["clusters"]):
            local_indices = cluster["member_indices"]
            member_global = [global_indices[li] for li in local_indices]

            # Collect original quotes (deduped)
            original_quotes = []
            seen_quotes: set[str] = set()
            for gi in member_global:
                quote = all_theses[gi].get("original_quote", "")
                if quote and quote not in seen_quotes:
                    original_quotes.append(quote)
                    seen_quotes.add(quote)

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


def _build_dedup_updates(
    tc_ids: list[str],
    dedup_results: dict[str, dict],
    speaker_theses: dict[str, list[tuple[int, dict]]],
    all_theses: list[dict],
) -> list[dict]:
    """Build update dicts for dedup flags on transcript_claims."""
    if not tc_ids:
        return []

    # Determine which global indices are duplicates and which are checkable
    checkable_indices: set[int] = set()
    duplicate_indices: set[int] = set()

    for speaker, result in dedup_results.items():
        global_indices = result["global_indices"]

        for cluster in result["clusters"]:
            local_indices = cluster["member_indices"]
            member_global = [global_indices[li] for li in local_indices]

            if cluster.get("checkable", False):
                checkable_indices.update(member_global)

            if len(member_global) > 1:
                rep = cluster["representative"]
                rep_stmt = rep.get("thesis_statement", "")
                for gi in member_global:
                    if all_theses[gi]["thesis_statement"] != rep_stmt:
                        duplicate_indices.add(gi)

    updates = []
    for speaker, indexed_theses in speaker_theses.items():
        for gi, _ in indexed_theses:
            if gi < len(tc_ids):
                updates.append({
                    "tc_id": tc_ids[gi],
                    "is_duplicate": gi in duplicate_indices,
                    "worth_checking": gi in checkable_indices,
                })

    return updates
