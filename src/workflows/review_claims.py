"""Temporal workflow for sequential claim review with accumulating groups.

Processes all claims for one speaker in batches of ~10. Each batch is a
separate Temporal activity that sees accumulated group context from prior
batches. Groups grow across batches — later batches can assign claims to
groups created earlier.
"""

from collections import Counter
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.activities.transcript_activities import review_batch_activity
    from src.transcript.claim_reviewer import REVIEW_BATCH_SIZE
    from src.utils.logging import log

MODULE = "review_claims_workflow"


@workflow.defn
class ReviewClaimsWorkflow:
    """Sequential batch review with accumulating group state.

    Input: all raw Phase 1 claims for one speaker + transcript date.
    Output: classifications for every claim + group definitions.
    """

    @workflow.run
    async def run(self, claims: list[dict], speaker: str,
                  current_date: str) -> dict:
        """Run sequential review batches.

        Args:
            claims: Serialized ExtractedThesis dicts for this speaker.
            speaker: Speaker name.
            current_date: ISO date string.

        Returns:
            Dict with:
              - dispositions: list of all claim dispositions (global indices)
              - groups: dict of group_id → group state
        """
        all_dispositions: list[dict] = []
        groups: dict[str, dict] = {}  # group_id → state

        total = len(claims)
        batch_count = (total + REVIEW_BATCH_SIZE - 1) // REVIEW_BATCH_SIZE
        log.info(workflow.logger, MODULE, "started",
                 f"Reviewing {total} claims for {speaker} "
                 f"in {batch_count} batches of ~{REVIEW_BATCH_SIZE}",
                 speaker=speaker, claim_count=total,
                 batch_count=batch_count)

        for batch_start in range(0, total, REVIEW_BATCH_SIZE):
            batch_end = min(batch_start + REVIEW_BATCH_SIZE, total)
            batch_claims = claims[batch_start:batch_end]
            batch_num = batch_start // REVIEW_BATCH_SIZE
            batch_label = f"b{batch_num}"

            # Prepare groups summary for context
            groups_summary = {}
            for gid, g in groups.items():
                groups_summary[gid] = {
                    "topic": g["topic"],
                    "member_count": len(g["member_indices"]),
                    "representative_statement": g["representative_statement"],
                }

            existing_group_ids = list(groups.keys())

            log.info(workflow.logger, MODULE, "batch_starting",
                     f"Batch {batch_num+1}/{batch_count}: "
                     f"claims [{batch_start}:{batch_end}], "
                     f"{len(existing_group_ids)} existing groups",
                     speaker=speaker, batch=batch_label,
                     batch_start=batch_start, batch_end=batch_end,
                     existing_groups=len(existing_group_ids))

            # Each batch is a separate activity for Temporal visibility
            result = await workflow.execute_activity(
                review_batch_activity,
                args=[batch_claims, speaker, current_date,
                      groups_summary, existing_group_ids, batch_label],
                start_to_close_timeout=timedelta(seconds=600),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

            # Process dispositions — remap local indices to global
            action_counts: Counter = Counter()
            groups_modified: set[str] = set()

            for disp in result["dispositions"]:
                global_index = disp["claim_index"] + batch_start
                global_disp = {**disp, "claim_index": global_index}
                all_dispositions.append(global_disp)
                action_counts[disp["action"]] += 1

                if disp["action"] == "new_group":
                    # Find matching new_group definition
                    ng_def = next(
                        ng for ng in result["new_groups"]
                        if ng["group_id"] == disp["group_id"]
                    )
                    groups[disp["group_id"]] = {
                        "topic": ng_def["topic"],
                        "checkable": ng_def["checkable"],
                        "checkability_rationale": ng_def["checkability_rationale"],
                        "member_indices": [global_index],
                        "representative_statement": claims[global_index].get(
                            "thesis_statement", ""
                        ),
                    }
                    groups_modified.add(disp["group_id"])

                elif disp["action"] == "add_to_group":
                    groups[disp["group_id"]]["member_indices"].append(
                        global_index
                    )
                    groups_modified.add(disp["group_id"])

            log.info(workflow.logger, MODULE, "batch_done",
                     f"Batch {batch_num+1}/{batch_count} done: "
                     f"{dict(action_counts)}, "
                     f"total groups: {len(groups)}",
                     speaker=speaker, batch=batch_label,
                     action_counts=dict(action_counts),
                     groups_modified=list(groups_modified),
                     total_groups=len(groups))

        # Final summary
        checkable = sum(1 for g in groups.values() if g.get("checkable"))
        multi_member = sum(
            1 for g in groups.values() if len(g["member_indices"]) > 1
        )

        log.info(workflow.logger, MODULE, "complete",
                 f"Review complete for {speaker}: "
                 f"{len(groups)} groups ({checkable} checkable, "
                 f"{multi_member} multi-member), "
                 f"{len(all_dispositions)} dispositions",
                 speaker=speaker,
                 dispositions=len(all_dispositions),
                 total_groups=len(groups),
                 checkable_groups=checkable,
                 multi_member_groups=multi_member)

        return {
            "dispositions": all_dispositions,
            "groups": {gid: g for gid, g in groups.items()},
        }
