"""Shell workflow for sequential claim verification.

Runs VerifyClaimWorkflow as a child for each claim, one at a time.
Each child gets both LLM slots (semaphore 2 inside VerifyClaimWorkflow).
If a single claim fails after retries, it is skipped and the rest continue.
"""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.workflows.verify import VerifyClaimWorkflow
    from src.utils.logging import log
    from src.config import TASK_QUEUE

MODULE = "verify_all_claims"


@workflow.defn
class VerifyAllClaimsWorkflow:
    """Sequentially verify all claims for a transcript."""

    def __init__(self) -> None:
        self._verified = 0
        self._failed = 0
        self._total = 0
        self._current_claim = ""

    @workflow.query
    def status(self) -> dict:
        return {
            "verified": self._verified,
            "failed": self._failed,
            "total": self._total,
            "current_claim": self._current_claim,
        }

    @workflow.run
    async def run(self, claims: list[dict]) -> dict:
        """Verify claims sequentially.

        Args:
            claims: List of dicts, each with:
                - claim_id: str
                - claim_text: str
                - speaker: str
                - speaker_description: str
                - transcript_date: str
                - transcript_title: str
                - supporting_quotes: list[str]

        Returns:
            Dict with verified/failed/skipped counts and failed claim IDs.
        """
        self._total = len(claims)
        failed_ids: list[str] = []

        log.info(workflow.logger, MODULE, "started",
                 f"Verifying {len(claims)} claims sequentially",
                 total=len(claims))

        for i, claim in enumerate(claims):
            claim_id = claim["claim_id"]
            self._current_claim = claim_id

            log.info(workflow.logger, MODULE, "claim_start",
                     f"Verifying claim {i + 1}/{len(claims)}",
                     claim_id=claim_id,
                     speaker=claim.get("speaker"))

            try:
                await workflow.execute_child_workflow(
                    VerifyClaimWorkflow.run,
                    args=[
                        claim_id,
                        claim["claim_text"],
                        claim.get("speaker"),
                        claim.get("transcript_date"),
                        True,  # is_child — skip queue chaining
                        claim.get("transcript_title"),
                        claim.get("speaker_description", ""),
                        claim.get("supporting_quotes", []),
                    ],
                    id=f"verify-{claim_id}",
                    task_queue=TASK_QUEUE,
                )
                self._verified += 1
            except Exception as e:
                self._failed += 1
                failed_ids.append(claim_id)
                log.warning(workflow.logger, MODULE, "claim_failed",
                            f"Claim {claim_id} failed, skipping",
                            claim_id=claim_id, error=str(e))

        self._current_claim = ""

        log.info(workflow.logger, MODULE, "complete",
                 f"Verification complete: {self._verified} verified, "
                 f"{self._failed} failed",
                 verified=self._verified, failed=self._failed)

        return {
            "verified": self._verified,
            "failed": self._failed,
            "total": self._total,
            "failed_ids": failed_ids,
        }
