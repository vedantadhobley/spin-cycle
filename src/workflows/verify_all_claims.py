"""Shell workflow for sequential claim verification.

Runs VerifyClaimWorkflow as a child for each claim, one at a time.
Each child gets both LLM slots (semaphore 2 inside VerifyClaimWorkflow).
If a single claim fails after retries, it is skipped and the rest continue.

Formerly VerifyAllClaimsWorkflow, renamed to VerifyClaimsWorkflow.
"""

from temporalio import workflow

from datetime import timedelta
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.workflows.verify import VerifyClaimWorkflow
    from src.activities.transcript_activities import load_verify_inputs
    from src.utils.logging import log
    from src.config import TASK_QUEUE, TIMEOUT_STORE_CLAIMS

MODULE = "verify_claims"


@workflow.defn
class VerifyClaimsWorkflow:
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
    async def run(
        self,
        claims: list[dict] | None = None,
        transcript_id: str | None = None,
    ) -> dict:
        """Verify claims sequentially.

        Args:
            claims: List of dicts, each with claim_id, claim_text, speaker, etc.
                    If None, loads from DB using transcript_id.
            transcript_id: Load claims linked to this transcript from DB.

        Returns:
            Dict with verified/failed/skipped counts and failed claim IDs.
        """
        # Load from DB if claims not provided
        if claims is None:
            if transcript_id is None:
                raise ValueError("Either claims or transcript_id must be provided")
            claims = await workflow.execute_activity(
                load_verify_inputs,
                args=[transcript_id],
                start_to_close_timeout=timedelta(seconds=TIMEOUT_STORE_CLAIMS),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        self._total = len(claims)
        failed_ids: list[str] = []

        log.info(workflow.logger, MODULE, "started",
                 "Starting sequential verification",
                 total=len(claims))

        for i, claim in enumerate(claims):
            claim_id = claim["claim_id"]
            self._current_claim = claim_id

            log.info(workflow.logger, MODULE, "claim_start",
                     "Verifying claim",
                     claim_id=claim_id,
                     index=i + 1, total=len(claims),
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
                        claim.get("transcript_description", ""),
                    ],
                    id=f"verify-{claim_id}",
                    task_queue=TASK_QUEUE,
                )
                self._verified += 1
            except Exception as e:
                self._failed += 1
                failed_ids.append(claim_id)
                log.warning(workflow.logger, MODULE, "claim_failed",
                            "Claim verification failed, skipping",
                            claim_id=claim_id, error=str(e))

        self._current_claim = ""

        log.info(workflow.logger, MODULE, "complete",
                 "Verification complete",
                 verified=self._verified, failed=self._failed)

        return {
            "verified": self._verified,
            "failed": self._failed,
            "total": self._total,
            "failed_ids": failed_ids,
        }
