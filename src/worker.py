"""Temporal worker entrypoint.

Registers workflows and activities, then starts the worker
listening on the configured task queue.
"""

import asyncio
import os

import structlog
from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.verify import VerifyClaimWorkflow
from src.activities.verify_activities import (
    decompose_claim,
    research_subclaim,
    judge_subclaim,
    synthesize_verdict,
    store_result,
)

logger = structlog.get_logger()

TASK_QUEUE = "spin-cycle-verify"
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")


async def main():
    logger.info("worker.starting", temporal_host=TEMPORAL_HOST, task_queue=TASK_QUEUE)

    client = await Client.connect(TEMPORAL_HOST)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[VerifyClaimWorkflow],
        activities=[
            decompose_claim,
            research_subclaim,
            judge_subclaim,
            synthesize_verdict,
            store_result,
        ],
    )

    logger.info("worker.ready", task_queue=TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
