"""Temporal worker entrypoint.

Registers workflows and activities, then starts the worker
listening on the configured task queue.

Logging: Uses structured JSON logging for Grafana Loki.
Set LOG_FORMAT=pretty for development-friendly output.
"""

import asyncio
import os
import sys

# Force unbuffered output so Docker/Promtail sees logs immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

# Configure structured logging BEFORE importing temporalio
from src.utils.logging import configure_logging, get_logger, log  # noqa: E402

configure_logging()

MODULE = "worker"
logger = get_logger()

from temporalio.client import Client  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from src.workflows.verify import VerifyClaimWorkflow  # noqa: E402
from src.activities.verify_activities import (  # noqa: E402
    create_claim,
    decompose_claim,
    research_subclaim,
    judge_subclaim,
    synthesize_verdict,
    store_result,
    start_next_queued_claim,
)

TASK_QUEUE = "spin-cycle-verify"
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")


async def main():
    log.info(logger, MODULE, "starting", "Connecting to Temporal",
             temporal_host=TEMPORAL_HOST, task_queue=TASK_QUEUE)

    client = await Client.connect(TEMPORAL_HOST)
    log.info(logger, MODULE, "connected", "Connected to Temporal server",
             temporal_host=TEMPORAL_HOST)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[VerifyClaimWorkflow],
        activities=[
            create_claim,
            decompose_claim,
            research_subclaim,
            judge_subclaim,
            synthesize_verdict,
            store_result,
            start_next_queued_claim,
        ],
        # Serialize activity execution - only one LLM call at a time
        # Workflows may still interleave but activities (where LLM calls happen)
        # will be strictly sequential
        max_concurrent_activities=2,
    )

    log.info(logger, MODULE, "ready", "Worker listening",
             task_queue=TASK_QUEUE, activity_count=7, workflow_count=1)
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info(logger, MODULE, "stopped", "Worker stopped by keyboard interrupt")
