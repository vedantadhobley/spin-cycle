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
from src.workflows.transcript_pipeline import TranscriptPipelineWorkflow  # noqa: E402
from src.workflows.fetch_and_store import FetchTranscriptWorkflow  # noqa: E402
from src.workflows.extract_claims import ExtractClaimsWorkflow  # noqa: E402
from src.workflows.classify_and_dedup import ClassifyAndDedupWorkflow  # noqa: E402
from src.workflows.synthesize_claims import SynthesizeClaimsWorkflow  # noqa: E402
from src.workflows.verify_all_claims import VerifyClaimsWorkflow  # noqa: E402
from src.activities.verify_activities import (  # noqa: E402
    create_claim,
    decompose_claim,
    research_subclaim,
    judge_subclaim,
    synthesize_verdict,
    store_result,
    start_next_queued_claim,
)
from src.activities.transcript_activities import (  # noqa: E402
    fetch_transcript,
    fetch_raw_transcript,
    attribute_speakers,
    sentencize_and_chunk_activity,
    decontextualize_chunk_activity,
    embed_and_group_activity,
    synthesize_claims_activity,
    classify_claims_activity,
    dedup_claims_activity,
    synthesize_claim_activity,
    store_transcript,
    store_transcript_claims,
    create_claims_for_transcript,
    update_transcript_status,
    update_transcript_claims_classification,
    update_transcript_claims_dedup,
    finish_transcript_and_start_next,
    load_extract_inputs,
    load_classify_inputs,
    load_synthesize_inputs,
    load_verify_inputs,
    notify_frontend_refresh,
    queue_claims_for_verification,
)

from src.config import TASK_QUEUE, TEMPORAL_HOST, MAX_CONCURRENT


async def main():
    log.info(logger, MODULE, "starting", "Connecting to Temporal",
             temporal_host=TEMPORAL_HOST, task_queue=TASK_QUEUE)

    # Bootstrap MBFC index from REST API if needed (first startup ~25s, skip if fresh)
    from src.tools.mbfc_index import bootstrap_mbfc_index, is_bootstrap_needed
    if is_bootstrap_needed():
        log.info(logger, MODULE, "mbfc_bootstrap_start", "Bootstrapping MBFC index from API")
        count = await bootstrap_mbfc_index()
        log.info(logger, MODULE, "mbfc_bootstrap_done", "MBFC index ready", record_count=count)

    client = await Client.connect(TEMPORAL_HOST)
    log.info(logger, MODULE, "connected", "Connected to Temporal server",
             temporal_host=TEMPORAL_HOST)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            VerifyClaimWorkflow,
            TranscriptPipelineWorkflow,
            FetchTranscriptWorkflow,
            ExtractClaimsWorkflow,
            ClassifyAndDedupWorkflow,
            SynthesizeClaimsWorkflow,
            VerifyClaimsWorkflow,
        ],
        activities=[
            # Verification pipeline
            create_claim,
            decompose_claim,
            research_subclaim,
            judge_subclaim,
            synthesize_verdict,
            store_result,
            start_next_queued_claim,
            # Transcript extraction
            fetch_transcript,
            fetch_raw_transcript,
            attribute_speakers,
            sentencize_and_chunk_activity,
            decontextualize_chunk_activity,
            embed_and_group_activity,
            synthesize_claims_activity,
            classify_claims_activity,
            dedup_claims_activity,
            synthesize_claim_activity,
            store_transcript,
            store_transcript_claims,
            create_claims_for_transcript,
            update_transcript_status,
            update_transcript_claims_classification,
            update_transcript_claims_dedup,
            finish_transcript_and_start_next,
            # DB loaders (for independent workflow testing)
            load_extract_inputs,
            load_classify_inputs,
            load_synthesize_inputs,
            load_verify_inputs,
            # Frontend notification
            notify_frontend_refresh,
            # Claim status management
            queue_claims_for_verification,
        ],
        # Match MAX_CONCURRENT=2 in the workflow — 2 LLM inference slots
        # with 65K context each (2 slots x 65K = 131072 total ctx).
        max_concurrent_activities=MAX_CONCURRENT,
    )

    log.info(logger, MODULE, "ready", "Worker listening",
             task_queue=TASK_QUEUE, activity_count=26, workflow_count=7)
    try:
        await worker.run()
    finally:
        from src.transcript.cspan import cleanup
        await cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info(logger, MODULE, "stopped", "Worker stopped by keyboard interrupt")
