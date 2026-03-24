"""Transcript extraction endpoints.

Submit a transcript URL for claim extraction.  The extraction runs as a
Temporal workflow with full activity visibility in Temporal UI.

Queuing: Only one pipeline (extract or verify) runs at a time.  If a
pipeline is active, new submissions are queued and processed in order.
"""

import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.db.session import async_session
from src.db.models import TranscriptRecord
from src.utils.logging import log, get_logger
from src.workflows.extract_transcript import ExtractTranscriptWorkflow

MODULE = "transcripts"
logger = get_logger()

TASK_QUEUE = "spin-cycle-verify"

router = APIRouter()


class TranscriptSubmit(BaseModel):
    """Request body for submitting a transcript."""
    url: str = Field(..., description="Rev.com transcript URL")


class TranscriptResponse(BaseModel):
    """Response after submitting a transcript for extraction."""
    transcript_id: str | None = None
    workflow_id: str | None = None
    url: str
    status: str


async def _any_pipeline_running(temporal) -> bool:
    """Check if any extract or verify workflow is currently running."""
    for query in [
        'WorkflowType="ExtractTranscriptWorkflow" AND ExecutionStatus="Running"',
        'WorkflowType="VerifyClaimWorkflow" AND ExecutionStatus="Running"',
    ]:
        async for _ in temporal.list_workflows(query):
            return True
    return False


@router.post("", response_model=TranscriptResponse, status_code=201)
async def submit_transcript(
    body: TranscriptSubmit,
    request: Request,
):
    """Submit a transcript URL for claim extraction.

    Idempotent: re-submitting a URL that is queued/extracting/verifying returns
    the existing status.  Completed or failed transcripts can be re-submitted.
    """
    temporal = request.app.state.temporal

    # Check if this URL already exists
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.url == body.url)
        )
        existing = result.scalar_one_or_none()

        if existing and existing.status in ("queued", "extracting", "verifying"):
            # Already in progress — return current status (idempotent)
            log.info(logger, MODULE, "already_active",
                     "Transcript already in pipeline",
                     url=body.url, status=existing.status)
            return TranscriptResponse(
                transcript_id=str(existing.id),
                url=body.url,
                status=existing.status,
            )

    # New submission or re-submission of complete/failed transcript
    pipeline_busy = await _any_pipeline_running(temporal)

    if pipeline_busy:
        # Queue it — upsert transcript record with status="queued"
        async with async_session() as session:
            result = await session.execute(
                select(TranscriptRecord).where(TranscriptRecord.url == body.url)
            )
            record = result.scalar_one_or_none()

            if record:
                record.status = "queued"
            else:
                record = TranscriptRecord(
                    url=body.url,
                    title="(pending extraction)",
                    speakers=[],
                    word_count=0,
                    segment_count=0,
                    display_text="",
                    status="queued",
                )
                session.add(record)

            await session.commit()
            transcript_id = str(record.id)

        log.info(logger, MODULE, "queued",
                 "Transcript queued (pipeline busy)",
                 url=body.url, transcript_id=transcript_id)

        return TranscriptResponse(
            transcript_id=transcript_id,
            url=body.url,
            status="queued",
        )

    # Pipeline idle — start immediately
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.url == body.url)
        )
        record = result.scalar_one_or_none()

        if record:
            record.status = "extracting"
        else:
            record = TranscriptRecord(
                url=body.url,
                title="(pending extraction)",
                speakers=[],
                word_count=0,
                segment_count=0,
                display_text="",
                status="extracting",
            )
            session.add(record)

        await session.commit()
        transcript_id = str(record.id)

    workflow_id = f"extract-{transcript_id}"
    await temporal.start_workflow(
        ExtractTranscriptWorkflow.run,
        args=[body.url],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    log.info(logger, MODULE, "started",
             "Transcript extraction started",
             workflow_id=workflow_id, url=body.url,
             transcript_id=transcript_id)

    return TranscriptResponse(
        transcript_id=transcript_id,
        workflow_id=workflow_id,
        url=body.url,
        status="started",
    )
