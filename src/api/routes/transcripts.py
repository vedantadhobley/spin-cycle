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
    """Request body for submitting a transcript.

    Exactly one of `url` or `raw_text` is required:
    - url: Fetch and parse from a Rev.com transcript page
    - raw_text: Parse raw transcript text directly (requires title)
    """
    url: str | None = Field(None, description="Rev.com transcript URL")
    raw_text: str | None = Field(None, description="Raw transcript text (alternative to URL)")
    title: str | None = Field(None, description="Transcript title (required with raw_text)")
    date: str | None = Field(None, description="ISO date string (optional, used with raw_text)")


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
    """Submit a transcript URL or raw text for claim extraction.

    Accepts either:
    - url: Fetch and parse from Rev.com
    - raw_text + title: Parse raw transcript text directly

    Idempotent: re-submitting a URL that is queued/extracting/verifying returns
    the existing status.  Completed or failed transcripts can be re-submitted.
    """
    from fastapi import HTTPException

    # Validate: exactly one of url or raw_text
    if not body.url and not body.raw_text:
        raise HTTPException(400, "Either 'url' or 'raw_text' is required")
    if body.url and body.raw_text:
        raise HTTPException(400, "Provide either 'url' or 'raw_text', not both")
    if body.raw_text and not body.title:
        raise HTTPException(400, "'title' is required when using 'raw_text'")

    temporal = request.app.state.temporal

    # For raw text, generate a stable URL-like identifier from the title
    if body.raw_text:
        import hashlib
        content_hash = hashlib.sha256(body.raw_text.encode()).hexdigest()[:12]
        effective_url = f"raw://{content_hash}/{body.title}"
    else:
        effective_url = body.url

    # Check if this URL already exists
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.url == effective_url)
        )
        existing = result.scalar_one_or_none()

        if existing and existing.status in ("queued", "extracting", "verifying"):
            log.info(logger, MODULE, "already_active",
                     "Transcript already in pipeline",
                     url=effective_url, status=existing.status)
            return TranscriptResponse(
                transcript_id=str(existing.id),
                url=effective_url,
                status=existing.status,
            )

    # New submission or re-submission of complete/failed transcript
    pipeline_busy = await _any_pipeline_running(temporal)

    if pipeline_busy:
        async with async_session() as session:
            result = await session.execute(
                select(TranscriptRecord).where(TranscriptRecord.url == effective_url)
            )
            record = result.scalar_one_or_none()

            if record:
                record.status = "queued"
            else:
                record = TranscriptRecord(
                    url=effective_url,
                    title=body.title or "(pending extraction)",
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
                 url=effective_url, transcript_id=transcript_id)

        return TranscriptResponse(
            transcript_id=transcript_id,
            url=effective_url,
            status="queued",
        )

    # Pipeline idle — start immediately
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.url == effective_url)
        )
        record = result.scalar_one_or_none()

        if record:
            record.status = "extracting"
        else:
            record = TranscriptRecord(
                url=effective_url,
                title=body.title or "(pending extraction)",
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

    # Build workflow args
    workflow_args = [effective_url]
    if body.raw_text:
        workflow_args.extend([body.raw_text, body.title, body.date])

    await temporal.start_workflow(
        ExtractTranscriptWorkflow.run,
        args=workflow_args,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    log.info(logger, MODULE, "started",
             "Transcript extraction started",
             workflow_id=workflow_id, url=effective_url,
             transcript_id=transcript_id,
             mode="raw_text" if body.raw_text else "url")

    return TranscriptResponse(
        transcript_id=transcript_id,
        workflow_id=workflow_id,
        url=effective_url,
        status="started",
    )
