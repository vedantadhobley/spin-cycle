"""Transcript extraction endpoints.

Submit a C-SPAN transcript URL (or raw text) for claim extraction.  The
extraction runs as a Temporal workflow with full activity visibility in
Temporal UI.

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
from src.workflows.transcript_pipeline import TranscriptPipelineWorkflow
from src.config import TASK_QUEUE

MODULE = "transcripts"
logger = get_logger()

router = APIRouter()


class TranscriptSubmit(BaseModel):
    """Request body for submitting a transcript.

    Exactly one of `url`, `program_id`, or `raw_text` is required:
    - url: C-SPAN program URL
    - program_id: C-SPAN program ID (constructs URL automatically)
    - raw_text: Parse raw transcript text directly (requires title)
    """
    url: str | None = Field(None, description="C-SPAN program URL")
    program_id: str | None = Field(None, description="C-SPAN program ID")
    raw_text: str | None = Field(None, description="Raw transcript text (alternative to URL)")
    title: str | None = Field(None, description="Transcript title (required with raw_text)")
    date: str | None = Field(None, description="ISO date string (optional)")
    stop_after: str | None = Field(
        None,
        description="Early exit: 'fetch' (parse only), 'store' (+ DB), 'extract' (+ theses, skip verify)",
    )


class TranscriptResponse(BaseModel):
    """Response after submitting a transcript for extraction."""
    transcript_id: str | None = None
    workflow_id: str | None = None
    url: str
    status: str


async def _any_pipeline_running(temporal) -> bool:
    """Check if any extract or verify workflow is currently running."""
    for query in [
        'WorkflowType="TranscriptPipelineWorkflow" AND ExecutionStatus="Running"',
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
    """Submit a transcript URL, program ID, or raw text for claim extraction.

    Accepts exactly one of:
    - url: C-SPAN program URL
    - program_id: C-SPAN program ID (constructs URL automatically)
    - raw_text + title: Parse raw transcript text directly

    Idempotent: re-submitting a URL that is queued/extracting/verifying returns
    the existing status.  Completed or failed transcripts can be re-submitted.
    """
    from fastapi import HTTPException

    # Validate: exactly one of url, program_id, or raw_text
    sources = sum(1 for s in [body.url, body.program_id, body.raw_text] if s)
    if sources == 0:
        raise HTTPException(400, "One of 'url', 'program_id', or 'raw_text' is required")
    if sources > 1:
        raise HTTPException(400, "Provide exactly one of 'url', 'program_id', or 'raw_text'")
    if body.raw_text and not body.title:
        raise HTTPException(400, "'title' is required when using 'raw_text'")

    temporal = request.app.state.temporal

    # Resolve effective URL
    if body.program_id:
        effective_url = f"https://www.c-span.org/program/{body.program_id}"
    elif body.raw_text:
        import hashlib
        content_hash = hashlib.sha256(body.raw_text.encode()).hexdigest()[:12]
        effective_url = f"raw://{content_hash}/{body.title}"
    else:
        effective_url = body.url

    # Build workflow args: (url, raw_text, title, date, stop_after)
    workflow_args = [effective_url]
    if body.raw_text or body.stop_after:
        workflow_args.extend([
            body.raw_text,  # None if not raw_text
            body.title,
            body.date,
            body.stop_after,
        ])

    # When stop_after is set, skip queuing/DB — just fire the workflow.
    # The workflow handles early exit; no DB record avoids stuck state.
    if body.stop_after:
        workflow_id = f"test-extract-{uuid.uuid4().hex[:8]}"

        await temporal.start_workflow(
            TranscriptPipelineWorkflow.run,
            args=workflow_args,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        log.info(logger, MODULE, "test_started",
                 "Test workflow started",
                 workflow_id=workflow_id, url=effective_url,
                 stop_after=body.stop_after)

        return TranscriptResponse(
            workflow_id=workflow_id,
            url=effective_url,
            status="started",
        )

    # --- Production path: idempotency check + queuing ---

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

    await temporal.start_workflow(
        TranscriptPipelineWorkflow.run,
        args=workflow_args,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    log.info(logger, MODULE, "started",
             "Transcript extraction started",
             workflow_id=workflow_id, url=effective_url,
             transcript_id=transcript_id,
             mode="raw_text" if body.raw_text else "program_id" if body.program_id else "url")

    return TranscriptResponse(
        transcript_id=transcript_id,
        workflow_id=workflow_id,
        url=effective_url,
        status="started",
    )


@router.get("/discover")
async def discover_transcripts(limit: int = 50):
    """Discover available C-SPAN programs from the JW Player feed.

    Returns up to `limit` trending/recent programs with metadata.
    Results are cached for 5 minutes.
    """
    from src.transcript.cspan_discovery import fetch_available
    return await fetch_available(limit=limit)
