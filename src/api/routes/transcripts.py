"""Transcript extraction endpoints.

Submit a transcript URL for claim extraction.  The extraction runs as a
Temporal workflow with full activity visibility in Temporal UI.
"""

import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.utils.logging import log, get_logger
from src.workflows.extract_transcript import ExtractTranscriptWorkflow

MODULE = "transcripts"
logger = get_logger()

TASK_QUEUE = "spin-cycle-verify"

router = APIRouter()


class TranscriptSubmit(BaseModel):
    """Request body for submitting a transcript."""
    url: str = Field(..., description="Rev.com transcript URL")
    auto_verify: bool = Field(False, description="Auto-submit extracted claims for verification")


class TranscriptResponse(BaseModel):
    """Response after submitting a transcript for extraction."""
    workflow_id: str
    url: str
    status: str = "started"


@router.post("", response_model=TranscriptResponse, status_code=201)
async def submit_transcript(
    body: TranscriptSubmit,
    request: Request,
):
    """Submit a transcript URL for claim extraction.

    Starts an ExtractTranscriptWorkflow in Temporal.  Track progress
    in Temporal UI at :4501 (dev) or :3501 (prod).
    """
    temporal = request.app.state.temporal

    workflow_id = f"extract-{uuid.uuid4()}"

    await temporal.start_workflow(
        ExtractTranscriptWorkflow.run,
        args=[body.url, body.auto_verify],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    log.info(logger, MODULE, "workflow_started",
             "Transcript extraction workflow started",
             workflow_id=workflow_id, url=body.url,
             auto_verify=body.auto_verify)

    return TranscriptResponse(
        workflow_id=workflow_id,
        url=body.url,
    )
