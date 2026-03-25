"""Claim submission and query endpoints.

Claims are queued to ensure only one verification runs at a time,
preventing LLM contention. The workflow triggers the next queued
claim when it completes.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from temporalio.client import Client as TemporalClient

from src.schemas import ClaimSubmit, ClaimBatchSubmit, ClaimBatchResponse, ClaimResponse, VerdictResponse, ClaimListResponse, SubClaimResponse, EvidenceResponse, CitationResponse
from src.db.models import Claim, SubClaim, Verdict
from src.db.session import get_session
from src.utils.logging import log, get_logger
from src.workflows.verify import VerifyClaimWorkflow

MODULE = "claims"
logger = get_logger()

TASK_QUEUE = "spin-cycle-verify"


async def count_running_workflows(temporal: TemporalClient) -> int:
    """Count currently running verification workflows."""
    count = 0
    async for _ in temporal.list_workflows(
        'WorkflowType="VerifyClaimWorkflow" AND ExecutionStatus="Running"'
    ):
        count += 1
    return count


async def count_queued_claims(session: AsyncSession) -> int:
    """Count claims waiting in queue."""
    result = await session.execute(
        select(func.count(Claim.id)).where(Claim.status == "queued")
    )
    return result.scalar() or 0


router = APIRouter()


def _build_citations(verdict) -> list[CitationResponse]:
    """Build CitationResponse list from Verdict JSONB citations."""
    if not verdict or not verdict.citations:
        return []
    return [
        CitationResponse(
            index=c.get("index", 0),
            url=c.get("url"),
            title=c.get("title"),
            domain=c.get("domain"),
        )
        for c in verdict.citations
    ]


def _build_sub_claim_tree(sub_claims) -> list[SubClaimResponse]:
    """Build a nested tree of SubClaimResponse from flat DB rows with parent_id.

    DB stores a flat list with parent_id references. This function
    reconstructs the tree by:
      1. Indexing all sub-claims by ID
      2. Finding root nodes (parent_id is None)
      3. Recursively attaching children
    """
    by_id = {}
    roots = []

    # Index all sub-claims
    for sc in sub_claims:
        evidence_list = []
        if sc.evidence:
            for e in sorted(sc.evidence, key=lambda e: e.judge_index or 999):
                evidence_list.append(EvidenceResponse(
                    judge_index=e.judge_index,
                    url=e.source_url,
                    title=e.title,
                    domain=e.domain,
                    source_type=e.source_type or "web",
                    bias=e.bias,
                    factual=e.factual,
                    tier=e.tier,
                    assessment=e.assessment,
                    is_independent=e.is_independent,
                    key_point=e.key_point,
                ))
        node = SubClaimResponse(
            text=sc.text,
            is_leaf=sc.is_leaf,
            verdict=sc.verdict,
            confidence=sc.confidence,
            reasoning=sc.reasoning,
            evidence=evidence_list,
            children=[],
        )
        by_id[sc.id] = (node, sc.parent_id)

    # Build tree
    for sc_id, (node, parent_id) in by_id.items():
        if parent_id is None:
            roots.append(node)
        elif parent_id in by_id:
            by_id[parent_id][0].children.append(node)

    return roots


@router.post("", response_model=ClaimResponse, status_code=201)
async def submit_claim(
    body: ClaimSubmit,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit a claim for verification.

    Claims are queued to ensure sequential processing (one at a time).
    If no workflow is running, starts immediately. Otherwise, queued.
    """
    temporal = request.app.state.temporal

    # Check if any verification is currently running
    running_count = await count_running_workflows(temporal)

    # Determine initial status
    if running_count > 0:
        initial_status = "queued"
    else:
        initial_status = "pending"

    claim = Claim(
        text=body.text,
        source_url=body.source,
        source_name=body.source_name,
        speaker=body.speaker,
        speaker_description=body.speaker_description,
        claim_date=body.claim_date,
        transcript_title=body.transcript_title,
        status=initial_status,
    )
    session.add(claim)
    await session.commit()
    await session.refresh(claim)

    claim_id = str(claim.id)

    if initial_status == "queued":
        queue_position = await count_queued_claims(session)
        log.info(logger, MODULE, "queued", "Claim queued for verification",
                 claim_id=claim_id, queue_position=queue_position)
    else:
        await temporal.start_workflow(
            VerifyClaimWorkflow.run,
            args=[claim_id, body.text, body.speaker, body.claim_date,
                  False, body.transcript_title, body.speaker_description or ""],
            id=f"verify-{claim_id}",
            task_queue=TASK_QUEUE,
        )
        log.info(logger, MODULE, "workflow_started", "Verification workflow started",
                 claim_id=claim_id)

    return ClaimResponse(
        id=claim_id,
        text=claim.text,
        status=claim.status,
        created_at=claim.created_at,
    )


@router.post("/batch", response_model=ClaimBatchResponse, status_code=201)
async def submit_claims_batch(
    body: ClaimBatchSubmit,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit multiple claims for verification.

    Inserts all claims into the database. If no workflow is currently
    running, starts the first claim immediately; the rest are queued.
    """
    temporal = request.app.state.temporal

    running_count = await count_running_workflows(temporal)
    started_first = False
    responses = []

    for item in body.claims:
        if running_count == 0 and not started_first:
            initial_status = "pending"
        else:
            initial_status = "queued"

        claim = Claim(
            text=item.text,
            source_url=item.source,
            source_name=item.source_name,
            speaker=item.speaker,
            speaker_description=item.speaker_description,
            claim_date=item.claim_date,
            transcript_title=item.transcript_title,
            status=initial_status,
        )
        session.add(claim)
        await session.commit()
        await session.refresh(claim)

        claim_id = str(claim.id)

        if initial_status == "pending":
            await temporal.start_workflow(
                VerifyClaimWorkflow.run,
                args=[claim_id, item.text, item.speaker, item.claim_date,
                      False, item.transcript_title, item.speaker_description or ""],
                id=f"verify-{claim_id}",
                task_queue=TASK_QUEUE,
            )
            started_first = True
            log.info(logger, MODULE, "batch_workflow_started",
                     "Batch: first claim workflow started",
                     claim_id=claim_id)
        else:
            log.info(logger, MODULE, "batch_queued",
                     "Batch: claim queued",
                     claim_id=claim_id)

        responses.append(ClaimResponse(
            id=claim_id,
            text=claim.text,
            status=claim.status,
            created_at=claim.created_at,
        ))

    log.info(logger, MODULE, "batch_submitted",
             "Batch submission complete",
             total=len(responses),
             started=1 if started_first else 0,
             queued=len(responses) - (1 if started_first else 0))

    return ClaimBatchResponse(claims=responses)


@router.get("/{claim_id}", response_model=VerdictResponse)
async def get_claim(
    claim_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get claim status and verdict."""
    try:
        claim_uuid = uuid.UUID(claim_id)
    except ValueError:
        log.warning(logger, MODULE, "invalid_id", "Invalid claim ID format",
                    claim_id=claim_id)
        raise HTTPException(status_code=400, detail="Invalid claim ID")

    result = await session.execute(
        select(Claim)
        .where(Claim.id == claim_uuid)
        .options(
            selectinload(Claim.sub_claims).selectinload(SubClaim.evidence),
            selectinload(Claim.verdict),
        )
    )
    claim = result.scalar_one_or_none()
    if not claim:
        log.info(logger, MODULE, "not_found", "Claim not found",
                 claim_id=claim_id)
        raise HTTPException(status_code=404, detail="Claim not found")

    log.debug(logger, MODULE, "get_claim", "Claim retrieved",
              claim_id=claim_id, status=claim.status)

    sub_claim_responses = _build_sub_claim_tree(claim.sub_claims)

    return VerdictResponse(
        id=str(claim.id),
        text=claim.text,
        status=claim.status,
        source=claim.source_url,
        source_name=claim.source_name,
        speaker=claim.speaker,
        verdict=claim.verdict.verdict if claim.verdict else None,
        confidence=claim.verdict.confidence if claim.verdict else None,
        reasoning=claim.verdict.reasoning if claim.verdict else None,
        citations=_build_citations(claim.verdict),
        sub_claims=sub_claim_responses,
        created_at=claim.created_at,
        updated_at=claim.updated_at,
    )


@router.get("", response_model=ClaimListResponse)
async def list_claims(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List claims with filters."""
    query = select(Claim).options(
        selectinload(Claim.sub_claims).selectinload(SubClaim.evidence),
        selectinload(Claim.verdict),
    )
    count_query = select(func.count(Claim.id))

    if status:
        query = query.where(Claim.status == status)
        count_query = count_query.where(Claim.status == status)

    query = query.order_by(Claim.created_at.desc()).limit(limit).offset(offset)

    result = await session.execute(query)
    claims = result.scalars().all()

    total_result = await session.execute(count_query)
    total = total_result.scalar()

    return ClaimListResponse(
        claims=[
            VerdictResponse(
                id=str(c.id),
                text=c.text,
                status=c.status,
                source=c.source_url,
                source_name=c.source_name,
                speaker=c.speaker,
                verdict=c.verdict.verdict if c.verdict else None,
                confidence=c.verdict.confidence if c.verdict else None,
                reasoning=c.verdict.reasoning if c.verdict else None,
                citations=_build_citations(c.verdict),
                sub_claims=_build_sub_claim_tree(c.sub_claims),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in claims
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
