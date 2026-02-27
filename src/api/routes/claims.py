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

from src.schemas import ClaimSubmit, ClaimResponse, VerdictResponse, ClaimListResponse, SubClaimResponse
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
        node = SubClaimResponse(
            text=sc.text,
            is_leaf=sc.is_leaf,
            verdict=sc.verdict,
            confidence=sc.confidence,
            reasoning=sc.reasoning,
            nuance=sc.nuance,
            evidence_count=len(sc.evidence) if sc.evidence else 0,
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
        status=initial_status,
    )
    session.add(claim)
    await session.commit()
    await session.refresh(claim)

    claim_id = str(claim.id)
    
    if initial_status == "queued":
        # Get queue position
        queue_position = await count_queued_claims(session)
        log.info(logger, MODULE, "queued", "Claim queued for verification",
                 claim_id=claim_id, queue_position=queue_position)
    else:
        # No workflow running — start immediately
        await temporal.start_workflow(
            VerifyClaimWorkflow.run,
            args=[claim_id, body.text],
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
        verdict=claim.verdict.verdict if claim.verdict else None,
        confidence=claim.verdict.confidence if claim.verdict else None,
        reasoning=claim.verdict.reasoning if claim.verdict else None,
        nuance=claim.verdict.nuance if claim.verdict else None,
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
                verdict=c.verdict.verdict if c.verdict else None,
                confidence=c.verdict.confidence if c.verdict else None,
                reasoning=c.verdict.reasoning if c.verdict else None,
                nuance=c.verdict.nuance if c.verdict else None,
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
