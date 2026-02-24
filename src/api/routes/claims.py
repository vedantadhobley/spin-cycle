"""Claim submission and query endpoints."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.data.schemas import ClaimSubmit, ClaimResponse, VerdictResponse, ClaimListResponse, SubClaimResponse
from src.db.models import Claim, SubClaim, Verdict
from src.db.session import get_session
from src.utils.logging import log, get_logger
from src.workflows.verify import VerifyClaimWorkflow

MODULE = "claims"
logger = get_logger()

TASK_QUEUE = "spin-cycle-verify"

router = APIRouter()


@router.post("", response_model=ClaimResponse, status_code=201)
async def submit_claim(
    body: ClaimSubmit,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Submit a claim for verification."""
    claim = Claim(
        text=body.text,
        source_url=body.source,
        source_name=body.source_name,
        status="pending",
    )
    session.add(claim)
    await session.commit()
    await session.refresh(claim)

    claim_id = str(claim.id)
    log.info(logger, MODULE, "submitted", "Claim submitted",
             claim_id=claim_id, text=body.text[:80])

    # Kick off Temporal workflow
    temporal = request.app.state.temporal
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
        raise HTTPException(status_code=404, detail="Claim not found")

    sub_claim_responses = [
        SubClaimResponse(
            text=sc.text,
            verdict=sc.verdict,
            confidence=sc.confidence,
            reasoning=sc.reasoning,
            nuance=sc.nuance,
            evidence_count=len(sc.evidence),
        )
        for sc in claim.sub_claims
    ]

    return VerdictResponse(
        id=str(claim.id),
        text=claim.text,
        status=claim.status,
        source=claim.source_url,
        source_name=claim.source_name,
        verdict=claim.verdict.verdict if claim.verdict else None,
        confidence=claim.verdict.confidence if claim.verdict else None,
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
                nuance=c.verdict.nuance if c.verdict else None,
                sub_claims=[
                    SubClaimResponse(
                        text=sc.text,
                        verdict=sc.verdict,
                        confidence=sc.confidence,
                        reasoning=sc.reasoning,
                        nuance=sc.nuance,
                        evidence_count=len(sc.evidence),
                    )
                    for sc in c.sub_claims
                ],
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in claims
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
