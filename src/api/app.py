"""FastAPI application for Spin Cycle.

Logging: Uses structured JSON logging for Grafana Loki.
Set LOG_FORMAT=pretty for development-friendly output.
"""

import os
import sys
from contextlib import asynccontextmanager

# Force unbuffered output so Docker/Promtail sees logs immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

# Configure structured logging BEFORE importing anything else
from src.utils.logging import configure_logging, get_logger, log  # noqa: E402

configure_logging()

MODULE = "api"
logger = get_logger()

from fastapi import FastAPI  # noqa: E402
from sqlalchemy import select  # noqa: E402
from temporalio.client import Client as TemporalClient  # noqa: E402

from src.api.routes.health import router as health_router  # noqa: E402
from src.api.routes.claims import router as claims_router  # noqa: E402
from src.db.session import engine, async_session  # noqa: E402
from src.db.models import Base, Claim  # noqa: E402
from src.workflows.verify import VerifyClaimWorkflow  # noqa: E402

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TASK_QUEUE = "spin-cycle-verify"


async def _kickstart_queue(temporal: TemporalClient):
    """Start first queued claim if queue is stalled (no running workflows).
    
    Called on API startup to handle edge cases like restarts where
    queued claims exist but nothing is processing them.
    """
    # Check for running workflows
    running_count = 0
    async for _ in temporal.list_workflows(
        'WorkflowType="VerifyClaimWorkflow" AND ExecutionStatus="Running"'
    ):
        running_count += 1
        break  # Just need to know if any are running
    
    if running_count > 0:
        log.info(logger, MODULE, "queue_active", 
                 "Workflow already running, queue is active")
        return
    
    # No running workflows — check for queued claims
    async with async_session() as session:
        result = await session.execute(
            select(Claim)
            .where(Claim.status == "queued")
            .order_by(Claim.created_at.asc())
            .limit(1)
        )
        claim = result.scalar_one_or_none()
        
        if not claim:
            log.info(logger, MODULE, "queue_empty", "No queued claims")
            return
        
        claim_id = str(claim.id)
        claim_text = claim.text
        
        # Update status and start workflow
        claim.status = "pending"
        await session.commit()
        
        await temporal.start_workflow(
            VerifyClaimWorkflow.run,
            args=[claim_id, claim_text],
            id=f"verify-{claim_id}",
            task_queue=TASK_QUEUE,
        )
        log.info(logger, MODULE, "queue_kickstart", 
                 "Started queued claim on startup",
                 claim_id=claim_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    # Create DB tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info(logger, MODULE, "db_ready", "Database tables ready")

    # Connect to Temporal
    temporal_client = await TemporalClient.connect(TEMPORAL_HOST)
    app.state.temporal = temporal_client
    log.info(logger, MODULE, "temporal_connected", "Connected to Temporal",
             temporal_host=TEMPORAL_HOST)

    # Kickstart queue if needed (handles restarts with orphaned queued claims)
    await _kickstart_queue(temporal_client)

    yield

    # Cleanup
    await engine.dispose()
    log.info(logger, MODULE, "shutdown", "Application shutdown complete")


app = FastAPI(
    title="Spin Cycle",
    description="News claim verification API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(claims_router, prefix="/claims", tags=["claims"])
