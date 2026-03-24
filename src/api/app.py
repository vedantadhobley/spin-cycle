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
from src.api.routes.transcripts import router as transcripts_router  # noqa: E402
from src.db.session import engine, async_session  # noqa: E402
from src.db.models import Base, Claim  # noqa: E402
from src.workflows.verify import VerifyClaimWorkflow  # noqa: E402

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TASK_QUEUE = "spin-cycle-verify"


async def _register_search_attributes(temporal: TemporalClient):
    """Register custom search attributes for workflow visibility in Temporal UI.

    Idempotent — checks what's already registered and only adds missing ones.
    Called on every API startup.
    """
    from temporalio.api.operatorservice.v1 import (
        AddSearchAttributesRequest,
        ListSearchAttributesRequest,
    )
    from temporalio.api.enums.v1 import IndexedValueType

    # All custom search attributes used by our workflows
    required = {
        # VerifyClaimWorkflow
        "Phase": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
        "FactCount": IndexedValueType.INDEXED_VALUE_TYPE_INT,
        "ResearchProgress": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
        "JudgeProgress": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
        "Verdict": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
        "Confidence": IndexedValueType.INDEXED_VALUE_TYPE_DOUBLE,
        # ExtractTranscriptWorkflow
        "ClaimCount": IndexedValueType.INDEXED_VALUE_TYPE_INT,
        "TranscriptTitle": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
    }

    # Check what's already registered
    resp = await temporal.operator_service.list_search_attributes(
        ListSearchAttributesRequest(namespace="default")
    )
    existing = set(resp.custom_attributes.keys())

    missing = {k: v for k, v in required.items() if k not in existing}
    if not missing:
        log.info(logger, MODULE, "search_attrs_ok",
                 "All search attributes already registered",
                 count=len(required))
        return

    await temporal.operator_service.add_search_attributes(
        AddSearchAttributesRequest(
            namespace="default",
            search_attributes=missing,
        )
    )
    log.info(logger, MODULE, "search_attrs_registered",
             "Registered missing search attributes",
             registered=list(missing.keys()))


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
        claim_speaker = claim.speaker

        # Update status and start workflow
        claim.status = "pending"
        await session.commit()

        await temporal.start_workflow(
            VerifyClaimWorkflow.run,
            args=[claim_id, claim_text, claim_speaker],
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

    # Migrate: add new nullable columns to existing tables
    from sqlalchemy import text, inspect as sa_inspect
    async with engine.begin() as conn:
        def _migrate(sync_conn):
            inspector = sa_inspect(sync_conn)
            # Evidence table: new metadata columns
            ev_cols = {c["name"] for c in inspector.get_columns("evidence")}
            migrations = {
                "title": "VARCHAR(512)",
                "domain": "VARCHAR(256)",
                "bias": "VARCHAR(64)",
                "factual": "VARCHAR(64)",
                "tier": "VARCHAR(64)",
                "judge_index": "INTEGER",
                "assessment": "VARCHAR(32)",
                "is_independent": "BOOLEAN",
                "key_point": "TEXT",
            }
            for col, dtype in migrations.items():
                if col not in ev_cols:
                    sync_conn.execute(text(
                        f"ALTER TABLE evidence ADD COLUMN {col} {dtype}"
                    ))
            # Verdicts table: citations JSONB
            v_cols = {c["name"] for c in inspector.get_columns("verdicts")}
            if "citations" not in v_cols:
                sync_conn.execute(text(
                    "ALTER TABLE verdicts ADD COLUMN citations JSONB"
                ))
            # Claims table: speaker column
            c_cols = {c["name"] for c in inspector.get_columns("claims")}
            if "speaker" not in c_cols:
                sync_conn.execute(text(
                    "ALTER TABLE claims ADD COLUMN speaker VARCHAR(256)"
                ))
        await conn.run_sync(_migrate)

    log.info(logger, MODULE, "db_ready", "Database tables ready")

    # Bootstrap MBFC index from REST API if needed
    from src.tools.mbfc_index import bootstrap_mbfc_index, is_bootstrap_needed
    if is_bootstrap_needed():
        log.info(logger, MODULE, "mbfc_bootstrap_start", "Bootstrapping MBFC index from API")
        count = await bootstrap_mbfc_index()
        log.info(logger, MODULE, "mbfc_bootstrap_done", "MBFC index ready", record_count=count)

    # Connect to Temporal
    temporal_client = await TemporalClient.connect(TEMPORAL_HOST)
    app.state.temporal = temporal_client
    log.info(logger, MODULE, "temporal_connected", "Connected to Temporal",
             temporal_host=TEMPORAL_HOST)

    # Register custom search attributes (idempotent — skips existing)
    await _register_search_attributes(temporal_client)

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
app.include_router(transcripts_router, prefix="/transcripts", tags=["transcripts"])
