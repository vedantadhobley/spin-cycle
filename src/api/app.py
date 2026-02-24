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
from temporalio.client import Client as TemporalClient  # noqa: E402

from src.api.routes.health import router as health_router  # noqa: E402
from src.api.routes.claims import router as claims_router  # noqa: E402
from src.db.session import engine  # noqa: E402
from src.db.models import Base  # noqa: E402

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")


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
