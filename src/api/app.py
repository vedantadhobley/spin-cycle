"""FastAPI application for Spin Cycle."""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from temporalio.client import Client as TemporalClient

from src.api.routes.health import router as health_router
from src.api.routes.claims import router as claims_router
from src.db.session import engine
from src.db.models import Base

logger = structlog.get_logger()

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    # Create DB tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db.ready")

    # Connect to Temporal
    temporal_client = await TemporalClient.connect(TEMPORAL_HOST)
    app.state.temporal = temporal_client
    logger.info("temporal.connected", host=TEMPORAL_HOST)

    yield

    # Cleanup
    await engine.dispose()
    logger.info("shutdown.complete")


app = FastAPI(
    title="Spin Cycle",
    description="News claim verification API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(claims_router, prefix="/claims", tags=["claims"])
