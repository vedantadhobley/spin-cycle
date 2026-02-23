"""Initialise the database schema.

Run once to create all tables:
    python -m scripts.init_db
"""

import asyncio
import os

import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.db.models import Base

logger = structlog.get_logger()


async def init() -> None:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "spincycle")
    user = os.getenv("POSTGRES_USER", "spincycle")
    password = os.getenv("POSTGRES_PASSWORD", "spin-cycle-dev")
    database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
    logger.info("init_db", url=database_url.split("@")[-1])  # log host only

    engine = create_async_engine(database_url, echo=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    logger.info("init_db.done")


if __name__ == "__main__":
    asyncio.run(init())
