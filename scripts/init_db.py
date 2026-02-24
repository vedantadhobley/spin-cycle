"""Initialise the database schema.

Run once to create all tables:
    python -m scripts.init_db
"""

import asyncio
import os

from sqlalchemy.ext.asyncio import create_async_engine

from src.db.models import Base
from src.utils.logging import configure_logging, log, get_logger

configure_logging()
MODULE = "init_db"
logger = get_logger()


async def init() -> None:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "spincycle")
    user = os.getenv("POSTGRES_USER", "spincycle")
    password = os.getenv("POSTGRES_PASSWORD", "spin-cycle-dev")
    database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
    log.info(logger, MODULE, "start", "Initializing database",
             url=database_url.split("@")[-1])

    engine = create_async_engine(database_url, echo=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    log.info(logger, MODULE, "done", "Database initialized")


if __name__ == "__main__":
    asyncio.run(init())
