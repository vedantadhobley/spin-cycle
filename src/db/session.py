"""Database session management."""

import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "spincycle")
POSTGRES_USER = os.getenv("POSTGRES_USER", "spincycle")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "spin-cycle-dev")

DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Get an async database session."""
    async with async_session() as session:
        yield session
