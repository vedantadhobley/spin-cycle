"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "spin-cycle", "version": "0.1.0"}


@router.get("/")
async def root():
    return {"service": "spin-cycle", "version": "0.1.0"}
