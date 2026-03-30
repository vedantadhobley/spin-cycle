"""Async embedding client for llama.cpp /v1/embeddings endpoint.

Raw httpx client — no LangChain. Reads LLAMA_EMBED_URL from env.
Batches texts per EMBEDDING_BATCH_SIZE. Returns numpy arrays.
"""

import os

import httpx
import numpy as np

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_TIMEOUT
from src.utils.logging import log, get_logger

MODULE = "embeddings"
logger = get_logger()


def _get_embed_url() -> str:
    url = os.getenv("LLAMA_EMBED_URL", "")
    if not url:
        raise RuntimeError("LLAMA_EMBED_URL not set")
    return url.rstrip("/")


async def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts via the llama.cpp embeddings endpoint.

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), dim) with L2-normalized embeddings.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    url = _get_embed_url()
    all_embeddings: list[list[float]] = []

    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]

            log.debug(logger, MODULE, "batch_request",
                      f"Embedding batch {i // EMBEDDING_BATCH_SIZE + 1}: "
                      f"{len(batch)} texts",
                      batch_size=len(batch), offset=i)

            resp = await client.post(
                f"{url}/v1/embeddings",
                json={"input": batch},
            )
            resp.raise_for_status()
            data = resp.json()

            # llama.cpp returns {"data": [{"embedding": [...], "index": N}, ...]}
            batch_embeddings = sorted(data["data"], key=lambda x: x["index"])
            for item in batch_embeddings:
                all_embeddings.append(item["embedding"])

    result = np.array(all_embeddings, dtype=np.float32)

    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    result = result / norms

    log.info(logger, MODULE, "embedded",
             f"Embedded {len(texts)} texts → shape {result.shape}",
             count=len(texts), dim=result.shape[1] if result.ndim > 1 else 0)

    return result
