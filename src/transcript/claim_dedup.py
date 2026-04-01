"""Embedding-based claim deduplication.

Replaces the broken LLM-based Phase 2 grouping with fast, deterministic
cosine similarity clustering using union-find.

Includes a numeric divergence guard: claims that are structurally identical
but contain different numbers (e.g. "lasted 34 years" vs "lasted 22 years")
are NOT merged, even if embedding similarity is high. Embeddings are blind
to numeric specificity.
"""

from __future__ import annotations

import re

import numpy as np

from src.config import EMBEDDING_SIMILARITY_THRESHOLD, NUMERIC_SKELETON_JACCARD_THRESHOLD
from src.llm.embeddings import embed_texts
from src.utils.logging import log, get_logger

MODULE = "claim_dedup"
_default_logger = get_logger()


class UnionFind:
    """Union-find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Numeric divergence guard
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r'\b\d[\d,]*(?:\.\d+)?%?\b')


def _extract_numbers(text: str) -> set[str]:
    """Extract normalized number strings from text.

    Keeps raw string form (not float) so "2,700" and "2700" match
    but "28,000" and "32,000" don't.
    """
    raw = _NUMBER_RE.findall(text)
    # Normalize: strip commas, strip trailing %
    return {n.replace(",", "").rstrip("%") for n in raw}


def _skeleton(text: str) -> set[str]:
    """Strip numbers from text and return word token set."""
    stripped = _NUMBER_RE.sub("", text).lower()
    return set(stripped.split())


def _numeric_divergence_blocks_merge(a: str, b: str) -> bool:
    """Return True if a and b are structurally similar but numerically different.

    This catches cases where embeddings score 0.98 for
    "war lasted 34 years" vs "war lasted 22 years".
    """
    nums_a = _extract_numbers(a)
    nums_b = _extract_numbers(b)

    # No numbers in either → no numeric divergence possible
    if not nums_a and not nums_b:
        return False

    # Same numbers → no divergence
    if nums_a == nums_b:
        return False

    # Different numbers — check if the sentences are structurally similar
    skel_a = _skeleton(a)
    skel_b = _skeleton(b)

    if not skel_a or not skel_b:
        return False

    intersection = skel_a & skel_b
    union = skel_a | skel_b
    jaccard = len(intersection) / len(union)

    return jaccard >= NUMERIC_SKELETON_JACCARD_THRESHOLD


async def dedup_speaker_claims(
    theses: list[dict],
    speaker: str,
    logger=None,
) -> dict:
    """Deduplicate claims for one speaker using embedding cosine similarity.

    Args:
        theses: List of thesis dicts (must have thesis_statement, original_quote,
                topic). May also have classification, checkable, check_rationale
                (added by classification phase).
        speaker: Speaker name (for logging).

    Returns:
        Dict with "clusters": list of cluster dicts, each containing:
            - representative: full thesis dict (longest original_quote)
            - member_indices: list[int] — indices into input theses
            - checkable: bool — any member checkable
            - topic: str — from representative
    """
    logger = logger or _default_logger
    n = len(theses)
    if n == 0:
        return {"clusters": []}

    if n == 1:
        return {"clusters": [{
            "representative": theses[0],
            "member_indices": [0],
            "checkable": theses[0].get("checkable", True),
            "topic": theses[0].get("topic", ""),
        }]}

    # Extract statements and embed
    statements = [t["thesis_statement"] for t in theses]

    log.info(logger, MODULE, "embedding_start",
             "Embedding claims for speaker",
             speaker=speaker, count=n)

    embeddings = await embed_texts(statements)

    # Cosine similarity matrix (embeddings are already L2-normalized)
    sim_matrix = embeddings @ embeddings.T

    # Union-find clustering
    uf = UnionFind(n)
    merge_count = 0
    blocked_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= EMBEDDING_SIMILARITY_THRESHOLD:
                if _numeric_divergence_blocks_merge(statements[i], statements[j]):
                    blocked_count += 1
                    log.info(logger, MODULE, "numeric_divergence_block",
                             "Blocked merge due to numeric divergence",
                             claim_a=statements[i][:80],
                             claim_b=statements[j][:80],
                             similarity=float(sim_matrix[i, j]))
                    continue
                uf.union(i, j)
                merge_count += 1

    # Group by connected component
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        components.setdefault(root, []).append(i)

    # Build cluster output
    clusters = []
    for member_indices in components.values():
        # Pick representative: longest original_quote
        rep_idx = max(
            member_indices,
            key=lambda i: len(theses[i].get("original_quote", "")),
        )

        # Cluster is checkable if ANY member is checkable
        checkable = any(
            theses[i].get("checkable", True) for i in member_indices
        )

        clusters.append({
            "representative": theses[rep_idx],
            "member_indices": member_indices,
            "checkable": checkable,
            "topic": theses[rep_idx].get("topic", ""),
        })

    multi_member = sum(1 for c in clusters if len(c["member_indices"]) > 1)

    log.info(logger, MODULE, "dedup_done",
             "Dedup complete for speaker",
             speaker=speaker,
             input_count=n,
             cluster_count=len(clusters),
             multi_member=multi_member,
             merge_count=merge_count,
             blocked_count=blocked_count,
             threshold=EMBEDDING_SIMILARITY_THRESHOLD)

    return {"clusters": clusters}
