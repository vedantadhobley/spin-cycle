"""Seed query routing for the research agent.

The decompose LLM writes targeted seed queries per fact — human-quality
search queries tailored to the specific evidence each fact needs. This
module wraps those LLM-provided queries in backend routing specs based
on the fact's categories, and adds mechanical base queries (raw sub-claim
+ Wikipedia).

The query specs are consumed by _run_seed_searches() in research.py,
which dispatches them to backends with availability fallback.

Backends: Serper (primary — Google results via API) + DuckDuckGo (free
fallback, always available) + SearXNG (optional padding, self-hosted
meta-search). If Serper is unavailable, seeds fall back to DDG.
"""


def generate_seed_queries(
    sub_claim: str,
    categories: list[str],
    seed_queries: list[str] | None = None,
) -> list[dict]:
    """Build seed query specs from LLM-provided queries + mechanical base queries.

    The LLM writes the search queries (good phrasing, targeted to the
    evidence need). This function handles backend routing:
    - Which backends to send each query to
    - How many results to request
    - Deduplication

    Returns a list of query specs, each with:
        query: str — the search query
        backends: list[str] — which backends to dispatch to
        searxng_category: str — kept for interface compat, unused by DDG
        max_results: int — max results per backend
        label: str — unique label for dedup and logging
    """
    specs: list[dict] = []

    # ── Base queries (always generated, mechanical) ──────────────────
    # 1. Raw sub-claim → Serper (Google, primary) + DDG (fallback) + SearXNG (padding)
    specs.append({
        "query": sub_claim[:120],
        "backends": ["serper", "duckduckgo", "searxng"],
        "searxng_category": "",
        "max_results": 20,
        "label": "primary",
    })

    # 2. Wikipedia — stable background knowledge
    specs.append({
        "query": sub_claim,
        "backends": ["wikipedia"],
        "searxng_category": "",
        "max_results": 3,
        "label": "wikipedia",
    })

    # ── LLM-provided seed queries ────────────────────────────────────
    if seed_queries:
        for i, query in enumerate(seed_queries):
            query = query.strip()
            if not query:
                continue
            specs.append({
                "query": query[:120],
                "backends": ["serper", "duckduckgo", "searxng"],
                "searxng_category": "",
                "max_results": 15,
                "label": f"seed_{i}",
            })

    # ── Dedup by (query, backends) — same query to different backends
    # produces different results, so only dedup exact duplicates ──────
    seen: set[tuple[str, tuple]] = set()
    deduped: list[dict] = []
    for spec in specs:
        key = (spec["query"].strip().lower(), tuple(sorted(spec["backends"])))
        if key not in seen:
            seen.add(key)
            deduped.append(spec)

    return deduped
