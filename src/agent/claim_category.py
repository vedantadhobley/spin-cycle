"""Seed query routing for the research agent.

The decompose LLM writes targeted seed queries per fact — human-quality
search queries tailored to the specific evidence each fact needs. This
module wraps those LLM-provided queries in backend routing specs based
on the fact's categories, and adds a few mechanical base queries (the
raw sub-claim, Wikipedia, counter-evidence).

Categories (also from the decompose LLM) determine:
- Which SearXNG category to use (general, news, science)
- Whether to also route through Serper/Brave when available

The query specs are consumed by _run_seed_searches() in research.py,
which dispatches them to backends with availability fallback.
"""


# Map categories to SearXNG search categories.
# CURRENT_EVENTS → news, SCIENTIFIC → science, everything else → general.
_CATEGORY_TO_SEARXNG = {
    "CURRENT_EVENTS": "news",
    "SCIENTIFIC": "science",
}


def generate_seed_queries(
    sub_claim: str,
    categories: list[str],
    structure: str | None,
    seed_queries: list[str] | None = None,
) -> list[dict]:
    """Build seed query specs from LLM-provided queries + mechanical base queries.

    The LLM writes the search queries (good phrasing, targeted to the
    evidence need). This function handles backend routing:
    - Which backends to send each query to
    - Which SearXNG category to use
    - How many results to request
    - Deduplication

    Returns a list of query specs, each with:
        query: str — the search query
        backends: list[str] — which backends to dispatch to
        searxng_category: str — SearXNG category (general, news, science)
        max_results: int — max results per backend
        label: str — unique label for dedup and logging
    """
    specs: list[dict] = []

    # Determine the best SearXNG category from the fact's categories
    searxng_cat = "general"
    for cat in categories:
        if cat in _CATEGORY_TO_SEARXNG:
            searxng_cat = _CATEGORY_TO_SEARXNG[cat]
            break

    # ── Base queries (always generated, mechanical) ──────────────────
    # 1. Raw sub-claim → SearXNG (wide net)
    specs.append({
        "query": sub_claim[:120],
        "backends": ["searxng"],
        "searxng_category": searxng_cat,
        "max_results": 15,
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
                "backends": ["searxng"],
                "searxng_category": searxng_cat,
                "max_results": 10,
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
