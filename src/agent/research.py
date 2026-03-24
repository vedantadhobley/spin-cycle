"""Two-phase evidence gathering: programmatic seed search + LangGraph ReAct agent.

## Architecture

Phase 1 — Programmatic seed search (~3-20s):
  1. _run_seed_searches(): Fire LLM-written queries + base queries (raw claim,
     Wikipedia) to Serper/DuckDuckGo concurrently. Yields ~30-50 raw URLs.
  2. _collect_domains() + await_ratings_parallel(): Warm MBFC cache for all
     seed domains (per-domain 15s httpx timeout).
  3. _enrich_parties_from_mbfc(): Extract owner names from MBFC ownership
     fields via SpaCy NER, Wikidata-expand to discover networks/media holdings.
     New parties/media influence conflict detection in the next step.
  4. _rank_and_filter_seeds(): Score URLs with score_url(), detect interested-party
     conflicts (affiliated media + publisher ownership), apply CONFLICT_PENALTY (-15)
     to conflicted sources, sort by adjusted score, keep top 30.
  5. _prefetch_seed_pages(): Fetch full page content from top-ranked seeds
     in parallel (~3-5s). Up to 10 pages, TIER 1/2 first. Saves the agent
     3-4 tool calls that would otherwise go to reading predictable sources.
  6. _build_seed_messages(): Package ranked seeds as synthetic AIMessage +
     ToolMessage pairs so the agent sees them as prior searches.

Phase 2 — LangGraph ReAct agent (~60-90s):

    ┌────────────┐     ┌──────────┐     ┌───────┐
    │ pre_model  │────▶│  agent   │────▶│ tools │
    │ (progress) │     │  (LLM)   │◀────│       │
    └────────────┘     └────┬─────┘     └───────┘
                            │ (no more tool calls)
                            ▼
                           END

  The agent starts with seed results and pre-fetched articles in its history.
  It spends its 8-12 tool call budget on targeted follow-up searches and
  fetching additional sources rather than reading the top-ranked seeds.

  The pre_model hook injects a progress note each iteration: tool call count,
  unique URLs/domains, seed tier/conflict coverage, and queries used.

Phase 3 — Programmatic enrichment:
  - _enrich_with_legislation(): LegiScan search for matching legislation,
    bill text, roll call votes, sponsors. Appended to agent evidence.
  - _enrich_parties_from_evidence_content(): SpaCy NER on evidence articles
    → Wikidata-expand new entities → add if graph overlaps existing parties.

## Agent tools

Dynamically registered based on API keys:
  - serper_search: Google results via Serper API (primary, needs SERPER_API_KEY)
  - web_search: DuckDuckGo (fallback, always available)
  - brave_search: Brave Search index (optional, needs BRAVE_API_KEY)
  - wikipedia_search: Wikipedia API (always available)
  - fetch_page_content: Read full page text from URLs (always available)

## Integration with Temporal

Runs inside the research_subclaim Temporal activity. Temporal provides
durable execution, timeouts, retries, and observability.
"""

import re as _re
import time as _time
import uuid as _uuid
from datetime import date
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.llm import get_llm
from src.prompts.verification import RESEARCH_SYSTEM, RESEARCH_USER, build_claim_date_line
from src.tools.web_search import get_web_search_tool
from src.tools.wikipedia import get_wikipedia_tool
from src.tools.page_fetcher import get_page_fetcher_tool
from src.tools.serper import get_serper_tool, is_available as serper_available, search_serper
from src.tools.brave import get_brave_tool, is_available as brave_available, search_brave
# SearXNG disabled — multi-engine fan-out returns too much topically
# irrelevant noise (DID pages for "did Pelosi", travel guides for "China").
# Container stays running; re-enable after adding relevance filtering.
from src.tools.legiscan import search_legislation, is_available as legiscan_available
from src.schemas.interested_parties import InterestedPartiesDict
from src.utils.evidence_ranker import score_url, tier_label
from src.utils.logging import log, get_logger

MODULE = "research"
logger = get_logger()


def _build_progress_note(messages: list) -> str | None:
    """Analyze conversation history and build a progress summary.

    Scans the agent's message history to understand:
    - How many tool calls have been made (and which tools)
    - What search queries were used (to avoid repeats)
    - How many unique URLs/domains are in the evidence
    - How many page fetches vs search-only results
    - Whether evidence appears one-sided

    Returns a concise progress note for injection before the next LLM call,
    or None if there's nothing useful to report yet (first call).
    """
    tool_calls = 0  # Only real agent calls (not synthetic seed/prefetch)
    search_queries: list[str] = []
    fetch_urls: list[str] = []  # Agent-initiated fetches
    prefetch_urls: list[str] = []  # Programmatic pre-fetches
    evidence_urls: set[str] = set()
    evidence_domains: set[str] = set()
    tools_used: set[str] = set()

    for msg in messages:
        # Count tool calls from AIMessages
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                is_synthetic = tc_id.startswith(("seed_", "prefetch_"))
                tool_name = tc.get("name", "")
                args = tc.get("args", {})

                if not is_synthetic:
                    tool_calls += 1
                    tools_used.add(tool_name)

                # Track search queries (from real agent calls only)
                if not is_synthetic and "search" in tool_name.lower():
                    query = args.get("query") or args.get("input") or ""
                    if query:
                        search_queries.append(query)

                # Track page fetch URLs (separate prefetch vs agent)
                if "fetch" in tool_name.lower():
                    url = args.get("url", "")
                    if url:
                        if is_synthetic:
                            prefetch_urls.append(url)
                        else:
                            fetch_urls.append(url)

        # Extract URLs from tool results
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            for url_match in _re.finditer(r"URL:\s*(https?://\S+)", content):
                url = url_match.group(1).strip()
                evidence_urls.add(url)
                try:
                    domain = urlparse(url).netloc.lower().replace("www.", "")
                    if domain:
                        evidence_domains.add(domain)
                except Exception:
                    pass

    # Don't inject progress on the first call (no real agent tool calls yet)
    if tool_calls == 0:
        return None

    # Count seed tier + conflict coverage from seed messages
    seed_urls = 0
    tier1_count = 0
    tier2_count = 0
    conflict_count = 0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            if tool_call_id.startswith("seed_"):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                for line in content.split("\n"):
                    if line.startswith("URL: "):
                        seed_urls += 1
                    elif line.startswith("Source tier: TIER 1"):
                        tier1_count += 1
                    elif line.startswith("Source tier: TIER 2"):
                        tier2_count += 1
                    elif line.startswith("Conflict:"):
                        conflict_count += 1

    lines = [f"[RESEARCH PROGRESS — {tool_calls} tool calls completed]"]

    # Seed coverage
    if seed_urls > 0:
        tier_parts = []
        if tier1_count:
            tier_parts.append(f"{tier1_count} TIER 1")
        if tier2_count:
            tier_parts.append(f"{tier2_count} TIER 2")
        if conflict_count:
            tier_parts.append(f"{conflict_count} conflicted")
        tier_str = f", {' + '.join(tier_parts)} sources" if tier_parts else ""
        lines.append(f"Seed coverage: {seed_urls} URLs pre-gathered{tier_str}")

    # Pre-fetch coverage
    if prefetch_urls:
        lines.append(f"Pre-fetched: {len(prefetch_urls)} full articles from top seeds")

    # Evidence inventory
    lines.append(
        f"Evidence: {len(evidence_urls)} unique URLs across "
        f"{len(evidence_domains)} domains"
    )

    # Page fetches (combined view)
    total_fetches = len(prefetch_urls) + len(fetch_urls)
    if prefetch_urls and fetch_urls:
        lines.append(
            f"Full articles read: {total_fetches} "
            f"({len(prefetch_urls)} pre-fetched + {len(fetch_urls)} agent-fetched)"
        )
    elif prefetch_urls:
        lines.append(f"Full articles read: {total_fetches} (all pre-fetched)")
    else:
        lines.append(f"Full articles read: {total_fetches}")

    # Queries used (so agent can avoid repeats)
    if search_queries:
        # Show last 6 queries to keep it concise
        recent = search_queries[-6:]
        q_list = ", ".join(f'"{q}"' for q in recent)
        if len(search_queries) > 6:
            q_list = f"... {q_list}"
        lines.append(f"Queries used: {q_list}")

    # Search engines used
    engine_names = []
    for t in tools_used:
        tl = t.lower()
        if "serper" in tl:
            engine_names.append("Serper")
        elif "searxng" in tl:
            engine_names.append("SearXNG")  # legacy: may appear from older runs
        elif "brave" in tl:
            engine_names.append("Brave")
        elif "web_search" in tl or "duckduckgo" in tl:
            engine_names.append("DuckDuckGo")
        elif "wikipedia" in tl:
            engine_names.append("Wikipedia")
    if engine_names:
        lines.append(f"Engines used: {', '.join(sorted(set(engine_names)))}")

    # Strategic suggestions based on current state
    suggestions = []
    # Only suggest engines that are actually configured
    available_engines = {"DuckDuckGo", "Wikipedia"}
    if serper_available():
        available_engines.add("Serper")
    # SearXNG disabled — too much topically irrelevant noise
    if brave_available():
        available_engines.add("Brave")
    unused_engines = available_engines - set(engine_names)
    if unused_engines and tool_calls <= 6:
        suggestions.append(
            f"try {', '.join(sorted(unused_engines))} for source diversity"
        )
    if len(fetch_urls) == 0 and len(prefetch_urls) == 0 and len(evidence_urls) >= 4:
        suggestions.append(
            "fetch full articles from your best URLs — snippets may miss detail"
        )
    if total_fetches >= 3 and tool_calls >= 8:
        suggestions.append(
            "you have good depth — consider wrapping up if both sides are covered"
        )

    if suggestions:
        lines.append(f"Suggestions: {'; '.join(suggestions)}")

    return "\n".join(lines)


def _research_pre_model_hook(state: dict) -> dict:
    """Pre-model hook: inject progress awareness before each LLM call.

    This runs before every LLM invocation in the ReAct loop. It analyzes
    what the agent has gathered so far and injects a progress note into
    the system message. The note is ephemeral — returned via
    llm_input_messages, so it doesn't modify the actual conversation state
    and won't accumulate across iterations.

    This gives the agent awareness of:
    - What it's already searched for (don't repeat queries)
    - How many unique sources it has (is diversity sufficient?)
    - Whether it's read full articles or only has snippets
    - Which search engines it hasn't tried yet
    """
    messages = list(state.get("messages", []))
    progress = _build_progress_note(messages)

    if progress:
        # Inject into the system message — llama.cpp/Jinja requires system
        # messages to be at the beginning, so we can't append a new one.
        # Instead, append the progress note to the existing system message.
        updated = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                updated.append(
                    SystemMessage(content=msg.content + "\n\n" + progress)
                )
            else:
                updated.append(msg)
        messages = updated

    return {"llm_input_messages": messages}


def _build_tool_list() -> list:
    """Build the agent's tool list based on which API keys are configured.

    Search tools (priority order):
      1. Serper (primary — Google results via Serper API, needs SERPER_API_KEY)
      2. DuckDuckGo (fallback — always available, free)
      3. SearXNG (optional padding — self-hosted meta-search)
      4. Brave (optional — independent index, needs BRAVE_API_KEY)

    Always included:
      - Wikipedia (free, reliable for established facts)
      - Page fetcher (reads URLs, no key needed)
    """
    tools = []

    # Serper — primary search (Google results)
    if serper_available():
        tools.append(get_serper_tool())
        log.info(logger, MODULE, "tool_enabled", "Serper search enabled (primary)")

    # DuckDuckGo — fallback, always available
    tools.append(get_web_search_tool())

    # SearXNG — disabled from agent tools (multi-engine fan-out returns
    # too much topically irrelevant noise that pollutes seed ranking).
    # Container stays running for potential future use with relevance filtering.

    # Brave — optional, independent index
    if brave_available():
        tools.append(get_brave_tool())
        log.info(logger, MODULE, "tool_enabled", "Brave search enabled")

    # Wikipedia — always available
    tools.append(get_wikipedia_tool())

    # Page fetcher — always available, lets agent read full articles
    tools.append(get_page_fetcher_tool())

    # NOTE: Wikidata, MBFC, and LegiScan are NOT agent tools. They run
    # programmatically:
    # - Wikidata: runs at decompose time, results passed as prompt context
    # - MBFC: runs in search tools (pre-filters) and judge (annotations)
    # - LegiScan: runs after agent finishes, appends legislative evidence
    # This frees up the agent's tool budget for actual evidence gathering.

    tool_names = [t.name for t in tools]
    log.info(logger, MODULE, "tools_loaded", "Research agent tools configured",
             tool_count=len(tools), tools=tool_names)

    return tools


def build_research_agent(interested_parties_context: str = "",
                         claim_date: str | None = None):
    """Build a ReAct agent for evidence gathering.

    Tools are dynamically loaded based on configured API keys.
    See _build_tool_list() for the full list and priority order.

    Args:
        interested_parties_context: Pre-formatted text describing the
            interested parties and their connections (from Wikidata
            expansion at decompose time). Injected into the system
            prompt so the agent knows who the players are without
            burning tool calls on lookups.
    """
    # thinking=off — the ReAct loop is pure tool-routing: pick a search
    # query, call the tool, repeat.  Thinking mode wastes ~25-45s per
    # iteration generating <think> blocks nobody reads, which eats the
    # entire timeout budget.  With thinking off, the same search queries
    # are produced in ~3s per iteration.
    #
    # temperature=0 for deterministic search queries.  The agent's job is
    # routing — "search for X", "fetch Y" — not creative writing.  temp=0
    # ensures the same sub-claim always generates the same queries, which
    # means the same evidence, which means consistent verdicts.
    llm = get_llm(temperature=0)
    tools = _build_tool_list()

    prompt = RESEARCH_SYSTEM.format(
        current_date=date.today().isoformat(),
        claim_date_line=build_claim_date_line(claim_date),
    )

    # Inject interested parties context into the prompt
    if interested_parties_context:
        prompt += "\n\n" + interested_parties_context

    return create_react_agent(
        llm,
        tools,
        prompt=prompt,
        pre_model_hook=_research_pre_model_hook,
    )


def _parse_tool_output(content: str, tool_name: str) -> list[dict]:
    """Parse structured evidence items from a tool's text output.

    All our search tools format results as blocks separated by '---':
        Title: ...
        URL: ...
        Snippet/Summary: ...

    The page fetcher uses:
        Page: ...
        URL: ...
        <content>

    This function splits multi-result outputs into individual evidence
    items with extracted metadata (url, title, source_type).
    """
    if not content or not content.strip():
        return []

    # Skip known empty responses
    stripped = content.strip()
    if stripped in (
        "No results found.",
        "No Wikipedia results found.",
        "No SearXNG results found. Try web_search as a fallback.",
        "SearXNG search failed. Try web_search as a fallback.",
        "Wikipedia search failed. Try web search instead.",
    ):
        return []

    # Determine source type from tool name
    tl = tool_name.lower()
    if "wikipedia" in tl:
        source_type = "wikipedia"
    else:
        source_type = "web"

    # Page fetcher — single result, different format
    if "fetch_page" in tl:
        title = ""
        url = None
        body = stripped
        # Format: "Page: ...\nURL: ...\n\n<content>"
        if stripped.startswith("Page:"):
            title = stripped[len("Page:"):].strip()
        url_m = _re.search(r"URL:\s*(https?://\S+)", stripped)
        if url_m:
            url = url_m.group(1).strip()
            # Content starts after the URL line + blank line
            after_url = stripped[url_m.end():].lstrip("\n")
            if after_url:
                body = after_url
        return [{
            "source_type": source_type,
            "source_url": url,
            "title": title,
            "content": body[:5000],
        }]

    # Blocked/error responses from page fetcher
    if stripped.startswith(("Blocked source:", "Failed to fetch", "Invalid URL")):
        return []

    # Multi-result tools (Serper, SearXNG, Brave, Wikipedia, DDG)
    # Split on '---' separator used by all our tool wrappers
    blocks = stripped.split("\n\n---\n\n")

    items = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title = ""
        url = None
        snippet = block  # fallback: use the whole block as content

        # Parse structured fields
        first_line = block.split("\n", 1)[0]
        for prefix in ("Title: ", "Page: "):
            if first_line.startswith(prefix):
                title = first_line[len(prefix):].strip()
                break

        url_m = _re.search(r"URL:\s*(https?://\S+)", block)
        if url_m:
            url = url_m.group(1).strip()

        # Extract the content/snippet portion
        snippet_m = _re.search(
            r"(?:Snippet|Summary|Content):\s*(.+)",
            block, _re.DOTALL,
        )
        if snippet_m:
            snippet = snippet_m.group(1).strip()

        items.append({
            "source_type": source_type,
            "source_url": url,
            "title": title,
            "content": snippet[:5000],
        })

    return items


def extract_evidence(messages: list) -> list[dict]:
    """Extract structured evidence records from the agent's conversation.

    After the ReAct agent runs, its message history contains:
      1. SystemMessage — research instructions (from RESEARCH_SYSTEM)
      2. HumanMessage — "Find evidence about: {claim}"
      3. AIMessage with tool_calls — agent decides to search
      4. ToolMessage — search results (structured text with Title/URL/Snippet)
      5. ... (more tool calls and results)
      N. AIMessage — agent's final summary (no tool_calls)

    We parse each ToolMessage to extract individual evidence items with
    metadata (URL, title, source_type). Multi-result tool outputs (e.g.,
    SearXNG returning 8 results) are split into separate evidence items.

    The agent's final AIMessage is NOT included — it's the agent's own
    interpretation, not primary evidence. The judge should reason only
    from actual source material.

    Deduplicates by URL — different search engines often return the same
    pages, and duplicates just bloat the judge prompt without adding signal.
    """
    evidence = []
    seen_urls: set[str] = set()
    total_items = 0
    deduped_count = 0

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "") or ""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        items = _parse_tool_output(content, tool_name)
        for item in items:
            total_items += 1
            url = item.get("source_url")
            if url:
                if url in seen_urls:
                    deduped_count += 1
                    continue
                seen_urls.add(url)
            evidence.append(item)

    log.info(logger, MODULE, "evidence_dedup",
             "Evidence extraction complete",
             total_items=total_items, unique_items=len(evidence),
             deduped=deduped_count, unique_urls=len(seen_urls))
    return evidence


async def _run_seed_searches(
    sub_claim: str,
    categories: list[str] | None = None,
    seed_queries: list[str] | None = None,
) -> list[dict]:
    """Run seed searches concurrently using LLM-written queries.

    The decompose LLM writes targeted search queries per fact. This
    function wraps them in backend routing (SearXNG category based on
    the fact's categories) and fires them alongside mechanical base
    queries (raw sub-claim, Wikipedia).

    Unavailable backends fall back to SearXNG (self-hosted, always available).

    Returns evidence dicts matching _parse_tool_output schema:
        {source_type, source_url, title, content}
    """
    import asyncio

    from src.agent.claim_category import generate_seed_queries
    from src.tools.web_search import search_duckduckgo
    from src.tools.wikipedia import search_wikipedia

    if not categories:
        categories = ["GENERAL"]
    query_specs = generate_seed_queries(sub_claim, categories, seed_queries)

    log.info(logger, MODULE, "seed_plan", "Seed search plan",
             sub_claim=sub_claim, categories=categories,
             query_count=len(query_specs),
             queries=[{"label": s["label"], "query": s["query"], "max_results": s["max_results"]} for s in query_specs])

    sem = asyncio.Semaphore(8)
    tasks = []

    for spec in query_specs:
        for backend in spec["backends"]:
            # Resolve backend with availability fallback
            resolved_backend = backend
            if backend == "serper" and not serper_available():
                resolved_backend = "duckduckgo"  # free fallback
            elif backend == "brave" and not brave_available():
                resolved_backend = None
            elif backend == "searxng":
                resolved_backend = None  # SearXNG disabled from seeds
            # duckduckgo is always available — no fallback needed

            if resolved_backend and resolved_backend != backend:
                log.info(logger, MODULE, "backend_fallback",
                         "Backend unavailable, falling back",
                         original=backend, resolved=resolved_backend)

            if not resolved_backend:
                continue

            async def _do_search(
                q=spec["query"],
                mx=spec["max_results"],
                cat=spec["searxng_category"],
                be=resolved_backend,
                lbl=spec["label"],
            ):
                async with sem:
                    try:
                        if be == "serper":
                            return (lbl, q, "web",
                                    await search_serper(q, max_results=mx))
                        elif be == "brave":
                            return (lbl, q, "web",
                                    await search_brave(q, max_results=mx))
                        elif be == "duckduckgo":
                            return (lbl, q, "web",
                                    await search_duckduckgo(q, max_results=mx))
                        elif be == "wikipedia":
                            return (lbl, q, "wikipedia",
                                    await search_wikipedia(q, max_results=mx))
                    except Exception as exc:
                        log.warning(logger, MODULE, "seed_task_failed",
                                    "Seed search task failed",
                                    backend=be, label=lbl, query=q,
                                    error=str(exc))
                        return (lbl, q, be, [])

            tasks.append(asyncio.wait_for(_do_search(), timeout=10))

    # Run all concurrently
    _t0 = _time.monotonic()
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_ms = int((_time.monotonic() - _t0) * 1000)

    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    evidence: list[dict] = []

    for result in raw_results:
        if isinstance(result, Exception):
            log.warning(logger, MODULE, "seed_search_failed",
                        "A seed search failed",
                        error=str(result), error_type=type(result).__name__)
            continue

        label, query, source_type_label, items = result

        for item in items:
            url = item.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            if source_type_label == "wikipedia":
                evidence.append({
                    "source_type": "wikipedia",
                    "source_url": url,
                    "title": item.get("title", ""),
                    "content": item.get("summary", ""),
                    "_seed_query": query,
                })
            else:
                evidence.append({
                    "source_type": "web",
                    "source_url": url,
                    "title": item.get("title", ""),
                    "content": item.get("snippet", ""),
                    "_seed_query": query,
                })

    log.info(logger, MODULE, "seed_done", "Seed searches complete",
             sub_claim=sub_claim, categories=categories,
             total_results=len(evidence),
             unique_urls=len(seen_urls), latency_ms=elapsed_ms)
    return evidence


def _format_seed_results(results: list[dict]) -> str:
    """Format seed results into the same text format search tools return.

    Uses Title/URL/Snippet blocks separated by '---' — the same format
    that _parse_tool_output and extract_evidence already understand.
    """
    if not results:
        return "No results found."

    parts = []
    for r in results:
        title = r.get("title", "")
        url = r.get("source_url", "")
        snippet = r.get("content", "")

        # Use pre-computed tier if available (from _rank_and_filter_seeds),
        # else compute on the fly
        tier = r.get("_tier") or tier_label(url)
        tier_line = f"Source tier: {tier}\n" if tier else ""

        # Conflict annotation
        conflict_flags = r.get("_conflict_flags", [])
        conflict_line = f"Conflict: {'; '.join(conflict_flags)}\n" if conflict_flags else ""

        parts.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"{tier_line}"
            f"{conflict_line}"
            f"Snippet: {snippet}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pre-fetch: read full page content from top-ranked seeds before the agent
# ---------------------------------------------------------------------------

PREFETCH_MAX = 10
PREFETCH_CONTENT_MAX = 5000  # chars per page (lower than page_fetcher's 8000)
PREFETCH_CONCURRENCY = 5


async def _prefetch_seed_pages(
    ranked_seeds: list[dict],
    max_pages: int = PREFETCH_MAX,
) -> tuple[list, list[str]]:
    """Pre-fetch full page content for top-ranked seeds.

    Selects seeds with TIER 1/TIER 2 labels first, then fills from
    remaining top-ranked seeds. Fetches in parallel using fetch_page()
    directly (not the LangChain tool wrapper).

    Returns (prefetch_messages, prefetched_urls):
      - prefetch_messages: synthetic AIMessage + ToolMessage pairs
        (same format as agent's fetch_page_content tool output)
      - prefetched_urls: URLs that were successfully fetched
    """
    import asyncio
    from src.tools.page_fetcher import fetch_page
    from src.tools.source_filter import is_blocked

    _t0 = _time.monotonic()

    # --- Select candidates: tiered first, then remaining top-ranked ---
    # Skip conflicted sources — don't waste pre-fetch slots on pages that
    # just quote interested parties. Independent sources get priority.
    seen_urls: set[str] = set()
    tiered: list[dict] = []
    untiered: list[dict] = []

    for seed in ranked_seeds:
        url = seed.get("source_url", "")
        if not url or url in seen_urls:
            continue
        if is_blocked(url):
            continue
        if seed.get("_conflict_flags"):
            continue
        seen_urls.add(url)

        tier = seed.get("_tier", "")
        if tier:  # TIER 1 or TIER 2
            tiered.append(seed)
        else:
            untiered.append(seed)

    candidates = (tiered + untiered)[:max_pages]

    if not candidates:
        return [], []

    # --- Parallel fetch ---
    sem = asyncio.Semaphore(PREFETCH_CONCURRENCY)

    async def _fetch_one(seed: dict) -> dict | None:
        url = seed.get("source_url", "")
        async with sem:
            try:
                result = await fetch_page(url)
            except Exception:
                return None
        if result.get("error") or not result.get("content"):
            return None
        # Truncate to prefetch limit (tighter than page_fetcher's 8000)
        content = result["content"]
        if len(content) > PREFETCH_CONTENT_MAX:
            content = content[:PREFETCH_CONTENT_MAX] + "\n\n[... content truncated ...]"
        result["content"] = content
        return result

    fetch_results = await asyncio.gather(
        *(_fetch_one(s) for s in candidates),
        return_exceptions=True,
    )

    # --- Build synthetic messages ---
    prefetch_messages: list = []
    prefetched_urls: list[str] = []

    for result in fetch_results:
        if isinstance(result, Exception) or result is None:
            continue

        url = result["url"]
        title = result.get("title", "")
        content = result["content"]

        # SpaCy NER on content (same as page fetcher tool does)
        entities_line = ""
        try:
            from src.utils.ner import extract_entities
            entities = extract_entities(content, labels={"PERSON", "ORG"})
            if entities:
                names = list(dict.fromkeys(e["text"] for e in entities))[:15]
                entities_line = f"Entities mentioned: {', '.join(names)}\n"
        except Exception:
            pass  # NER failure is non-fatal

        tool_call_id = f"prefetch_{_uuid.uuid4().hex[:12]}"

        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "fetch_page_content",
                "args": {"url": url},
                "id": tool_call_id,
                "type": "tool_call",
            }],
        )

        header = f"Page: {title}\nURL: {url}\n"
        if entities_line:
            header += entities_line
        header += "\n"
        tool_msg = ToolMessage(
            content=header + content,
            tool_call_id=tool_call_id,
            name="fetch_page_content",
        )

        prefetch_messages.append(ai_msg)
        prefetch_messages.append(tool_msg)
        prefetched_urls.append(url)

    elapsed_ms = int((_time.monotonic() - _t0) * 1000)
    candidate_urls = [s.get("source_url", "") for s in candidates]
    failed_urls = [u for u in candidate_urls if u not in prefetched_urls]
    log.info(logger, MODULE, "prefetch_complete",
             "Pre-fetched top seed pages",
             attempted=len(candidates),
             succeeded=len(prefetched_urls),
             failed=len(failed_urls),
             latency_ms=elapsed_ms,
             urls=prefetched_urls,
             failed_urls=failed_urls)

    return prefetch_messages, prefetched_urls


def _build_seed_messages(seed_results: list[dict]) -> list:
    """Build synthetic AIMessage + ToolMessage pairs from seed results.

    Groups results by source query, then creates pairs that look like the
    agent already made those searches. This way _build_progress_note and
    extract_evidence parse them without special handling.
    """
    if not seed_results:
        return []

    # Group by query
    groups: dict[str, list[dict]] = {}
    for r in seed_results:
        query = r.get("_seed_query", "seed")
        groups.setdefault(query, []).append(r)

    messages = []
    for query, results in groups.items():
        # Determine tool name from source type
        is_wiki = all(r.get("source_type") == "wikipedia" for r in results)
        tool_name = "wikipedia_search" if is_wiki else "web_search"

        # Unique tool call ID
        tool_call_id = f"seed_{_uuid.uuid4().hex[:12]}"

        # AIMessage with tool_calls (looks like the agent decided to search)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": tool_name,
                "args": {"query": query},
                "id": tool_call_id,
                "type": "tool_call",
            }],
        )

        # ToolMessage with formatted results
        formatted = _format_seed_results(results)
        tool_msg = ToolMessage(
            content=formatted,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

        messages.append(ai_msg)
        messages.append(tool_msg)

    return messages


# Conflict penalty applied to seeds from sources affiliated with interested parties.
# -15 pushes a mostly-factual conflicted source (score ~22) below most non-conflicted
# sources, but a gov-affiliated source (score ~37) still ranks respectably at ~22.
CONFLICT_PENALTY = -15


def _collect_domains(seed_results: list[dict]) -> set[str]:
    """Extract unique domains from seed results for MBFC cache warming."""
    from src.tools.source_ratings import extract_domain

    domains = set()
    for r in seed_results:
        url = r.get("source_url", "")
        if url:
            domains.add(extract_domain(url))
    domains.discard("")
    return domains


async def _enrich_parties_from_mbfc(
    seed_results: list[dict],
    all_parties: list[str],
    affiliated_media: list[str],
) -> tuple[list[str], list[str]]:
    """Check MBFC ownership for connections to existing interested parties.

    After await_ratings_parallel(), every seed domain has a cached MBFC rating
    with an 'ownership' field (e.g., "Owned by Rupert Murdoch's News Corporation").
    This function extracts owner names via SpaCy NER, then Wikidata-expands them
    to check if they connect to EXISTING interested parties.

    Overlap-gated: only adds to parties/media when an MBFC owner's Wikidata
    graph intersects with the claim's known interested parties. This prevents
    unrelated corporate trees (e.g., Thomson Reuters subsidiaries) from
    polluting the parties list on claims that have nothing to do with them.

    When overlap IS found, only the owner + their MEDIA HOLDINGS are added —
    not all subsidiaries, board members, or unrelated orgs. The purpose is
    conflict detection on news sources, not corporate graph exploration.

    Runs BEFORE seed scoring so discovered media influence conflict detection.

    Returns new (all_parties, affiliated_media) lists — does NOT mutate inputs.
    """
    import asyncio
    from src.tools.source_ratings import get_source_rating_sync, extract_domain
    from src.tools.media_matching import extract_owners_from_mbfc
    from src.tools.wikidata import get_ownership_chain, collect_all_connected_parties

    enriched_parties = list(all_parties)
    enriched_media = list(affiliated_media)
    all_parties_lower = {p.lower() for p in enriched_parties}

    # Collect unique owners from all seed domains' MBFC ratings
    seen_domains = set()
    new_owners = []
    for r in seed_results:
        url = r.get("source_url", "")
        if not url:
            continue
        domain = extract_domain(url)
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        rating = get_source_rating_sync(url)
        if not rating or not rating.get("ownership"):
            continue

        owners = extract_owners_from_mbfc(rating["ownership"])
        for owner in owners:
            if owner.lower() not in all_parties_lower:
                new_owners.append(owner)
                all_parties_lower.add(owner.lower())

    if not new_owners:
        return enriched_parties, enriched_media

    log.info(logger, MODULE, "mbfc_owner_discovery",
             "MBFC ownership entities discovered for Wikidata expansion",
             new_owners=new_owners[:10])

    # Wikidata-expand new owners in parallel, capped at 6
    # Only add if owner's graph overlaps with existing interested parties
    existing_parties_lower = {p.lower() for p in enriched_parties}

    async def _expand_owner(owner: str) -> None:
        try:
            result = await get_ownership_chain(owner)
            if result.get("error"):
                return

            connected = collect_all_connected_parties(result, skip_family_expanded=True)

            # Check if this owner connects to any existing interested party
            all_connected_lower = {n.lower() for n in (
                connected["people"] + connected["orgs"] + connected["media"]
                + [owner]
            )}
            overlap = all_connected_lower & existing_parties_lower

            if not overlap:
                log.debug(logger, MODULE, "mbfc_owner_no_overlap",
                          "MBFC owner has no connection to interested parties, skipping",
                          owner=owner)
                return

            # Overlap found — this owner is connected to the claim's parties.
            # Add the owner to all_parties for conflict detection.
            if owner not in enriched_parties:
                enriched_parties.append(owner)

            # Add media holdings only — these are the outlets we need to
            # flag as conflicted. Not subsidiaries, not board members.
            for media in connected["media"]:
                if media not in enriched_media:
                    enriched_media.append(media)

            log.info(logger, MODULE, "mbfc_owner_connected",
                     "MBFC owner connects to interested party",
                     owner=owner,
                     overlap=list(overlap)[:5],
                     media_added=len(connected["media"]))
        except Exception as e:
            log.warning(logger, MODULE, "mbfc_owner_expand_failed",
                        "MBFC owner Wikidata expansion failed",
                        owner=owner, error=str(e))

    await asyncio.gather(*[_expand_owner(o) for o in new_owners[:6]])

    log.info(logger, MODULE, "mbfc_enrichment_done",
             "MBFC→Wikidata enrichment complete",
             parties_before=len(all_parties),
             parties_after=len(enriched_parties),
             media_before=len(affiliated_media),
             media_after=len(enriched_media))

    return enriched_parties, enriched_media


async def _enrich_parties_from_evidence_content(
    evidence: list[dict],
    all_parties: list[str],
    affiliated_media: list[str],
) -> tuple[list[str], list[str]]:
    """Discover new interested parties from evidence article content via NER + Wikidata.

    Runs AFTER extract_evidence() + LegiScan. SpaCy NER extracts people/orgs
    mentioned in evidence articles, then Wikidata checks if they connect to
    existing interested parties.

    Unlike MBFC enrichment (authoritative, add unconditionally), evidence mentions
    are only added if their Wikidata graph OVERLAPS with existing parties.

    Returns new (all_parties, affiliated_media) lists — does NOT mutate inputs.
    """
    import asyncio
    from src.tools.wikidata import get_ownership_chain, collect_all_connected_parties
    from src.utils.ner import extract_quoted_entities

    enriched_parties = list(all_parties)
    enriched_media = list(affiliated_media)

    # NER on concatenated evidence content
    all_content = " ".join(ev.get("content", "") for ev in evidence)
    discovered_entities = extract_quoted_entities(all_content)

    all_parties_lower = {p.lower() for p in enriched_parties}
    new_entities = [
        e for e in discovered_entities
        if e.lower() not in all_parties_lower
    ]

    if not new_entities:
        return enriched_parties, enriched_media

    log.debug(logger, MODULE, "evidence_ner_entities",
              "SpaCy NER extracted new entities from evidence content",
              new_entities=new_entities[:15])

    # Wikidata-expand in parallel, capped at 8.
    # Only add if graph overlaps with existing parties.
    async def _expand_entity(entity: str) -> None:
        try:
            result = await get_ownership_chain(entity)
            if result.get("error"):
                return

            connected = collect_all_connected_parties(result)
            all_connected = (
                set(connected["people"])
                | set(connected["orgs"])
                | set(connected["media"])
            )
            overlap = all_connected & set(enriched_parties)

            if overlap:
                if entity not in enriched_parties:
                    enriched_parties.append(entity)
                for person in connected["people"]:
                    if person not in enriched_parties:
                        enriched_parties.append(person)
                for media in connected["media"]:
                    if media not in enriched_media:
                        enriched_media.append(media)

                log.info(logger, MODULE, "evidence_ner_enrichment_found",
                         "Evidence entity connects to interested party",
                         entity=entity, overlap=list(overlap)[:5])
        except Exception as e:
            log.warning(logger, MODULE, "evidence_ner_expand_failed",
                        "Evidence entity Wikidata expansion failed",
                        entity=entity, error=str(e))

    await asyncio.gather(*[_expand_entity(e) for e in new_entities[:8]])

    # Hard cap to prevent party explosion
    MAX_ALL_PARTIES = 40
    if len(enriched_parties) > MAX_ALL_PARTIES:
        log.warning(logger, MODULE, "parties_capped",
                    "Capping all_parties to prevent explosion",
                    before=len(enriched_parties), after=MAX_ALL_PARTIES)
        enriched_parties = enriched_parties[:MAX_ALL_PARTIES]

    log.info(logger, MODULE, "evidence_ner_enrichment_done",
             "Evidence NER→Wikidata enrichment complete",
             parties_before=len(all_parties),
             parties_after=len(enriched_parties),
             media_before=len(affiliated_media),
             media_after=len(enriched_media))

    return enriched_parties, enriched_media


async def _rank_and_filter_seeds(
    seed_results: list[dict],
    affiliated_media: list[str],
    all_parties: list[str],
    max_seeds: int = 30,
) -> list[dict]:
    """Rank seeds by quality + conflict detection, keep top N.

    IMPORTANT: Caller must ensure MBFC cache is warm before calling this
    (via _collect_domains + await_ratings_parallel). This function no longer
    awaits MBFC itself — the caller does it so that MBFC→Wikidata enrichment
    can run between the await and ranking.

    1. Score each seed with score_url() (assumes MBFC data is cached)
    2. Detect conflicts: affiliated media URL match + publisher ownership
    3. Annotate each seed with _tier and _conflict_flags metadata
    4. Apply CONFLICT_PENALTY to conflicted sources
    5. Sort by adjusted score DESC, keep top max_seeds
    """
    if not seed_results:
        return []

    from src.tools.media_matching import url_matches_media, check_publisher_ownership

    # Score and detect conflicts
    scored: list[tuple[int, dict]] = []
    for r in seed_results:
        url = r.get("source_url", "")
        quality_score, _ = score_url(url)
        tier = tier_label(url)

        # Conflict detection
        conflict_flags = []
        if url and affiliated_media:
            url_lower = url.lower()
            for media in affiliated_media:
                if url_matches_media(url_lower, media):
                    conflict_flags.append(f"affiliated: {media}")
                    break

        if url and all_parties and not conflict_flags:
            owner_match = check_publisher_ownership(url, all_parties)
            if owner_match:
                conflict_flags.append(f"owned by: {owner_match}")
                log.info(logger, MODULE, "publisher_ownership_conflict",
                         "Publisher ownership match detected",
                         url=url, owner=owner_match)

        # Apply conflict penalty
        adjusted_score = quality_score
        if conflict_flags:
            adjusted_score += CONFLICT_PENALTY

        # Annotate seed with metadata for _format_seed_results
        r["_tier"] = tier
        r["_conflict_flags"] = conflict_flags

        scored.append((adjusted_score, r))

    # Step 5: Sort by adjusted score DESC (stable sort preserves discovery order for ties)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Step 6: Keep top N
    ranked = [r for _, r in scored[:max_seeds]]

    conflict_count = sum(1 for _, r in scored[:max_seeds] if r.get("_conflict_flags"))
    top_5 = [
        {"url": r.get("source_url", ""), "score": s, "tier": r.get("_tier", ""),
         "conflict": r.get("_conflict_flags", [])}
        for s, r in scored[:5]
    ]
    dropped = [
        {"url": r.get("source_url", ""), "score": s}
        for s, r in scored[max_seeds:]
    ] if len(scored) > max_seeds else []
    log.info(logger, MODULE, "seed_ranked",
             "Seeds ranked and filtered",
             total_seeds=len(seed_results),
             kept=len(ranked),
             conflicted=conflict_count,
             top_5=top_5,
             dropped_count=len(dropped),
             dropped_sample=dropped[:5])

    return ranked


async def research_claim(
    sub_claim: str,
    interested_parties: InterestedPartiesDict | None = None,
    max_steps: int = 38,
    timeout_secs: int = 300,
    categories: list[str] | None = None,
    seed_queries: list[str] | None = None,
    speaker: str | None = None,
    claim_date: str | None = None,
) -> tuple[list[dict], InterestedPartiesDict]:
    """Run the research agent to gather evidence for a sub-claim.

    This is the main entry point called by the research_subclaim activity.

    Uses streaming (astream) instead of ainvoke so that when the agent
    hits its step limit or times out, we keep ALL evidence gathered up
    to that point instead of falling back to a limited direct search.

    Args:
        sub_claim: The specific sub-claim to find evidence for.
        interested_parties: Structured dict from decompose with
            all_parties, affiliated_media, wikidata_context, etc.
            Used for programmatic conflict detection at seed ranking
            time, and wikidata_context is injected into the agent
            prompt.
        max_steps: Maximum number of graph steps (the agent's budget).
                   Each tool call costs ~3 steps (pre_model + agent + tools).
                   38 steps allows ~12 tool calls. The final stop costs 2
                   steps (pre_model + agent deciding no more tools).
        timeout_secs: Soft timeout in seconds. If the agent exceeds this,
                      we return whatever evidence was gathered so far.
                      Default 300s (5 min).
        categories: Evidence-need categories from decompose (e.g. ["QUANTITATIVE",
                    "LEGISLATIVE"]). Determines SearXNG category routing for
                    seed queries. None defaults to ["GENERAL"].
        seed_queries: LLM-written search queries from decompose. These are
                      human-quality queries tailored to the specific evidence
                      this fact needs. None if decompose didn't produce them.

    Returns:
        Tuple of (evidence, enriched_interested_parties):
          - evidence: List of evidence dicts (deduplicated by URL)
          - enriched_interested_parties: Updated InterestedPartiesDict with
            new parties/media discovered from MBFC ownership + evidence NER
    """
    import asyncio

    # Extract structured data from interested_parties dict
    if interested_parties is None:
        interested_parties = {}
    wikidata_context = interested_parties.get("wikidata_context", "")
    affiliated_media = interested_parties.get("affiliated_media", [])
    all_parties = interested_parties.get("all_parties", [])

    # Build the enriched parties dict (will be updated through the pipeline)
    enriched_parties: InterestedPartiesDict = dict(interested_parties)

    log.info(logger, MODULE, "start", "Starting research agent",
             sub_claim=sub_claim)

    # Phase 1a: Programmatic seed searches (no LLM, ~3-8 seconds)
    # Gathers a broad, deterministic evidence pool before the agent starts.
    # The agent sees these as prior searches and spends its budget on
    # deep-dive fetching and targeted follow-up instead of initial discovery.
    try:
        seed_results = await _run_seed_searches(
            sub_claim, categories=categories,
            seed_queries=seed_queries,
        )
    except Exception as e:
        log.warning(logger, MODULE, "seed_failed",
                    "Seed searches failed, agent will search from scratch",
                    error=str(e), error_type=type(e).__name__)
        seed_results = []

    # Phase 1b-i: Collect domains and await MBFC ratings in parallel.
    # Separated from _rank_and_filter_seeds so we can run MBFC→Wikidata
    # enrichment between the MBFC await and ranking.
    from src.tools.source_ratings import await_ratings_parallel

    domains = _collect_domains(seed_results)
    if domains:
        await await_ratings_parallel(list(domains), max_concurrent=8)

    # Phase 1b-ii: MBFC→Wikidata enrichment (Step A).
    # Extract owner names from MBFC ownership fields, Wikidata-expand them
    # to discover networks/media holdings. Runs BEFORE ranking so new
    # parties influence conflict detection.
    all_parties, affiliated_media = await _enrich_parties_from_mbfc(
        seed_results, all_parties, affiliated_media,
    )
    enriched_parties["all_parties"] = all_parties
    enriched_parties["affiliated_media"] = affiliated_media

    # Phase 1b-iii: Rank seeds by quality + conflict detection (with enriched parties)
    seed_results = await _rank_and_filter_seeds(
        seed_results,
        affiliated_media=affiliated_media,
        all_parties=all_parties,
    )

    # Phase 1c: Pre-fetch full page content from top-ranked seeds.
    # Parallel HTTP fetches (~3-5s) before the agent starts, so it sees
    # full articles from the best sources without spending tool calls.
    prefetch_messages, prefetched_urls = await _prefetch_seed_pages(seed_results)

    # Cap conflicted seeds shown to the agent — independent sources should
    # dominate what the agent sees so it doesn't spend tool calls on pages
    # that just quote interested parties.
    MAX_CONFLICTED_SEEDS = 5
    independent_seeds = [s for s in seed_results if not s.get("_conflict_flags")]
    conflicted_seeds = [s for s in seed_results if s.get("_conflict_flags")]
    agent_seeds = independent_seeds + conflicted_seeds[:MAX_CONFLICTED_SEEDS]
    seed_messages = _build_seed_messages(agent_seeds)

    log.info(logger, MODULE, "seed_complete",
             "Seed phase complete, starting agent",
             seed_results=len(seed_results),
             agent_seeds=len(agent_seeds),
             conflicted_hidden=len(conflicted_seeds) - min(len(conflicted_seeds), MAX_CONFLICTED_SEEDS),
             seed_messages=len(seed_messages),
             prefetched=len(prefetched_urls),
             prefetched_urls=prefetched_urls)

    # Phase 2: Agentic deep dive — LLM decides what to fetch/follow-up.
    # Collect messages incrementally via streaming so we keep evidence
    # even if the agent hits recursion limit or times out.
    # Prefetch messages go after seeds so the agent sees them as prior fetches.
    collected_messages: list = list(seed_messages) + list(prefetch_messages)

    try:
        agent = build_research_agent(wikidata_context, claim_date=claim_date)
        input_msg = HumanMessage(
            content=RESEARCH_USER.format(
                sub_claim=sub_claim,
                speaker_line=f"\nSpeaker: {speaker}" if speaker else "",
            )
        )

        async def _run_stream():
            async for chunk in agent.astream(
                {"messages": [input_msg] + seed_messages + prefetch_messages},
                config={"recursion_limit": max_steps},
                stream_mode="updates",
            ):
                # Each chunk is {node_name: state_update}
                # The "tools" node emits {"messages": [ToolMessage, ...]}
                for node_name, update in chunk.items():
                    msgs = update.get("messages", [])
                    collected_messages.extend(msgs)

        await asyncio.wait_for(_run_stream(), timeout=timeout_secs)

        evidence = extract_evidence(collected_messages)
        log.info(logger, MODULE, "done", "Research agent complete",
                 sub_claim=sub_claim, evidence_count=len(evidence),
                 agent_steps=len(collected_messages),
                 seed_results=len(seed_results))

        # Phase 3a: Programmatic enrichment: LegiScan (legislation, votes, bill text)
        evidence = await _enrich_with_legislation(evidence, sub_claim)

        # Phase 3b: Evidence NER→Wikidata enrichment (Step B).
        # Discover new interested parties from evidence article content.
        all_parties, affiliated_media = await _enrich_parties_from_evidence_content(
            evidence, all_parties, affiliated_media,
        )
        enriched_parties["all_parties"] = all_parties
        enriched_parties["affiliated_media"] = affiliated_media

        return evidence, enriched_parties

    except Exception as e:
        from langgraph.errors import GraphRecursionError

        # Extract whatever evidence was gathered before the interruption
        evidence = extract_evidence(collected_messages)

        if isinstance(e, asyncio.TimeoutError):
            error_type = "TimeoutError"
            log_msg = "Research agent timed out"
        elif isinstance(e, GraphRecursionError):
            error_type = "GraphRecursionError"
            log_msg = "Research agent hit step limit"
        else:
            error_type = type(e).__name__
            log_msg = "Research agent failed"

        if evidence:
            # We have partial evidence — use it instead of falling back
            log.info(logger, MODULE, "partial_evidence",
                     f"{log_msg}, using {len(evidence)} items gathered so far",
                     error_type=error_type, sub_claim=sub_claim,
                     evidence_count=len(evidence),
                     messages_collected=len(collected_messages))

            # Programmatic enrichment: LegiScan
            evidence = await _enrich_with_legislation(evidence, sub_claim)

            # Attempt Step B enrichment on partial evidence
            try:
                all_parties, affiliated_media = await _enrich_parties_from_evidence_content(
                    evidence, all_parties, affiliated_media,
                )
                enriched_parties["all_parties"] = all_parties
                enriched_parties["affiliated_media"] = affiliated_media
            except Exception:
                pass  # Keep whatever enrichment we have

            return evidence, enriched_parties

        # No evidence gathered at all — fall back to direct search
        log.error(logger, MODULE, "agent_failed",
                  f"{log_msg} with no evidence, falling back to direct search",
                  error=str(e), error_type=error_type,
                  sub_claim=sub_claim)
        fallback_evidence = await _research_fallback(sub_claim)
        return fallback_evidence, enriched_parties


_US_LEGISLATION_SIGNALS = {
    "congress", "senate", "house of representatives", "bill", "legislation",
    "act", "law", "federal", "statute", "amendment", "voted", "vote",
    "bipartisan", "republican", "democrat", "gop", "capitol hill",
    "signed into law", "executive order", "inflation reduction",
    "affordable care", "appropriations", "filibuster",
}

_NON_US_SIGNALS = {
    "eu", "european", "parliament", "brexit", "nato",
    "switzerland", "swiss", "germany", "france", "china", "india",
    "japan", "australia", "canada", "brazil", "russia", "uk",
    "united kingdom", "african", "asia", "latin america",
}


def _is_us_legislation_relevant(sub_claim: str) -> bool:
    """Check if a subclaim is likely related to US legislation.

    LegiScan only covers US bills. Searching non-US claims returns
    irrelevant matches on common words (e.g., "military" → random US bills).
    """
    lower = sub_claim.lower()

    # If it contains US-specific legislative terms, search
    if any(signal in lower for signal in _US_LEGISLATION_SIGNALS):
        # But not if it's clearly about another country
        if not any(signal in lower for signal in _NON_US_SIGNALS):
            return True
        # US legislative terms + non-US country → could still be about US
        # (e.g., "US trade bill with China"), check for explicit US mention
        return "united states" in lower or "u.s." in lower or "american" in lower

    return False


async def _enrich_with_legislation(
    evidence: list[dict], sub_claim: str
) -> list[dict]:
    """Programmatic LegiScan enrichment: search for matching US legislation.

    Runs after the agent finishes gathering web evidence. Only searches
    if the subclaim appears related to US legislation (keyword check).

    Deduplicates against URLs already in the evidence set.
    """
    if not legiscan_available():
        return evidence

    if not _is_us_legislation_relevant(sub_claim):
        log.debug(logger, MODULE, "legiscan_skipped",
                  "Subclaim not US-legislation-relevant, skipping LegiScan",
                  sub_claim=sub_claim[:80])
        return evidence

    try:
        legiscan_items = await search_legislation(sub_claim)
    except Exception as e:
        log.warning(logger, MODULE, "legiscan_failed",
                    "LegiScan enrichment failed",
                    error=str(e), sub_claim=sub_claim)
        return evidence

    if not legiscan_items:
        return evidence

    # No URL dedup against agent evidence — LegiScan returns structured
    # data (votes, bill text, sponsors) that's fundamentally different from
    # what the agent found at the same URL via web search. The agent gets
    # the HTML page; LegiScan gets roll call votes with member positions.
    evidence.extend(legiscan_items)
    log.info(logger, MODULE, "legiscan_enriched",
             "Added legislative evidence",
             sub_claim=sub_claim,
             items_added=len(legiscan_items),
             total_evidence=len(evidence))

    return evidence


async def _research_fallback(sub_claim: str) -> list[dict]:
    """Fallback: direct tool calls without the agent loop.

    If the ReAct agent fails (e.g., tool calling not supported by the LLM),
    we fall back to running search tools directly. No LLM reasoning about
    what to search — we just search for the claim text directly.

    Uses the best available search tool (Serper > SearXNG > Brave > DDG).
    """
    log.info(logger, MODULE, "fallback_start", "Running fallback direct search",
             sub_claim=sub_claim)
    evidence = []

    # Try Serper first (Google results — primary)
    if serper_available():
        try:
            results = await search_serper(sub_claim, max_results=8)
            for r in results:
                evidence.append({
                    "source_type": "web",
                    "source_url": r.get("url"),
                    "title": r.get("title", ""),
                    "content": r.get("snippet", ""),
                })
        except Exception as e:
            log.warning(logger, MODULE, "fallback_serper_failed",
                        "Serper fallback search failed",
                        error=str(e), error_type=type(e).__name__)

    # SearXNG fallback disabled — too much topically irrelevant noise

    # Try Brave (different index = different results)
    if brave_available():
        try:
            results = await search_brave(sub_claim, max_results=5)
            for r in results:
                evidence.append({
                    "source_type": "web",
                    "source_url": r.get("url"),
                    "title": r.get("title", ""),
                    "content": r.get("snippet", ""),
                })
        except Exception as e:
            log.warning(logger, MODULE, "fallback_brave_failed",
                        "Brave fallback search failed",
                        error=str(e), error_type=type(e).__name__)

    # DuckDuckGo fallback (always available)
    if not evidence:
        try:
            ddg_tool = get_web_search_tool()
            results = ddg_tool.invoke(sub_claim)
            if results and results.strip():
                evidence.append({
                    "source_type": "web",
                    "source_url": None,
                    "content": results[:3000],
                })
        except Exception as e:
            log.warning(logger, MODULE, "fallback_ddg_failed",
                        "DuckDuckGo fallback search failed",
                        error=str(e), error_type=type(e).__name__)

    # Wikipedia search (always available)
    try:
        from src.tools.wikipedia import search_wikipedia
        wiki_results = await search_wikipedia(sub_claim, max_results=3)
        for r in wiki_results:
            evidence.append({
                "source_type": "wikipedia",
                "source_url": r.get("url"),
                "title": r.get("title", ""),
                "content": r.get("summary", ""),
            })
    except Exception as e:
        log.warning(logger, MODULE, "fallback_wiki_failed",
                    "Wikipedia fallback search failed",
                    error=str(e), error_type=type(e).__name__)

    # Cap evidence to avoid overwhelming the judge — 6 items is plenty
    # for a verdict. More just means slower LLM inference.
    if len(evidence) > 6:
        log.info(logger, MODULE, "fallback_capped",
                 "Capping fallback evidence",
                 original=len(evidence), capped=6)
        evidence = evidence[:6]

    # Programmatic enrichment: LegiScan
    evidence = await _enrich_with_legislation(evidence, sub_claim)

    log.info(logger, MODULE, "fallback_done", "Fallback search complete",
             sub_claim=sub_claim, evidence_count=len(evidence))
    return evidence
