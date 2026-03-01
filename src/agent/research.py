"""LangGraph research agent for evidence gathering.

This module implements the core agentic AI pattern: a ReAct (Reason + Act)
agent that autonomously searches for evidence using web tools.

## How it works

LangGraph's create_react_agent builds a StateGraph with a pre-model hook:

    ┌────────────┐     ┌──────────┐     ┌───────┐
    │ pre_model  │────▶│  agent   │────▶│ tools │
    │ (progress) │     │  (LLM)   │◀────│       │
    └────────────┘     └────┬─────┘     └───────┘
                            │ (no more tool calls)
                            ▼
                           END

1. The "pre_model" hook analyzes the conversation so far — counting tool
   calls, unique URLs, search queries used, engines tried — and injects
   a progress summary into the LLM's input (without modifying state).

2. The "agent" node calls the LLM with tool definitions + progress note.
   The LLM reads the conversation history and progress, then decides:
     (a) Call a tool — returns an AIMessage with tool_calls
     (b) Respond — returns an AIMessage with text content (no tool_calls)

3. If the LLM chose (a), the "tools" node executes the tool calls and
   appends ToolMessage results to the conversation.

4. Loop back to pre_model → agent. The progress note updates each
   iteration, giving the agent real-time awareness of what it has.

5. When the LLM chooses (b), the graph ends.

This is the ReAct pattern with progress awareness — the agent knows what
it's already searched, how many unique sources it has, and which engines
it hasn't tried, so it can make strategic decisions about next steps.

## Why this is "agentic"

Unlike a simple prompt→response, this agent:
  - Decides what to search for (based on the claim)
  - Executes real web searches (Serper/Google, Brave, DuckDuckGo, Wikipedia)
  - Reads actual page content when search snippets aren't enough
  - Decides if it needs more information and adapts its strategy
  - Stops when it has enough evidence (or gives up)

The LLM is making autonomous decisions at each step — that's what makes
it an agent rather than just a chatbot.

## Tool selection

Tools are dynamically registered based on which API keys are configured:
  - searxng_search: Self-hosted meta-search aggregating many engines (needs SEARXNG_URL)
  - serper_search: Google results via Serper API (needs SERPER_API_KEY)
  - brave_search: Brave Search independent index (needs BRAVE_API_KEY)
  - web_search: DuckDuckGo fallback (always available, no key needed)
  - wikipedia_search: Wikipedia API (always available, no key needed)
  - fetch_page_content: Read full page text from URLs (always available)

Set API keys in .env to enable/disable tools. No key = tool not loaded.

Programmatic enrichment (runs after the agent, not as agent tools):
  - LegiScan: US legislation, bill text, votes, sponsors (needs LEGISCAN_API_KEY)

## Integration with Temporal

This agent runs INSIDE the research_subclaim Temporal activity. This gives us:
  - Durable execution: if the container crashes mid-search, Temporal retries
  - Timeouts: if the agent loops forever, the activity timeout kills it
  - Retries: if the agent errors, Temporal retries the whole activity
  - Observability: activity status visible in Temporal UI
"""

import re as _re
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.llm import get_llm
from src.prompts.verification import RESEARCH_SYSTEM, RESEARCH_USER
from src.tools.web_search import get_web_search_tool
from src.tools.wikipedia import get_wikipedia_tool
from src.tools.page_fetcher import get_page_fetcher_tool
from src.tools.serper import get_serper_tool, is_available as serper_available
from src.tools.brave import get_brave_tool, is_available as brave_available
from src.tools.searxng import get_searxng_tool, is_available as searxng_available
from src.tools.legiscan import search_legislation, is_available as legiscan_available
from src.utils.logging import log, get_logger
from src.utils.text_cleanup import cleanup_text

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
    tool_calls = 0
    search_queries: list[str] = []
    fetch_urls: list[str] = []
    evidence_urls: set[str] = set()
    evidence_domains: set[str] = set()
    tools_used: set[str] = set()

    for msg in messages:
        # Count tool calls from AIMessages
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls += 1
                tool_name = tc.get("name", "")
                tools_used.add(tool_name)
                args = tc.get("args", {})

                # Track search queries
                if "search" in tool_name.lower():
                    query = args.get("query") or args.get("input") or ""
                    if query:
                        search_queries.append(query)

                # Track page fetch URLs
                if "fetch" in tool_name.lower():
                    url = args.get("url", "")
                    if url:
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

    # Don't inject progress on the first call (no tool calls yet)
    if tool_calls == 0:
        return None

    lines = [f"[RESEARCH PROGRESS — {tool_calls} tool calls completed]"]

    # Evidence inventory
    lines.append(
        f"Evidence: {len(evidence_urls)} unique URLs across "
        f"{len(evidence_domains)} domains"
    )

    # Page fetches
    lines.append(f"Full articles read: {len(fetch_urls)}")

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
        if "searxng" in tl:
            engine_names.append("SearXNG")
        elif "serper" in tl:
            engine_names.append("Serper")
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
    if searxng_available():
        available_engines.add("SearXNG")
    if serper_available():
        available_engines.add("Serper")
    if brave_available():
        available_engines.add("Brave")
    unused_engines = available_engines - set(engine_names)
    if unused_engines and tool_calls <= 6:
        suggestions.append(
            f"try {', '.join(sorted(unused_engines))} for source diversity"
        )
    if len(fetch_urls) == 0 and len(evidence_urls) >= 4:
        suggestions.append(
            "fetch full articles from your best URLs — snippets may miss detail"
        )
    if len(fetch_urls) >= 3 and tool_calls >= 8:
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

    Priority order for search tools:
      1. SearXNG (self-hosted meta-search, free, aggregates many engines)
      2. Serper (Google index, reliable paid API)
      3. Brave (independent index, good for diversity)
      4. DuckDuckGo (free fallback, no key needed)

    Always included:
      - Wikipedia (free, reliable for established facts)
      - Page fetcher (reads URLs, no key needed)
    """
    tools = []

    # Search tools — add based on availability
    if searxng_available():
        tools.append(get_searxng_tool())
        log.debug(logger, MODULE, "tool_enabled", "SearXNG meta-search enabled")

    if serper_available():
        tools.append(get_serper_tool())
        log.debug(logger, MODULE, "tool_enabled", "Serper search enabled")

    if brave_available():
        tools.append(get_brave_tool())
        log.debug(logger, MODULE, "tool_enabled", "Brave search enabled")

    # DuckDuckGo — always available as fallback
    tools.append(get_web_search_tool())

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
    log.debug(logger, MODULE, "tools_loaded", "Research agent tools configured",
              tool_count=len(tools), tools=tool_names)

    return tools


def build_research_agent(interested_parties_context: str = ""):
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

    from datetime import date
    prompt = RESEARCH_SYSTEM.format(current_date=date.today().isoformat())

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
        title_m = _re.match(r"Page:\s*(.+)", stripped)
        if title_m:
            title = title_m.group(1).strip()
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
            "supports_claim": None,
        }]

    # Blocked/error responses from page fetcher
    if stripped.startswith(("Blocked source:", "Failed to fetch", "Invalid URL")):
        return []

    # Multi-result tools (SearXNG, Serper, Brave, Wikipedia, DDG)
    # Split on '---' separator used by all our tool wrappers
    blocks = _re.split(r"\n\n---\n\n", stripped)

    items = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title = ""
        url = None
        snippet = block  # fallback: use the whole block as content

        # Parse structured fields
        title_m = _re.match(r"(?:Title|Page):\s*(.+)", block)
        if title_m:
            title = title_m.group(1).strip()

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
            "supports_claim": None,
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
            # Clean up evidence text — catches grammar oddities from
            # web content and LLM-generated snippets
            if item.get("content"):
                item["content"] = cleanup_text(item["content"]) or item["content"]
            if item.get("title"):
                item["title"] = cleanup_text(item["title"]) or item["title"]
            evidence.append(item)

    log.info(logger, MODULE, "evidence_dedup",
             "Evidence extraction complete",
             total_items=total_items, unique_items=len(evidence),
             deduped=deduped_count, unique_urls=len(seen_urls))
    return evidence


async def research_claim(
    sub_claim: str,
    interested_parties_context: str = "",
    max_steps: int = 38,
    timeout_secs: int = 120,
) -> list[dict]:
    """Run the research agent to gather evidence for a sub-claim.

    This is the main entry point called by the research_subclaim activity.

    Uses streaming (astream) instead of ainvoke so that when the agent
    hits its step limit or times out, we keep ALL evidence gathered up
    to that point instead of falling back to a limited direct search.

    Args:
        sub_claim: The specific sub-claim to find evidence for.
        interested_parties_context: Pre-formatted text about interested
            parties and their Wikidata connections, injected into the
            agent's system prompt as context.
        max_steps: Maximum number of graph steps (the agent's budget).
                   Each tool call costs ~3 steps (pre_model + agent + tools).
                   38 steps allows ~12 tool calls. The final stop costs 2
                   steps (pre_model + agent deciding no more tools).
        timeout_secs: Soft timeout in seconds. If the agent exceeds this,
                      we return whatever evidence was gathered so far.
                      Default 120s (2 min).

    Returns:
        List of evidence dicts (deduplicated by URL), each with:
          - source_type: "web" or "wikipedia"
          - source_url: URL extracted from tool output (None for DDG)
          - title: Page/article title
          - content: The evidence text/snippet
          - supports_claim: None (determined by the judge step later)
    """
    import asyncio

    log.info(logger, MODULE, "start", "Starting research agent",
             sub_claim=sub_claim)

    # Collect messages incrementally via streaming so we keep evidence
    # even if the agent hits recursion limit or times out.
    collected_messages: list = []

    try:
        agent = build_research_agent(interested_parties_context)
        input_msg = HumanMessage(
            content=RESEARCH_USER.format(sub_claim=sub_claim)
        )

        async def _run_stream():
            async for chunk in agent.astream(
                {"messages": [input_msg]},
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
                 agent_steps=len(collected_messages))

        # Programmatic enrichment: LegiScan (legislation, votes, bill text)
        evidence = await _enrich_with_legislation(evidence, sub_claim)

        return evidence

    except (asyncio.TimeoutError, Exception) as e:
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

            return evidence

        # No evidence gathered at all — fall back to direct search
        log.error(logger, MODULE, "agent_failed",
                  f"{log_msg} with no evidence, falling back to direct search",
                  error=str(e), error_type=error_type,
                  sub_claim=sub_claim)
        return await _research_fallback(sub_claim)


async def _enrich_with_legislation(
    evidence: list[dict], sub_claim: str
) -> list[dict]:
    """Programmatic LegiScan enrichment: search for matching legislation.

    Runs after the agent finishes gathering web evidence. If the subclaim
    matches any legislation, appends bill details, roll call votes, and
    bill text as additional evidence items.

    Like Wikidata (decompose) and MBFC (source_filter), this is NOT an
    agent tool — it runs deterministically for every subclaim. LegiScan
    queries that don't match return empty, so there's no harm searching.

    Deduplicates against URLs already in the evidence set.
    """
    if not legiscan_available():
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

    Uses the best available search tool (SearXNG > Serper > Brave > DDG).
    """
    log.info(logger, MODULE, "fallback_start", "Running fallback direct search",
             sub_claim=sub_claim)
    evidence = []

    # Try SearXNG first (self-hosted meta-search)
    if searxng_available():
        try:
            from src.tools.searxng import search_searxng
            results = await search_searxng(sub_claim, max_results=8)
            for r in results:
                evidence.append({
                    "source_type": "web",
                    "source_url": r.get("url"),
                    "content": f"{r['title']}: {r['snippet']}",
                    "supports_claim": None,
                })
        except Exception as e:
            log.warning(logger, MODULE, "fallback_searxng_failed",
                        "SearXNG fallback search failed",
                        error=str(e), error_type=type(e).__name__)

    # Try Serper (best paid results)
    if serper_available():
        try:
            from src.tools.serper import search_serper
            results = await search_serper(sub_claim, max_results=5)
            for r in results:
                evidence.append({
                    "source_type": "web",
                    "source_url": r.get("url"),
                    "content": f"{r['title']}: {r['snippet']}",
                    "supports_claim": None,
                })
        except Exception as e:
            log.warning(logger, MODULE, "fallback_serper_failed",
                        "Serper fallback search failed",
                        error=str(e), error_type=type(e).__name__)

    # Try Brave (different index = different results)
    if brave_available():
        try:
            from src.tools.brave import search_brave
            results = await search_brave(sub_claim, max_results=5)
            for r in results:
                evidence.append({
                    "source_type": "web",
                    "source_url": r.get("url"),
                    "content": f"{r['title']}: {r['snippet']}",
                    "supports_claim": None,
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
                    "supports_claim": None,
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
                "content": f"{r['title']}: {r['summary']}",
                "supports_claim": None,
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

    log.info(logger, MODULE, "fallback_done", "Fallback search complete",
             sub_claim=sub_claim, evidence_count=len(evidence))
    return evidence
