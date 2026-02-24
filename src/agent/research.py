"""LangGraph research agent for evidence gathering.

This module implements the core agentic AI pattern: a ReAct (Reason + Act)
agent that autonomously searches for evidence using web tools.

## How it works

LangGraph's create_react_agent builds a StateGraph with two nodes:

    ┌──────────┐     ┌───────┐
    │  agent   │────▶│ tools │
    │  (LLM)   │◀────│       │
    └────┬─────┘     └───────┘
         │ (no more tool calls)
         ▼
        END

1. The "agent" node calls the LLM with tool definitions in the request.
   The LLM reads the conversation history and decides either:
     (a) Call a tool — returns an AIMessage with tool_calls
     (b) Respond — returns an AIMessage with text content (no tool_calls)

2. If the LLM chose (a), the "tools" node executes the tool calls and
   appends ToolMessage results to the conversation.

3. Loop back to the "agent" node. The LLM now sees the tool results
   and decides what to do next.

4. When the LLM chooses (b), the graph ends.

This is the ReAct pattern — the LLM Reasons about what to do, Acts by
calling tools, Observes the results, and Reasons again. It's the standard
agentic AI pattern used across the LangChain/LangGraph ecosystem.

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

## Integration with Temporal

This agent runs INSIDE the research_subclaim Temporal activity. This gives us:
  - Durable execution: if the container crashes mid-search, Temporal retries
  - Timeouts: if the agent loops forever, the activity timeout kills it
  - Retries: if the agent errors, Temporal retries the whole activity
  - Observability: activity status visible in Temporal UI
"""

import logging

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from src.llm import get_reasoning_llm
from src.prompts.verification import RESEARCH_SYSTEM, RESEARCH_USER
from src.tools.web_search import get_web_search_tool
from src.tools.wikipedia import get_wikipedia_tool
from src.tools.page_fetcher import get_page_fetcher_tool
from src.tools.serper import get_serper_tool, is_available as serper_available
from src.tools.brave import get_brave_tool, is_available as brave_available
from src.tools.searxng import get_searxng_tool, is_available as searxng_available
from src.utils.logging import log, get_logger

MODULE = "research"
logger = get_logger()


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
        log.info(logger, MODULE, "tool_enabled", "SearXNG meta-search enabled")

    if serper_available():
        tools.append(get_serper_tool())
        log.info(logger, MODULE, "tool_enabled", "Serper search enabled")

    if brave_available():
        tools.append(get_brave_tool())
        log.info(logger, MODULE, "tool_enabled", "Brave search enabled")

    # DuckDuckGo — always available as fallback
    tools.append(get_web_search_tool())

    # Wikipedia — always available
    tools.append(get_wikipedia_tool())

    # Page fetcher — always available, lets agent read full articles
    tools.append(get_page_fetcher_tool())

    tool_names = [t.name for t in tools]
    log.info(logger, MODULE, "tools_loaded", "Research agent tools configured",
             tool_count=len(tools), tools=tool_names)

    return tools


def build_research_agent():
    """Build a ReAct agent for evidence gathering.

    Tools are dynamically loaded based on configured API keys.
    See _build_tool_list() for the full list and priority order.

    The `prompt` parameter injects a SystemMessage that tells the agent
    how to research (search strategies, when to stop, what to report).
    See RESEARCH_SYSTEM in src/prompts/verification.py for the full prompt.
    """
    llm = get_reasoning_llm(temperature=0.2)
    tools = _build_tool_list()

    return create_react_agent(
        llm,
        tools,
        prompt=RESEARCH_SYSTEM,
    )


def extract_evidence(messages: list) -> list[dict]:
    """Extract structured evidence records from the agent's conversation.

    After the ReAct agent runs, its message history contains:
      1. SystemMessage — research instructions (from RESEARCH_SYSTEM)
      2. HumanMessage — "Find evidence about: {claim}"
      3. AIMessage with tool_calls — agent decides to search
      4. ToolMessage — search results from DuckDuckGo/Wikipedia
      5. ... (more tool calls and results)
      N. AIMessage — agent's final summary (no tool_calls)

    We extract evidence from ToolMessage objects, since those contain
    the actual search results. Each ToolMessage becomes one evidence record.

    The agent's final AIMessage is captured separately as an "agent_summary"
    evidence item — this contains the agent's analysis of what it found,
    which is useful context for the judge step.
    """
    evidence = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        # Determine source type from the tool that was called
        tool_name = getattr(msg, "name", "") or ""
        if "wikipedia" in tool_name.lower():
            source_type = "wikipedia"
        elif "fetch_page" in tool_name.lower():
            source_type = "web"  # Page content is still "web" evidence
        else:
            source_type = "web"  # serper, brave, google, ddg → all "web"

        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Skip empty or unhelpful results
        if not content or content.strip() in (
            "",
            "No results found.",
            "No Wikipedia results found.",
        ):
            continue

        evidence.append({
            "source_type": source_type,
            "source_url": None,    # URLs are embedded in the content text
            "content": content[:3000],  # Truncate very long search results
            "supports_claim": None,     # Determined later by the judge step
        })

    return evidence


async def research_claim(sub_claim: str, max_steps: int = 25, timeout_secs: int = 240) -> list[dict]:
    """Run the research agent to gather evidence for a sub-claim.

    This is the main entry point called by the research_subclaim activity.

    Args:
        sub_claim: The specific sub-claim to find evidence for.
        max_steps: Maximum number of graph steps (prevents infinite loops).
                   Each tool call costs ~2 steps (agent node + tool node).
                   25 steps allows ~10-12 tool calls, which is generous.
        timeout_secs: Soft timeout in seconds. If the agent exceeds this,
                      we cancel it and return whatever evidence was gathered
                      so far via the fallback. Default 180s (3 min) to stay
                      well under the 300s Temporal activity timeout.

    Returns:
        List of evidence dicts, each with:
          - source_type: "web", "wikipedia", or "agent_summary"
          - source_url: URL if available (often None — embedded in content)
          - content: The evidence text/snippet
          - supports_claim: None (determined by the judge step later)
    """
    import asyncio

    log.info(logger, MODULE, "start", "Starting research agent",
             sub_claim=sub_claim)

    try:
        agent = build_research_agent()

        result = await asyncio.wait_for(
            agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=RESEARCH_USER.format(sub_claim=sub_claim)
                        )
                    ]
                },
                config={"recursion_limit": max_steps},
            ),
            timeout=timeout_secs,
        )

        # Extract evidence from tool results in the conversation
        evidence = extract_evidence(result["messages"])

        # Also capture the agent's final analysis as a summary
        final_msg = result["messages"][-1]
        if isinstance(final_msg, AIMessage) and final_msg.content:
            evidence.append({
                "source_type": "agent_summary",
                "source_url": None,
                "content": final_msg.content[:2000],
                "supports_claim": None,
            })

        log.info(logger, MODULE, "done", "Research agent complete",
                 sub_claim=sub_claim, evidence_count=len(evidence),
                 agent_steps=len(result["messages"]))
        return evidence

    except asyncio.TimeoutError:
        log.warning(logger, MODULE, "agent_timeout",
                    "Research agent timed out, falling back to direct search",
                    timeout_secs=timeout_secs, sub_claim=sub_claim)
        return await _research_fallback(sub_claim)

    except Exception as e:
        # If the ReAct agent fails, fall back to direct tool calls.
        # This can happen if:
        #   - The LLM doesn't support tool calling properly
        #   - DuckDuckGo is rate-limited or down
        #   - Network issues
        # The judge step will see the evidence and can still work with it,
        # or will return "unverifiable" if there's nothing useful.
        log.warning(logger, MODULE, "agent_failed",
                    "Research agent failed, falling back to direct search",
                    error=str(e), error_type=type(e).__name__,
                    sub_claim=sub_claim)
        return await _research_fallback(sub_claim)


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
