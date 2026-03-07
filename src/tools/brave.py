"""Brave Search tool for evidence gathering.

Brave Search (https://brave.com/search/api/) has its own independent index —
it doesn't just proxy Google. This gives genuinely different results, which
is valuable for cross-referencing.
  - 2,000 free queries/month
  - Independent index (not Google-derived)
  - Returns titles, descriptions, URLs, extra snippets

Env var gated: no BRAVE_API_KEY → tool is not registered with the agent.
"""

import os
import httpx
from langchain_core.tools import tool

from src.tools.source_filter import filter_results, warm_mbfc_cache_background
from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Circuit breaker: disable after first HTTP error to avoid wasting tool budget.
_disabled = False


def is_available() -> bool:
    """Check if the Brave API key is configured and not circuit-broken."""
    return bool(BRAVE_API_KEY) and not _disabled


async def search_brave(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Brave Search API.

    Returns a list of {title, snippet, url} dicts.
    """
    global _disabled
    if not BRAVE_API_KEY or _disabled:
        return []

    import time as _time
    log.info(logger, MODULE, "brave_start", "Brave search starting",
             query=query, max_results=max_results)
    _t0 = _time.monotonic()

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                BRAVE_URL,
                params={
                    "q": query,
                    "count": max_results,
                    "text_decorations": False,
                    "search_lang": "en",
                },
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_API_KEY,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:max_results + 5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                })

            raw_count = len(results)

            await warm_mbfc_cache_background(results)
            results = filter_results(results)[:max_results]

            log.info(logger, MODULE, "brave_done", "Brave search complete",
                     query=query, raw_count=raw_count, result_count=len(results),
                     latency_ms=int((_time.monotonic() - _t0) * 1000))
            return results

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 401, 403, 422):
                _disabled = True
                log.warning(logger, MODULE, "brave_disabled",
                            "Brave returned error, disabling for this process",
                            error=str(e), status=e.response.status_code,
                            query=query)
            else:
                log.warning(logger, MODULE, "brave_failed",
                            "Brave search failed",
                            error=str(e), error_type=type(e).__name__,
                            query=query)
            return []
        except Exception as e:
            log.warning(logger, MODULE, "brave_failed", "Brave search failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return []


def get_brave_tool():
    """Get a LangChain tool that wraps Brave Search.

    Returns a @tool-decorated async function compatible with LangGraph agents.
    """

    @tool
    async def brave_search(query: str) -> str:
        """Search the web using Brave Search for evidence about a claim.

        Best for: getting a DIFFERENT perspective from Google. Brave has its
        own independent search index, so it often finds sources that Google
        misses or ranks differently. Use this alongside Google/Serper for
        source diversity.
        """
        if _disabled:
            return "Brave Search is unavailable. Use SearXNG or DuckDuckGo instead."

        log.debug(logger, MODULE, "brave_query", "Brave search query",
                  query=query)
        try:
            results = await search_brave(query, max_results=5)
        except Exception as e:
            log.warning(logger, MODULE, "brave_tool_failed",
                        "Brave search tool failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return "Brave search failed. Try another search tool."

        if not results:
            if _disabled:
                return "Brave Search is unavailable. Use SearXNG or DuckDuckGo instead."
            return "No Brave search results found."

        parts = []
        for r in results:
            parts.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['snippet']}"
            )
        return "\n\n---\n\n".join(parts)

    return brave_search
