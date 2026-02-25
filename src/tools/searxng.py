"""SearXNG meta-search tool for evidence gathering.

SearXNG is a self-hosted meta-search engine that aggregates results from
multiple search engines (Google, Bing, DuckDuckGo, Brave, Mojeek, etc.).
  - Free, unlimited, self-hosted
  - Aggregates 70+ search engines
  - JSON API for programmatic access
  - No API keys needed

Requires the SearXNG container to be running (see docker-compose.dev.yml).
The SEARXNG_URL env var points to the container's JSON API endpoint.
"""

import os
import httpx
from langchain_core.tools import tool

from src.tools.source_filter import filter_results
from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

SEARXNG_URL = os.getenv("SEARXNG_URL", "")


def is_available() -> bool:
    """Check if SearXNG is configured."""
    return bool(SEARXNG_URL)


async def search_searxng(
    query: str,
    max_results: int = 5,
    categories: str = "general",
) -> list[dict]:
    """Search via self-hosted SearXNG instance.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        categories: SearXNG categories — "general", "news", "science", etc.

    Returns a list of {title, snippet, url, engines} dicts.
    """
    if not SEARXNG_URL:
        return []

    import time as _time
    log.debug(logger, MODULE, "searxng_start", "SearXNG search starting",
              query=query, max_results=max_results, categories=categories)
    _t0 = _time.monotonic()

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": categories,
                    "language": "en",
                    "pageno": 1,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:max_results + 5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url": item.get("url", ""),
                    "engines": item.get("engines", []),
                })

            results = filter_results(results)[:max_results]

            log.debug(logger, MODULE, "searxng_done", "SearXNG search complete",
                      query=query, result_count=len(results),
                      latency_ms=int((_time.monotonic() - _t0) * 1000))
            return results

        except Exception as e:
            log.warning(logger, MODULE, "searxng_failed", "SearXNG search failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return []


def get_searxng_tool():
    """Get a LangChain tool that wraps SearXNG meta-search.

    Returns a @tool-decorated async function compatible with LangGraph agents.
    """

    @tool
    async def searxng_search(query: str) -> str:
        """Search the web using SearXNG, a meta-search engine that aggregates
        results from Google, Bing, DuckDuckGo, Brave, and many other engines.

        This is the primary search tool — it searches multiple engines at once
        and returns combined results. Use this first for broad web searches.
        For established facts and background, also use wikipedia_search.
        When you find a promising URL, use fetch_page_content to read the full article.
        """
        log.debug(logger, MODULE, "searxng_query", "SearXNG search",
                  query=query)
        try:
            results = await search_searxng(query, max_results=8)
        except Exception as e:
            log.warning(logger, MODULE, "searxng_tool_failed",
                        "SearXNG search tool failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return "SearXNG search failed. Try web_search as a fallback."

        if not results:
            return "No SearXNG results found. Try web_search as a fallback."

        parts = []
        for r in results:
            engines_str = ", ".join(r["engines"][:3]) if r["engines"] else "unknown"
            parts.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Engines: {engines_str}\n"
                f"Snippet: {r['snippet']}"
            )
        return "\n\n---\n\n".join(parts)

    return searxng_search
