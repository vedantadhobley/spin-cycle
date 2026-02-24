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

from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def is_available() -> bool:
    """Check if the Brave API key is configured."""
    return bool(BRAVE_API_KEY)


async def search_brave(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Brave Search API.

    Returns a list of {title, snippet, url} dicts.
    """
    if not BRAVE_API_KEY:
        return []

    log.debug(logger, MODULE, "brave_start", "Brave search starting",
              query=query[:80], max_results=max_results)

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
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                })

            log.debug(logger, MODULE, "brave_done", "Brave search complete",
                      query=query[:50], result_count=len(results))
            return results

        except Exception as e:
            log.warning(logger, MODULE, "brave_failed", "Brave search failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query[:80])
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
        try:
            results = await search_brave(query, max_results=5)
        except Exception as e:
            log.warning(logger, MODULE, "brave_tool_failed",
                        "Brave search tool failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query[:80])
            return "Brave search failed. Try another search tool."

        if not results:
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
