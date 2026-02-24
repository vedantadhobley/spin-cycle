"""Serper search tool for evidence gathering.

Serper (https://serper.dev) provides Google search results via a clean JSON API.
  - 2,500 free queries/month
  - Returns titles, snippets, URLs, knowledge graph, "people also ask"
  - Google index = best coverage

Env var gated: no SERPER_API_KEY → tool is not registered with the agent.
"""

import os
import httpx
from langchain_core.tools import tool

from src.tools.source_filter import filter_results
from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"


def is_available() -> bool:
    """Check if the Serper API key is configured."""
    return bool(SERPER_API_KEY)


async def search_serper(query: str, max_results: int = 5) -> list[dict]:
    """Search Google via Serper API.

    Returns a list of {title, snippet, url} dicts.
    """
    if not SERPER_API_KEY:
        return []

    log.debug(logger, MODULE, "serper_start", "Serper search starting",
              query=query, max_results=max_results)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                SERPER_URL,
                json={"q": query, "num": max_results},
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []

            # Organic results (main search results)
            for item in data.get("organic", [])[:max_results + 5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                })

            # Knowledge graph (if present — often has authoritative info)
            kg = data.get("knowledgeGraph", {})
            if kg and kg.get("description"):
                results.append({
                    "title": kg.get("title", "Knowledge Graph"),
                    "snippet": kg.get("description", ""),
                    "url": kg.get("descriptionLink") or kg.get("website") or "",
                })

            results = filter_results(results)[:max_results]

            log.debug(logger, MODULE, "serper_done", "Serper search complete",
                      query=query, result_count=len(results))
            return results

        except Exception as e:
            log.warning(logger, MODULE, "serper_failed", "Serper search failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return []


def get_serper_tool():
    """Get a LangChain tool that wraps Serper search.

    Returns a @tool-decorated async function compatible with LangGraph agents.
    """

    @tool
    async def serper_search(query: str) -> str:
        """Search Google via Serper for evidence about a claim.

        Best for: finding news articles, official reports, press releases,
        government documents, and any publicly available information.
        This searches the full Google index — the most comprehensive
        web search available.
        """
        try:
            results = await search_serper(query, max_results=5)
        except Exception as e:
            log.warning(logger, MODULE, "serper_tool_failed",
                        "Serper search tool failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return "Google search failed. Try another search tool."

        if not results:
            return "No Google results found."

        parts = []
        for r in results:
            parts.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['snippet']}"
            )
        return "\n\n---\n\n".join(parts)

    return serper_search
