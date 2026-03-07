"""Web search tool for evidence gathering.

DuckDuckGo via langchain_community — uses the official DDG API (not scraping),
so it works reliably even when SearXNG's DDG engine is CAPTCHA-blocked.

Two interfaces:
  - search_duckduckgo(): async, returns list[dict] — for seed searches
  - get_web_search_tool(): LangChain @tool for the ReAct agent
"""

import re as _re
import time as _time

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from src.tools.source_filter import filter_results
from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()


def _parse_ddg_text(raw: str) -> list[dict]:
    """Parse DuckDuckGoSearchResults text output into structured dicts.

    DDG returns a flat string like:
      'snippet: ..., title: ..., link: https://..., snippet: ..., title: ..., link: ...'
    We split on ', link: https?://' boundaries and then extract fields from each chunk.
    """
    results = []
    # Split on link boundaries — each chunk ends with the URL for that result
    # Pattern: everything before the link, then the URL
    # Use findall to grab each (snippet_block, url)
    pattern = r"snippet:\s*(.+?),\s*title:\s*(.+?),\s*link:\s*(https?://\S+)"
    for match in _re.finditer(pattern, raw):
        snippet = match.group(1).strip().rstrip(",")
        title = match.group(2).strip().rstrip(",")
        url = match.group(3).strip().rstrip(",")
        if url:
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
            })
    return results


async def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return structured results.

    Uses langchain_community's DuckDuckGoSearchResults (official API).
    Returns list of {title, snippet, url} dicts — same schema as
    search_serper/search_searxng/search_brave.
    """
    log.info(logger, MODULE, "ddg_start", "DuckDuckGo search starting",
             query=query, max_results=max_results)
    t0 = _time.monotonic()

    try:
        ddg = DuckDuckGoSearchResults(
            name="_ddg_seed",
            max_results=max_results + 5,
        )
        raw = ddg.invoke(query)
        raw_str = raw if isinstance(raw, str) else str(raw)
        results = _parse_ddg_text(raw_str)

        raw_count = len(results)
        results = filter_results(results)[:max_results]

        latency_ms = round((_time.monotonic() - t0) * 1000)
        log.info(logger, MODULE, "ddg_done", "DuckDuckGo search complete",
                 query=query, raw_count=raw_count, result_count=len(results),
                 latency_ms=latency_ms)
        return results

    except Exception as e:
        latency_ms = round((_time.monotonic() - t0) * 1000)
        log.warning(logger, MODULE, "ddg_error", "DuckDuckGo search failed",
                    query=query, error=str(e), latency_ms=latency_ms)
        return []


def get_web_search_tool():
    """Get a DuckDuckGo web search tool with logging wrapper."""
    _ddg = DuckDuckGoSearchResults(
        name="_ddg_inner",
        max_results=5,
    )

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web using DuckDuckGo. This is the primary search tool — use it for finding news articles, reports, and evidence about claims."""
        log.info(logger, MODULE, "ddg_query", "DuckDuckGo search",
                 query=query)
        t0 = _time.monotonic()
        try:
            result = _ddg.invoke(query)
            latency_ms = round((_time.monotonic() - t0) * 1000)
            result_str = result if isinstance(result, str) else str(result)
            # Count results by snippet separators
            result_count = result_str.count("snippet:") if result_str else 0
            log.info(logger, MODULE, "ddg_done", "DuckDuckGo search complete",
                     query=query, result_count=result_count,
                     latency_ms=latency_ms)
            return result_str
        except Exception as e:
            latency_ms = round((_time.monotonic() - t0) * 1000)
            log.warning(logger, MODULE, "ddg_error", "DuckDuckGo search failed",
                        query=query, error=str(e), latency_ms=latency_ms)
            return f"DuckDuckGo search failed: {e}"

    log.info(logger, MODULE, "ddg_init", "DuckDuckGo search tool initialized")
    return web_search
