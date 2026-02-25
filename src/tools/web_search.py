"""Web search tool for evidence gathering."""

import time as _time

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()


def get_web_search_tool():
    """Get a DuckDuckGo web search tool with logging wrapper."""
    _ddg = DuckDuckGoSearchResults(
        name="_ddg_inner",
        max_results=5,
    )

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web using DuckDuckGo. Free fallback search — use serper_search or brave_search first if available, as they give better results. Good for quick general queries."""
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

    log.debug(logger, MODULE, "ddg_init", "DuckDuckGo search tool initialized")
    return web_search
