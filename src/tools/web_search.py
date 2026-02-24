"""Web search tool for evidence gathering."""

from langchain_community.tools import DuckDuckGoSearchResults

from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()


def get_web_search_tool():
    """Get a DuckDuckGo web search tool for the agent."""
    tool = DuckDuckGoSearchResults(
        name="web_search",
        description="Search the web using DuckDuckGo. Free fallback search — use serper_search or brave_search first if available, as they give better results. Good for quick general queries.",
        max_results=5,
    )
    log.debug(logger, MODULE, "ddg_init", "DuckDuckGo search tool initialized")
    return tool
