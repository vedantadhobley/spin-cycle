"""Web search tool for evidence gathering."""

from langchain_community.tools import DuckDuckGoSearchResults

from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()


def get_web_search_tool():
    """Get a DuckDuckGo web search tool for the agent."""
    tool = DuckDuckGoSearchResults(
        name="web_search",
        description="Search the web for evidence about a claim. Use for finding news articles, fact-check reports, and primary sources.",
        max_results=5,
    )
    log.debug(logger, MODULE, "ddg_init", "DuckDuckGo search tool initialized")
    return tool
