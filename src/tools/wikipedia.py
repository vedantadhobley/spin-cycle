"""Wikipedia lookup tool for evidence gathering.

Two interfaces:
  1. search_wikipedia() — raw async function returning structured dicts
  2. get_wikipedia_tool() — LangChain @tool wrapper for use with LangGraph agents

The LangChain tool returns formatted text (for the LLM to read), while the
raw function returns dicts (for programmatic use).
"""

import httpx
from langchain_core.tools import tool


async def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """Search Wikipedia for articles related to the query.

    Returns a list of {title, summary, url} dicts.
    """
    async with httpx.AsyncClient() as client:
        # Wikipedia API search
        # NOTE: Wikipedia requires a User-Agent header. Without it, you get 403.
        # See: https://meta.wikimedia.org/wiki/User-Agent_policy
        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
            headers={
                "User-Agent": "SpinCycle/0.1 (claim verification research tool)",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item["title"],
                "summary": item.get("snippet", ""),
                "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
            })

        return results


def get_wikipedia_tool():
    """Get a LangChain tool that wraps Wikipedia search.

    Returns a @tool-decorated function compatible with LangGraph agents.
    The tool returns formatted text — not raw dicts — because LangGraph
    tool nodes pass tool results as text to the next LLM call.
    """

    @tool
    async def wikipedia_search(query: str) -> str:
        """Search Wikipedia for factual information about a topic.

        Use for established facts, historical events, organisations,
        notable people, and verifiable statistics. Returns article
        titles, URLs, and summary snippets.
        """
        try:
            results = await search_wikipedia(query, max_results=3)
        except Exception:
            return "Wikipedia search failed. Try web search instead."

        if not results:
            return "No Wikipedia results found."

        parts = []
        for r in results:
            parts.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Summary: {r['summary']}"
            )
        return "\n\n---\n\n".join(parts)

    return wikipedia_search
