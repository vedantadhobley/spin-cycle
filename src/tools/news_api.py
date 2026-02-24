"""News API search tool for evidence gathering."""

import os
import httpx

from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")


async def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Search NewsAPI for recent articles related to the query.

    Returns a list of {title, description, url, source, published_at} dicts.
    """
    if not NEWSAPI_KEY:
        log.debug(logger, MODULE, "newsapi_skipped",
                  "NewsAPI search skipped — no API key configured")
        return []

    log.debug(logger, MODULE, "newsapi_start", "NewsAPI search starting",
              query=query, max_results=max_results)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "pageSize": max_results,
                    "sortBy": "relevancy",
                    "language": "en",
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for article in data.get("articles", []):
                results.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                })

            log.debug(logger, MODULE, "newsapi_done", "NewsAPI search complete",
                      query=query, result_count=len(results))
            return results

        except Exception as e:
            log.warning(logger, MODULE, "newsapi_failed", "NewsAPI search failed",
                        error=str(e), error_type=type(e).__name__,
                        query=query)
            return []
