"""News API search tool for evidence gathering."""

import os
import httpx


NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")


async def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Search NewsAPI for recent articles related to the query.

    Returns a list of {title, description, url, source, published_at} dicts.
    """
    if not NEWSAPI_KEY:
        return []  # No API key configured

    async with httpx.AsyncClient() as client:
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

        return results
