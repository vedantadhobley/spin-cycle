"""Page fetcher tool for reading web page content.

Given a URL, fetches the page and extracts readable text content.
This lets the research agent actually READ sources it finds via search,
rather than relying only on short search snippets.

No API key needed — this uses direct HTTP requests + HTML parsing.
Requires: beautifulsoup4, lxml
"""

import re
import httpx
from langchain_core.tools import tool

from src.tools.source_filter import is_blocked
from src.utils.logging import log, get_logger

MODULE = "tools"
logger = get_logger()

# Max content length to return (characters). Pages can be huge — we truncate
# to avoid blowing up the LLM context window.
MAX_CONTENT_LENGTH = 8000

# Timeout for fetching pages (seconds)
FETCH_TIMEOUT = 15

# User agent — identify ourselves honestly
USER_AGENT = "SpinCycle/0.1 (claim verification research tool; +https://github.com/vedantadhobley/spin-cycle)"


def _extract_text(html: str) -> str:
    """Extract readable text from HTML, stripping boilerplate.

    Uses BeautifulSoup to parse HTML and extract the main text content,
    removing navigation, scripts, styles, and other non-content elements.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove elements that are never useful content
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "iframe", "noscript", "svg", "form",
                     "button", "input", "select", "textarea"]):
        tag.decompose()

    # Try to find the main content area first
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", {"id": re.compile(r"content|article|main", re.I)})
        or soup.find("div", {"class": re.compile(r"content|article|main|post|entry", re.I)})
    )

    target = main if main else soup.body if soup.body else soup

    # Get text, collapsing whitespace
    text = target.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of whitespace on each line
    lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    return text


async def fetch_page(url: str) -> dict:
    """Fetch a web page and extract its text content.

    Returns a dict with:
      - url: the URL fetched
      - title: page title (from <title> tag)
      - content: extracted text content (truncated to MAX_CONTENT_LENGTH)
      - error: error message if fetch failed (None on success)
    """
    log.debug(logger, MODULE, "fetch_start", "Fetching page",
              url=url)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=FETCH_TIMEOUT,
    ) as client:
        try:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return {
                    "url": url,
                    "title": "",
                    "content": f"Non-HTML content type: {content_type}",
                    "error": "not_html",
                }

            html = resp.text
            text = _extract_text(html)

            # Extract title
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Truncate
            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "\n\n[... content truncated ...]"

            log.debug(logger, MODULE, "fetch_done", "Page fetched successfully",
                      url=url, title=title,
                      content_length=len(text))

            return {
                "url": url,
                "title": title,
                "content": text,
                "error": None,
            }

        except httpx.HTTPStatusError as e:
            log.warning(logger, MODULE, "fetch_http_error", "Page fetch HTTP error",
                        url=url, status_code=e.response.status_code)
            return {
                "url": url,
                "title": "",
                "content": "",
                "error": f"HTTP {e.response.status_code}",
            }

        except Exception as e:
            log.warning(logger, MODULE, "fetch_failed", "Page fetch failed",
                        url=url, error=str(e),
                        error_type=type(e).__name__)
            return {
                "url": url,
                "title": "",
                "content": "",
                "error": str(e),
            }


def get_page_fetcher_tool():
    """Get a LangChain tool that fetches and reads web page content.

    Returns a @tool-decorated async function compatible with LangGraph agents.
    """

    @tool
    async def fetch_page_content(url: str) -> str:
        """Fetch and read the full text content of a web page.

        Use this when search results give you a promising URL but the
        snippet isn't detailed enough. This tool fetches the page and
        extracts the readable text content, giving you the full article
        or document to work with.

        Input should be a complete URL starting with http:// or https://.
        """
        if not url.startswith(("http://", "https://")):
            return "Invalid URL. Must start with http:// or https://"

        if is_blocked(url):
            return f"Blocked source: {url} is not a citable source (social media, forum, or content farm). Find a reputable publication instead."

        try:
            result = await fetch_page(url)
        except Exception as e:
            log.warning(logger, MODULE, "fetch_tool_failed",
                        "Page fetch tool failed",
                        error=str(e), error_type=type(e).__name__,
                        url=url)
            return f"Failed to fetch page: {str(e)}"

        if result["error"]:
            return f"Failed to fetch {url}: {result['error']}"

        if not result["content"]:
            return f"Page at {url} returned no readable content."

        header = f"Page: {result['title']}\nURL: {result['url']}\n\n"
        return header + result["content"]

    return fetch_page_content
