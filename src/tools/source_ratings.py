"""MBFC (Media Bias/Fact Check) ratings: DB-backed lookup and lazy ownership scrape.

After bootstrap (see mbfc_index.py), every MBFC domain is in the source_ratings
table with bias, factual_reporting, credibility, country, traffic, and mbfc_url.
Ownership/bias_score/media_type are NOT in the API — they're lazy-scraped from
the known-correct mbfc_url on first access.

Key functions:
  - get_source_rating(domain): async, DB lookup + lazy ownership scrape
  - get_source_rating_sync(domain): sync, cache-only (no network)
  - await_ratings_parallel(domains): parallel DB lookups + batch ownership scrape
  - extract_domain(url): canonical domain normalization

Usage:
    from src.tools.source_ratings import get_source_rating

    rating = await get_source_rating("reuters.com")
    # Returns: {"bias": "center", "factual_reporting": "very-high", ...}
    # Or None if domain not in MBFC index
"""

import re
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select

from src.db.session import get_sync_session
from src.db.models import SourceRating
from src.utils.logging import log, get_logger

MODULE = "source_ratings"
logger = get_logger()

# How long before we consider cached data stale
CACHE_TTL_DAYS = 30

# MBFC base URL
MBFC_BASE = "https://mediabiasfactcheck.com"

# Map MBFC text to our enum values
BIAS_MAP = {
    "left": "left",
    "left-center": "left-center",
    "least biased": "center",
    "center": "center",
    "right-center": "right-center",
    "right": "right",
    "extreme left": "extreme-left",
    "extreme right": "extreme-right",
    "satire": "satire",
    "conspiracy-pseudoscience": "conspiracy-pseudoscience",
    "questionable source": "conspiracy-pseudoscience",
    "questionable sources": "conspiracy-pseudoscience",
    "pro-science": "center",  # MBFC category for science-focused outlets
}

def extract_domain(url_or_domain: str) -> str:
    """Normalize input to a clean domain (no www, no path)."""
    if not url_or_domain.startswith(("http://", "https://")):
        url_or_domain = f"https://{url_or_domain}"
    parsed = urlparse(url_or_domain)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


# Known government domain patterns and their descriptions
GOVERNMENT_DOMAINS = {
    # US Federal
    "whitehouse.gov": "White House",
    "justice.gov": "Department of Justice",
    "defense.gov": "Department of Defense",
    "state.gov": "Department of State",
    "fbi.gov": "FBI",
    "cia.gov": "CIA",
    "dhs.gov": "Department of Homeland Security",
    "cdc.gov": "Centers for Disease Control",
    "nih.gov": "National Institutes of Health",
    "fda.gov": "Food and Drug Administration",
    "epa.gov": "Environmental Protection Agency",
    "treasury.gov": "Department of Treasury",
    "commerce.gov": "Department of Commerce",
    "congress.gov": "US Congress",
    "senate.gov": "US Senate",
    "house.gov": "US House of Representatives",
    "supremecourt.gov": "Supreme Court",
    # Military
    "army.mil": "US Army",
    "navy.mil": "US Navy",
    "af.mil": "US Air Force",
    "marines.mil": "US Marine Corps",
    "spaceforce.mil": "US Space Force",
}


def _is_government_domain(domain: str) -> bool:
    """Check if domain is a government/military TLD."""
    # Direct .gov/.mil
    if domain.endswith(".gov") or domain.endswith(".mil"):
        return True
    # International government TLDs
    gov_patterns = [".gov.uk", ".gov.au", ".gov.ca", ".gc.ca", ".gov.nz", ".gov.in"]
    return any(domain.endswith(p) for p in gov_patterns)


def _get_government_info(domain: str) -> Optional[dict]:
    """Get government metadata for a domain.

    Returns agency name and country for government domains.
    Always returns is_government=True for gov domains.
    """
    if not _is_government_domain(domain):
        return None

    # Check specific known domains first for friendly names
    if domain in GOVERNMENT_DOMAINS:
        agency = GOVERNMENT_DOMAINS[domain]
        return {
            "is_government": True,
            "agency_name": agency,
            "country": "USA",
        }

    # Determine country by TLD pattern
    country = "USA"  # Default for .gov/.mil
    if ".gov.uk" in domain:
        country = "UK"
    elif ".gov.au" in domain:
        country = "Australia"
    elif ".gov.ca" in domain or ".gc.ca" in domain:
        country = "Canada"
    elif ".gov.nz" in domain:
        country = "New Zealand"
    elif ".gov.in" in domain:
        country = "India"

    # Extract agency name from subdomain
    agency_name = domain.split(".")[0].upper()

    return {
        "is_government": True,
        "agency_name": agency_name,
        "country": country,
    }


def _get_government_rating(domain: str) -> Optional[dict]:
    """Return synthetic rating for government domains not in MBFC.

    Government sources are treated as interested parties - they publish
    official statements but are not independent journalists.
    """
    gov_info = _get_government_info(domain)
    if not gov_info:
        return None

    return {
        "domain": domain,
        "bias": None,  # Government sources have institutional bias, not left/right
        "bias_score": None,
        "factual_reporting": None,  # Varies by administration/context
        "credibility": None,
        "country": gov_info["country"],
        "media_type": "Government",
        "ownership": f"Official government website ({domain})",
        "traffic": None,
        "mbfc_url": None,
        "is_government": True,
        "agency_name": gov_info["agency_name"],
    }


def _parse_mbfc_page(html: str, url: str) -> Optional[dict]:
    """Parse an MBFC source page for ownership, bias_score, and media_type.

    Used by _lazy_scrape_ownership() to fill fields not available in the API.
    """
    soup = BeautifulSoup(html, "html.parser")

    result = {
        "bias_score": None,
        "media_type": None,
        "ownership": None,
    }

    # Look for the ratings in the page content
    content = soup.select_one("div.entry-content, article.post")
    if not content:
        return None

    # Get text as lines for structured parsing
    lines = [line.strip() for line in content.get_text(separator="\n").split("\n") if line.strip()]

    # Helper to find value after a label line
    def get_value_after(label: str, max_lines: int = 3) -> Optional[str]:
        """Find the value on lines following a label."""
        for i, line in enumerate(lines):
            if label.lower() in line.lower():
                # Collect next few lines until we hit another label or empty
                values = []
                for j in range(1, max_lines + 1):
                    if i + j >= len(lines):
                        break
                    next_line = lines[i + j]
                    # Stop if we hit another field label
                    if any(lbl in next_line.lower() for lbl in [
                        "bias rating", "factual reporting", "country:", "media type:",
                        "traffic", "credibility", "history", "founded", "funded by"
                    ]) and next_line.endswith(":"):
                        break
                    values.append(next_line)
                return " ".join(values).strip() if values else None
        return None

    # Extract bias score from bias rating line
    bias_value = get_value_after("Bias Rating:")
    if bias_value:
        score_match = re.search(r'\(([+-]?\d+\.?\d*)\)', bias_value)
        if score_match:
            try:
                result["bias_score"] = float(score_match.group(1))
            except ValueError:
                pass

    # Extract media type
    media_value = get_value_after("Media Type:", max_lines=1)
    if media_value and not media_value.endswith(":"):
        result["media_type"] = media_value

    # Extract ownership/funding (more complex - may span multiple lines)
    ownership_value = get_value_after("Funded by / Ownership", max_lines=5)
    if ownership_value:
        result["ownership"] = ownership_value[:200]

    # Return if we got at least one useful field
    if any(v is not None for v in result.values()):
        return result

    return None


def _row_to_dict(cached: SourceRating) -> dict:
    """Convert a SourceRating ORM row to a plain dict."""
    return {
        "domain": cached.domain,
        "bias": cached.bias,
        "bias_score": cached.bias_score,
        "factual_reporting": cached.factual_reporting,
        "credibility": cached.credibility,
        "country": cached.country,
        "media_type": cached.media_type,
        "ownership": cached.ownership,
        "traffic": cached.traffic,
        "mbfc_url": cached.mbfc_url,
    }


def _is_stale(scraped_at: datetime) -> bool:
    """Check if cached rating is older than TTL."""
    if scraped_at is None:
        return True
    now = datetime.now(timezone.utc)
    # Ensure scraped_at is timezone-aware
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=timezone.utc)
    return (now - scraped_at) > timedelta(days=CACHE_TTL_DAYS)


async def _lazy_scrape_ownership(domain: str) -> None:
    """Lazy-scrape ownership/bias_score/media_type from the known-correct mbfc_url.

    Only scrapes if ownership is NULL and mbfc_url is set.
    Single HTTP request to a known-correct URL (no slug guessing).
    """
    with get_sync_session() as session:
        stmt = select(SourceRating).where(SourceRating.domain == domain)
        row = session.execute(stmt).scalar_one_or_none()
        if not row:
            return
        if row.ownership:
            return  # Already populated
        mbfc_url = row.mbfc_url
        if not mbfc_url:
            return

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "SpinCycle/1.0 (news verification)"},
        ) as client:
            resp = await client.get(mbfc_url)
            if resp.status_code != 200:
                log.warning(logger, MODULE, "lazy_scrape_failed",
                            "Ownership scrape HTTP error",
                            domain=domain, status=resp.status_code)
                return
            parsed = _parse_mbfc_page(resp.text, mbfc_url)
    except Exception as e:
        log.warning(logger, MODULE, "lazy_scrape_error",
                    "Ownership scrape failed", domain=domain, error=str(e))
        return

    if not parsed:
        return

    with get_sync_session() as session:
        stmt = (
            SourceRating.__table__.update()
            .where(SourceRating.domain == domain)
            .values(
                ownership=parsed.get("ownership"),
                bias_score=parsed.get("bias_score"),
                media_type=parsed.get("media_type"),
                updated_at=datetime.now(timezone.utc),
            )
        )
        session.execute(stmt)
        session.commit()

    log.debug(logger, MODULE, "lazy_scrape_done",
              "Ownership scraped", domain=domain,
              has_ownership=bool(parsed.get("ownership")))


async def get_source_rating(url_or_domain: str, force_refresh: bool = False) -> Optional[dict]:
    """Get source rating from the pre-populated index (DB lookup).

    After bootstrap, this is an instant DB SELECT. If ownership is missing
    and mbfc_url is available, triggers a lazy scrape for that single domain.

    Args:
        url_or_domain: URL or domain to look up
        force_refresh: If True, force lazy ownership re-scrape

    Returns:
        Dict with bias, factual_reporting, credibility, or None if not found
    """
    domain = extract_domain(url_or_domain)
    gov_info = _get_government_info(domain)

    with get_sync_session() as session:
        stmt = select(SourceRating).where(SourceRating.domain == domain)
        cached = session.execute(stmt).scalar_one_or_none()

        if cached and not _is_stale(cached.scraped_at):
            result = _row_to_dict(cached)
            needs_ownership = (
                result.get("mbfc_url")
                and not result.get("ownership")
            )

            if needs_ownership or force_refresh:
                # Release DB session before doing network I/O
                pass
            else:
                if gov_info:
                    result.update(gov_info)
                return result

    # Lazy-fill ownership if needed
    if cached and not _is_stale(cached.scraped_at):
        await _lazy_scrape_ownership(domain)
        # Re-read updated row
        with get_sync_session() as session:
            stmt = select(SourceRating).where(SourceRating.domain == domain)
            cached = session.execute(stmt).scalar_one_or_none()
            if cached:
                result = _row_to_dict(cached)
                if gov_info:
                    result.update(gov_info)
                return result

    # Not in DB or stale — check government fallback
    if gov_info:
        log.info(logger, MODULE, "gov_source",
                 "Domain is a government source (not in MBFC index)", domain=domain)
        return _get_government_rating(domain)

    log.debug(logger, MODULE, "not_found", "Domain not in MBFC index", domain=domain)
    return None


async def await_ratings_parallel(
    domains: list[str],
    max_concurrent: int = 8,
) -> dict[str, Optional[dict]]:
    """Parallel DB lookups + batch lazy ownership scrape.

    After bootstrap, the DB lookups are instant. The only network I/O
    is lazy ownership scrapes for domains that need them.

    Returns dict mapping domain -> rating dict (or None).
    """
    import asyncio

    unique_domains = list(set(domains))
    results: dict[str, Optional[dict]] = {}

    # Phase 1: Fast DB lookups (all sync, instant)
    for domain in unique_domains:
        results[domain] = get_source_rating_sync(domain)

    # Phase 2: Batch lazy-scrape ownership for domains that need it
    needs_ownership = [
        d for d in unique_domains
        if results[d]
        and results[d].get("mbfc_url")
        and not results[d].get("ownership")
    ]

    if needs_ownership:
        sem = asyncio.Semaphore(4)

        async def _scrape(d: str) -> None:
            async with sem:
                try:
                    await _lazy_scrape_ownership(d)
                except Exception as e:
                    log.warning(logger, MODULE, "ownership_scrape_failed",
                                "Lazy ownership scrape failed", domain=d, error=str(e))

        await asyncio.gather(*[_scrape(d) for d in needs_ownership])

        # Re-read updated rows
        for d in needs_ownership:
            results[d] = get_source_rating_sync(d)

        log.info(logger, MODULE, "ownership_batch_done",
                 "Batch ownership scrape complete",
                 scraped_count=len(needs_ownership))

    return results


def get_source_rating_sync(url_or_domain: str) -> Optional[dict]:
    """Synchronous version of get_source_rating (cache only, no network).

    For use in evidence formatting where we can't await.
    Falls back to government detection if not in cache.
    """
    domain = extract_domain(url_or_domain)

    with get_sync_session() as session:
        stmt = select(SourceRating).where(SourceRating.domain == domain)
        cached = session.execute(stmt).scalar_one_or_none()

        if cached:
            return _row_to_dict(cached)

    # Not in cache — check if it's a government source
    return _get_government_rating(domain)
