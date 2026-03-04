"""MBFC (Media Bias/Fact Check) scraping, caching, and domain utilities.

Single responsibility: fetch, parse, cache, and serve MBFC ratings.
Government domain detection provides synthetic ratings for .gov/.mil domains.

Key functions:
  - get_source_rating(domain): async, scrapes MBFC if not cached
  - get_source_rating_sync(domain): sync, cache-only (no network)
  - await_ratings_parallel(domains): parallel MBFC await (no global timeout)
  - extract_domain(url): canonical domain normalization

Usage:
    from src.tools.source_ratings import get_source_rating

    rating = await get_source_rating("reuters.com")
    # Returns: {"bias": "center", "factual_reporting": "very-high", ...}
    # Or None if domain not found in MBFC
"""

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

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


def _domain_to_mbfc_slug(domain: str) -> str:
    """Convert domain to likely MBFC URL slug.
    
    Examples:
        reuters.com -> reuters
        foxnews.com -> fox-news
        nytimes.com -> new-york-times
    """
    # Remove TLD
    name = domain.rsplit(".", 1)[0]
    # Replace common patterns
    name = name.replace("news", "-news").replace("times", "-times")
    # Clean up
    name = re.sub(r"-+", "-", name).strip("-")
    return name


async def scrape_mbfc(domain: str) -> Optional[dict]:
    """Scrape MBFC for a domain's bias/factual ratings.
    
    Returns dict with keys: bias, factual_reporting, credibility, mbfc_url, raw_data
    Returns None if domain not found.
    """
    slug = _domain_to_mbfc_slug(domain)
    
    # Try a few URL patterns (MBFC isn't perfectly consistent)
    url_patterns = [
        f"{MBFC_BASE}/{slug}/",
        f"{MBFC_BASE}/{slug.replace('-', '')}/",
        f"{MBFC_BASE}/{domain.rsplit('.', 1)[0]}/",
    ]
    
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for url in url_patterns:
            try:
                resp = await client.get(url, headers={"User-Agent": "SpinCycle/1.0 (news verification)"})
                if resp.status_code == 200:
                    return _parse_mbfc_page(resp.text, url)
            except httpx.RequestError as e:
                log.warning(logger, MODULE, "scrape_request_failed",
                            "MBFC request failed", url=url, error=str(e))
                continue
        
        # Try search as fallback
        try:
            search_url = f"{MBFC_BASE}/?s={domain.rsplit('.', 1)[0]}"
            resp = await client.get(search_url)
            if resp.status_code == 200:
                # Parse search results, get first link
                soup = BeautifulSoup(resp.text, "html.parser")
                result = soup.select_one("article.post h2 a")
                if result and result.get("href"):
                    page_resp = await client.get(result["href"])
                    if page_resp.status_code == 200:
                        return _parse_mbfc_page(page_resp.text, result["href"])
        except httpx.RequestError as e:
            log.warning(logger, MODULE, "scrape_search_failed",
                        "MBFC search fallback failed", domain=domain, error=str(e))

    log.warning(logger, MODULE, "scrape_all_failed",
                "MBFC: all lookup patterns failed", domain=domain)
    return None


def _parse_mbfc_page(html: str, url: str) -> Optional[dict]:
    """Parse an MBFC source page for ratings.
    
    MBFC pages have a consistent structure with fields on consecutive lines:
        Bias Rating:
        LEAST BIASED (-0.5)
        Factual Reporting:
        VERY
        HIGH (0.0)
        Country:
        United Kingdom
        ...
    """
    soup = BeautifulSoup(html, "html.parser")
    
    result = {
        "bias": None,
        "bias_score": None,
        "factual_reporting": None,
        "credibility": None,
        "country": None,
        "media_type": None,
        "ownership": None,
        "traffic": None,
        "mbfc_url": url,
        "raw_data": {},
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
    
    # Extract bias rating and score
    # Format: "LEAST BIASED (-0.5)" or "LEFT-CENTER (+1.5)"
    bias_value = get_value_after("Bias Rating:")
    if bias_value:
        result["raw_data"]["bias_text"] = bias_value
        bias_lower = bias_value.lower()
        
        # Extract numeric score if present: (-0.5) or (+3.0)
        score_match = re.search(r'\(([+-]?\d+\.?\d*)\)', bias_value)
        if score_match:
            try:
                result["bias_score"] = float(score_match.group(1))
            except ValueError:
                pass
        
        # Map to enum
        for key, value in BIAS_MAP.items():
            if key in bias_lower:
                result["bias"] = value
                break
    
    # Extract factual reporting
    # Format: "VERY HIGH (0.0)" - may span multiple lines like "VERY" + "HIGH (0.0)"
    factual_value = get_value_after("Factual Reporting:")
    if factual_value:
        result["raw_data"]["factual_text"] = factual_value
        factual_lower = factual_value.lower()
        
        # Check for each factual level
        if "very high" in factual_lower:
            result["factual_reporting"] = "very-high"
        elif "very low" in factual_lower:
            result["factual_reporting"] = "very-low"
        elif "mostly factual" in factual_lower:
            result["factual_reporting"] = "mostly-factual"
        elif "high" in factual_lower:
            result["factual_reporting"] = "high"
        elif "mixed" in factual_lower:
            result["factual_reporting"] = "mixed"
        elif "low" in factual_lower:
            result["factual_reporting"] = "low"
    
    # Extract credibility
    # Format: "HIGH CREDIBILITY"
    cred_value = get_value_after("MBFC Credibility Rating:")
    if cred_value:
        result["raw_data"]["credibility_text"] = cred_value
        cred_lower = cred_value.lower()
        if "high" in cred_lower:
            result["credibility"] = "high"
        elif "medium" in cred_lower:
            result["credibility"] = "medium"
        elif "low" in cred_lower:
            result["credibility"] = "low"
    
    # Extract country
    country_value = get_value_after("Country:", max_lines=1)
    if country_value and not country_value.endswith(":"):
        result["country"] = country_value
        result["raw_data"]["country"] = country_value
    
    # Extract media type
    media_value = get_value_after("Media Type:", max_lines=1)
    if media_value and not media_value.endswith(":"):
        result["media_type"] = media_value
        result["raw_data"]["media_type"] = media_value
    
    # Extract traffic
    traffic_value = get_value_after("Traffic/Popularity:", max_lines=1)
    if traffic_value and not traffic_value.endswith(":"):
        result["traffic"] = traffic_value
        result["raw_data"]["traffic"] = traffic_value
    
    # Extract ownership/funding (more complex - may span multiple lines)
    ownership_value = get_value_after("Funded by / Ownership", max_lines=5)
    if ownership_value:
        # Clean up and truncate to reasonable length
        ownership_clean = ownership_value[:200]
        result["ownership"] = ownership_clean
        result["raw_data"]["ownership"] = ownership_value
    
    # If we got at least bias or factual, consider it a success
    if result["bias"] or result["factual_reporting"]:
        return result
    
    return None


def _is_stale(scraped_at: datetime) -> bool:
    """Check if cached rating is older than TTL."""
    if scraped_at is None:
        return True
    now = datetime.now(timezone.utc)
    # Ensure scraped_at is timezone-aware
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=timezone.utc)
    return (now - scraped_at) > timedelta(days=CACHE_TTL_DAYS)


async def get_source_rating(url_or_domain: str, force_refresh: bool = False) -> Optional[dict]:
    """Get source rating, checking cache first, scraping if needed.
    
    Args:
        url_or_domain: URL or domain to look up
        force_refresh: If True, skip cache and scrape fresh
        
    Returns:
        Dict with bias, factual_reporting, credibility, or None if not found
    """
    domain = extract_domain(url_or_domain)
    
    # Always check if this is a government domain (for warning flag)
    gov_info = _get_government_info(domain)
    
    with get_sync_session() as session:
        # Check cache first (unless force refresh)
        if not force_refresh:
            stmt = select(SourceRating).where(SourceRating.domain == domain)
            cached = session.execute(stmt).scalar_one_or_none()
            
            if cached and not _is_stale(cached.scraped_at):
                log.debug(logger, MODULE, "cache_hit",
                         "MBFC cache hit", domain=domain)
                result = {
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
                # Add government info if applicable
                if gov_info:
                    result.update(gov_info)
                return result
    
    # Cache miss or stale — scrape MBFC
    log.info(logger, MODULE, "scrape_start", "Scraping MBFC", domain=domain)
    scraped = await scrape_mbfc(domain)
    
    if scraped:
        # Validate: if domain is .gov/.mil but MBFC returned non-government media type,
        # the MBFC result is likely a similarly-named publication (e.g., "Defense News" 
        # instead of defense.gov). Reject it and fall through to government fallback.
        is_gov_domain = domain.endswith(".gov") or domain.endswith(".mil")
        mbfc_is_gov = scraped.get("media_type", "").lower() == "government"
        
        if is_gov_domain and not mbfc_is_gov:
            log.warning(logger, MODULE, "gov_mismatch",
                        "MBFC returned non-government media type for gov domain — rejecting",
                        domain=domain, media_type=scraped.get("media_type"))
            # Fall through to government fallback below
        else:
            # Valid MBFC result — upsert into cache
            with get_sync_session() as session:
                stmt = insert(SourceRating).values(
                    domain=domain,
                    bias=scraped["bias"],
                    bias_score=scraped.get("bias_score"),
                    factual_reporting=scraped["factual_reporting"],
                    credibility=scraped["credibility"],
                    country=scraped.get("country"),
                    media_type=scraped.get("media_type"),
                    ownership=scraped.get("ownership"),
                    traffic=scraped.get("traffic"),
                    mbfc_url=scraped["mbfc_url"],
                    raw_data=scraped["raw_data"],
                    scraped_at=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["domain"],
                    set_={
                        "bias": scraped["bias"],
                        "bias_score": scraped.get("bias_score"),
                        "factual_reporting": scraped["factual_reporting"],
                        "credibility": scraped["credibility"],
                        "country": scraped.get("country"),
                        "media_type": scraped.get("media_type"),
                        "ownership": scraped.get("ownership"),
                        "traffic": scraped.get("traffic"),
                        "mbfc_url": scraped["mbfc_url"],
                        "raw_data": scraped["raw_data"],
                        "scraped_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                session.execute(stmt)
                session.commit()
            
            result = {
                "domain": domain,
                "bias": scraped["bias"],
                "bias_score": scraped.get("bias_score"),
                "factual_reporting": scraped["factual_reporting"],
                "credibility": scraped["credibility"],
                "country": scraped.get("country"),
                "media_type": scraped.get("media_type"),
                "ownership": scraped.get("ownership"),
                "traffic": scraped.get("traffic"),
                "mbfc_url": scraped["mbfc_url"],
            }
            # Add government info if applicable (for INTERESTED PARTY warning)
            if gov_info:
                result.update(gov_info)
            return result
    
    # Not found in MBFC — check if it's a government source (return gov-only rating)
    if gov_info:
        log.info(logger, MODULE, "gov_source",
                 "Domain is a government source (not in MBFC)", domain=domain)
        return _get_government_rating(domain)
    
    log.debug(logger, MODULE, "not_found", "Domain not found in MBFC", domain=domain)
    return None


async def await_ratings_parallel(
    domains: list[str],
    max_concurrent: int = 8,
) -> dict[str, Optional[dict]]:
    """Parallel MBFC lookups — waits for ALL domains to complete.

    Each domain bounded by 15s httpx timeout in scrape_mbfc().
    No global timeout — we wait for every domain rather than
    cancelling in-flight scrapes.

    Returns dict mapping domain -> rating dict (or None).
    """
    import asyncio

    unique_domains = list(set(domains))
    results: dict[str, Optional[dict]] = {}
    sem = asyncio.Semaphore(max_concurrent)

    async def _lookup(domain: str) -> None:
        async with sem:
            try:
                rating = await get_source_rating(domain)
                results[domain] = rating
            except Exception as e:
                log.warning(logger, MODULE, "lookup_failed",
                            "MBFC lookup failed", domain=domain, error=str(e))
                results[domain] = None

    await asyncio.gather(*[_lookup(d) for d in unique_domains])

    return results


def get_source_rating_sync(url_or_domain: str) -> Optional[dict]:
    """Synchronous version of get_source_rating (cache only, no scrape).
    
    For use in evidence formatting where we can't await.
    Falls back to government detection if not in cache.
    """
    domain = extract_domain(url_or_domain)
    
    with get_sync_session() as session:
        stmt = select(SourceRating).where(SourceRating.domain == domain)
        cached = session.execute(stmt).scalar_one_or_none()
        
        if cached:
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
    
    # Not in cache — check if it's a government source
    return _get_government_rating(domain)




