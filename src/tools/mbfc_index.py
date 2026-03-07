"""MBFC REST API index bootstrap and incremental refresh.

Downloads the full MBFC source index (~10,300 records) from the WordPress
REST API at wp-json/wp/v2/ratings, upserting into the source_ratings table.
Replaces unreliable slug-guessing with exact domain→record mapping.

Startup flow:
  1. is_bootstrap_needed() — check sentinel row for freshness (< 7 days)
  2. bootstrap_mbfc_index() — full or incremental paginated download
  3. All subsequent lookups are instant DB SELECTs (no network)

Ownership/bias_score/media_type are NOT in the API — those are lazy-scraped
from the known-correct mbfc_url when first needed (see source_ratings.py).
"""

import asyncio
from datetime import datetime, timezone, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.db.session import get_sync_session
from src.db.models import SourceRating
from src.tools.source_ratings import BIAS_MAP, extract_domain
from src.utils.logging import log, get_logger

MODULE = "mbfc_index"
logger = get_logger()

API_BASE = "https://mediabiasfactcheck.com/wp-json/wp/v2/ratings"
API_FIELDS = "domain,slug,source,bias,factual_reporting,credibility_rating,country,traffic_estimate,mbfc_url,questionable_reasoning"
PAGE_SIZE = 100
MAX_CONCURRENT_PAGES = 8
SENTINEL_DOMAIN = "__mbfc_index_meta__"
REFRESH_INTERVAL_DAYS = 7

# Map API factual_reporting values to our DB enum
FACTUAL_MAP = {
    "very_high": "very-high",
    "high": "high",
    "mostly_factual": "mostly-factual",
    "mixed": "mixed",
    "low": "low",
    "very_low": "very-low",
}


def _map_api_record(record: dict) -> dict | None:
    """Convert one API record to source_ratings column values.

    Returns None if the record has no usable domain.
    """
    raw_domain = (record.get("domain") or "").strip()
    if not raw_domain:
        return None

    domain = extract_domain(raw_domain)
    if not domain:
        return None

    # Bias: API returns list like ["left_center"], map to our enum
    bias = None
    bias_list = record.get("bias") or []
    if bias_list:
        raw_bias = str(bias_list[0]).replace("_", " ").lower().strip()
        bias = BIAS_MAP.get(raw_bias)

    # Factual reporting: API returns list like ["very_high"]
    factual = None
    factual_list = record.get("factual_reporting") or []
    if factual_list:
        raw_factual = str(factual_list[0]).strip().lower()
        factual = FACTUAL_MAP.get(raw_factual)

    # Credibility: API returns list like ["high_credibility"]
    credibility = None
    cred_list = record.get("credibility_rating") or []
    if cred_list:
        raw_cred = str(cred_list[0]).strip().lower()
        # "high_credibility" → "high", "medium_credibility" → "medium"
        first_word = raw_cred.split("_")[0].split(" ")[0]
        if first_word in ("high", "medium", "low"):
            credibility = first_word

    # Country: first element as-is
    country = None
    country_list = record.get("country") or []
    if country_list:
        country = str(country_list[0]).strip() or None

    # Traffic: first element as-is
    traffic = None
    traffic_list = record.get("traffic_estimate") or []
    if traffic_list:
        traffic = str(traffic_list[0]).strip() or None

    mbfc_url = (record.get("mbfc_url") or "").strip() or None

    # Store extra fields in raw_data for debugging
    raw_data = {}
    qr = record.get("questionable_reasoning")
    if qr:
        raw_data["questionable_reasoning"] = qr
    source_name = record.get("source")
    if source_name:
        raw_data["source_name"] = source_name

    return {
        "domain": domain,
        "bias": bias,
        "factual_reporting": factual,
        "credibility": credibility,
        "country": country,
        "traffic": traffic,
        "mbfc_url": mbfc_url,
        "raw_data": raw_data if raw_data else None,
        # ownership, bias_score, media_type left NULL — lazy scrape fills these
    }


def is_bootstrap_needed() -> bool:
    """Check if the MBFC index needs bootstrapping.

    Returns True if sentinel row is missing or older than REFRESH_INTERVAL_DAYS.
    """
    with get_sync_session() as session:
        stmt = select(SourceRating).where(SourceRating.domain == SENTINEL_DOMAIN)
        sentinel = session.execute(stmt).scalar_one_or_none()

        if not sentinel or not sentinel.scraped_at:
            return True

        age = datetime.now(timezone.utc) - sentinel.scraped_at.replace(tzinfo=timezone.utc)
        return age > timedelta(days=REFRESH_INTERVAL_DAYS)


def _get_last_bootstrap_time() -> datetime | None:
    """Get the timestamp of the last successful bootstrap."""
    with get_sync_session() as session:
        stmt = select(SourceRating.scraped_at).where(SourceRating.domain == SENTINEL_DOMAIN)
        result = session.execute(stmt).scalar_one_or_none()
        if result:
            return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
        return None


def _upsert_page(records: list[dict]) -> int:
    """Batch upsert mapped records into source_ratings. Returns count upserted."""
    if not records:
        return 0

    with get_sync_session() as session:
        for rec in records:
            stmt = insert(SourceRating).values(
                domain=rec["domain"],
                bias=rec["bias"],
                factual_reporting=rec["factual_reporting"],
                credibility=rec["credibility"],
                country=rec["country"],
                traffic=rec["traffic"],
                mbfc_url=rec["mbfc_url"],
                raw_data=rec["raw_data"],
                scraped_at=datetime.now(timezone.utc),
            ).on_conflict_do_update(
                index_elements=["domain"],
                set_={
                    "bias": rec["bias"],
                    "factual_reporting": rec["factual_reporting"],
                    "credibility": rec["credibility"],
                    "country": rec["country"],
                    "traffic": rec["traffic"],
                    "mbfc_url": rec["mbfc_url"],
                    "raw_data": rec["raw_data"],
                    "scraped_at": datetime.now(timezone.utc),
                    # ownership, bias_score, media_type are omitted — columns not
                    # in set_ are untouched by the UPDATE, preserving lazy-scraped values.
                },
            )
            session.execute(stmt)
        session.commit()
        return len(records)


def _upsert_sentinel() -> None:
    """Upsert the sentinel row to mark bootstrap timestamp."""
    with get_sync_session() as session:
        stmt = insert(SourceRating).values(
            domain=SENTINEL_DOMAIN,
            scraped_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            index_elements=["domain"],
            set_={
                "scraped_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        session.execute(stmt)
        session.commit()


async def bootstrap_mbfc_index(force: bool = False) -> int:
    """Download MBFC index from REST API and upsert into source_ratings.

    Args:
        force: If True, ignore sentinel freshness and re-download.

    Returns:
        Total number of records upserted.
    """
    if not force and not is_bootstrap_needed():
        log.info(logger, MODULE, "skip", "MBFC index is fresh, skipping bootstrap")
        return 0

    last_bootstrap = _get_last_bootstrap_time()
    incremental = last_bootstrap is not None and not force

    mode = "incremental" if incremental else "full"
    log.info(logger, MODULE, "start", f"MBFC index bootstrap ({mode})",
             last_bootstrap=str(last_bootstrap) if last_bootstrap else None)

    sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
    total_upserted = 0
    total_pages_fetched = 0

    async def _fetch_page(client: httpx.AsyncClient, page: int) -> list[dict] | None:
        """Fetch one page from the API. Returns mapped records or None on error."""
        params = {
            "per_page": PAGE_SIZE,
            "page": page,
            "_fields": API_FIELDS,
        }
        if incremental and last_bootstrap:
            params["modified_after"] = last_bootstrap.strftime("%Y-%m-%dT%H:%M:%S")

        async with sem:
            try:
                resp = await client.get(API_BASE, params=params)
                if resp.status_code == 400:
                    # WP returns 400 for page beyond total — signals end
                    return None
                resp.raise_for_status()
                raw_records = resp.json()
                if not raw_records:
                    return None
                mapped = []
                for rec in raw_records:
                    row = _map_api_record(rec)
                    if row:
                        mapped.append(row)
                return mapped
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    return None  # Past last page
                log.warning(logger, MODULE, "page_error",
                            "API page fetch failed", page=page, status=e.response.status_code)
                return None
            except Exception as e:
                log.warning(logger, MODULE, "page_error",
                            "API page fetch failed", page=page, error=str(e))
                return None

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "SpinCycle/1.0 (news verification)"},
    ) as client:
        # Fetch page 1 to get total pages from X-WP-TotalPages header
        params = {
            "per_page": PAGE_SIZE,
            "page": 1,
            "_fields": API_FIELDS,
        }
        if incremental and last_bootstrap:
            params["modified_after"] = last_bootstrap.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            resp = await client.get(API_BASE, params=params)
            resp.raise_for_status()
        except Exception as e:
            log.error(logger, MODULE, "first_page_failed",
                      "Failed to fetch first API page", error=str(e))
            return 0

        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        total_records_api = int(resp.headers.get("X-WP-Total", 0))
        log.info(logger, MODULE, "pages_discovered",
                 f"API reports {total_records_api} records across {total_pages} pages")

        # Process page 1
        raw_records = resp.json()
        if raw_records:
            mapped = [_map_api_record(r) for r in raw_records]
            mapped = [m for m in mapped if m]
            if mapped:
                count = _upsert_page(mapped)
                total_upserted += count
                total_pages_fetched += 1

        if total_pages <= 1:
            _upsert_sentinel()
            log.info(logger, MODULE, "done", "MBFC index bootstrap complete",
                     total_upserted=total_upserted, mode=mode)
            return total_upserted

        # Fetch remaining pages concurrently
        remaining = list(range(2, total_pages + 1))

        # Process in batches to log progress
        batch_size = 10
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            results = await asyncio.gather(*[_fetch_page(client, p) for p in batch])

            for page_records in results:
                if page_records:
                    count = _upsert_page(page_records)
                    total_upserted += count
                    total_pages_fetched += 1

            log.info(logger, MODULE, "progress",
                     f"Bootstrap progress: {total_pages_fetched}/{total_pages} pages, "
                     f"{total_upserted} records upserted")

    _upsert_sentinel()
    log.info(logger, MODULE, "done", "MBFC index bootstrap complete",
             total_upserted=total_upserted, total_pages=total_pages_fetched, mode=mode)
    return total_upserted
