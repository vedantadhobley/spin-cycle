"""Evidence quality ranking for judge prompt capping.

When the research agent returns more evidence than the judge can handle
(MAX_JUDGE_EVIDENCE), we rank by quality signals rather than discovery order.
This prevents high-quality sources (government data portals, highly-rated
outlets) from being dropped just because they were discovered late.

Scoring uses only cached MBFC data + URL heuristics — zero network calls.
Political bias is deliberately NOT a scoring signal.
"""

import logging
from collections import Counter
from typing import Optional

from src.tools.source_ratings import extract_domain, get_source_rating_sync

logger = logging.getLogger(__name__)

# --- Source type scoring (0-30) ---
SOURCE_TYPE_SCORES = {
    "wikipedia": 30,
    "news_api": 15,
    "web": 10,
}

# --- MBFC factual reporting scoring (0-30) ---
FACTUAL_SCORES = {
    "very-high": 30,
    "high": 24,
    "mostly-factual": 16,
    "mixed": 8,
    "low": 4,
    "very-low": 2,
}
FACTUAL_UNRATED_GOV = 20  # .gov/.edu/.mil — trustworthy even without MBFC
FACTUAL_UNRATED = 4        # unknown domains — low default, must earn trust via MBFC

# --- MBFC credibility scoring (0-10) ---
CREDIBILITY_SCORES = {
    "high": 10,
    "medium": 5,
    "low": 2,
}
CREDIBILITY_UNRATED = 2

# --- Government/institutional TLD scoring (0-15) ---
GOV_TLD_SCORE = 15   # .gov, .mil
EDU_TLD_SCORE = 10   # .edu

# --- Content richness thresholds (0-15) ---
CONTENT_RICH = 2000   # 15 points
CONTENT_MEDIUM = 800  # 10 points
CONTENT_MINIMAL = 200 # 5 points


def _is_legiscan_url(url: str) -> bool:
    """Detect LegiScan evidence by URL since source_type is 'web'."""
    if not url:
        return False
    return "legiscan.com" in url.lower()


def _tld_score(domain: str) -> int:
    """Score based on domain TLD (.gov/.mil/.edu)."""
    if not domain:
        return 0
    if domain.endswith(".gov") or domain.endswith(".mil"):
        return GOV_TLD_SCORE
    # International government TLDs
    gov_intl = (".gov.uk", ".gov.au", ".gov.ca", ".gc.ca", ".gov.nz", ".gov.in")
    if any(domain.endswith(p) for p in gov_intl):
        return GOV_TLD_SCORE
    if domain.endswith(".edu"):
        return EDU_TLD_SCORE
    return 0


def _content_score(content: str) -> int:
    """Score based on content length (proxy for richness)."""
    if not content:
        return 0
    length = len(content)
    if length > CONTENT_RICH:
        return 15
    if length > CONTENT_MEDIUM:
        return 10
    if length > CONTENT_MINIMAL:
        return 5
    return 0


def score_evidence(ev: dict) -> tuple[float, dict]:
    """Score a single evidence item on quality signals.

    Returns (score, breakdown) where breakdown maps component names
    to their individual scores for debugging.
    """
    breakdown = {}
    url = ev.get("source_url", "") or ""
    source_type = ev.get("source_type", "web")
    content = ev.get("content", "") or ""

    # 1. Source type (0-30)
    if _is_legiscan_url(url):
        breakdown["source_type"] = 28
    else:
        breakdown["source_type"] = SOURCE_TYPE_SCORES.get(source_type, 10)

    # 2. MBFC factual reporting (0-30)
    domain = extract_domain(url) if url else ""
    rating = get_source_rating_sync(url) if url else None

    factual = rating.get("factual_reporting") if rating else None
    if factual and factual in FACTUAL_SCORES:
        breakdown["factual"] = FACTUAL_SCORES[factual]
    elif _tld_score(domain) > 0:
        # Government/edu domains are trustworthy even without MBFC rating
        breakdown["factual"] = FACTUAL_UNRATED_GOV
    else:
        # Unknown domains get low default — MBFC-rated sources should win
        breakdown["factual"] = FACTUAL_UNRATED

    # 3. Government/institutional TLD (0-15)
    breakdown["gov_tld"] = _tld_score(domain)

    # 4. Content richness (0-15)
    breakdown["content"] = _content_score(content)

    # 5. MBFC credibility (0-10)
    credibility = rating.get("credibility") if rating else None
    if credibility and credibility in CREDIBILITY_SCORES:
        breakdown["credibility"] = CREDIBILITY_SCORES[credibility]
    else:
        breakdown["credibility"] = CREDIBILITY_UNRATED

    total = sum(breakdown.values())
    return total, breakdown


def rank_and_select(
    evidence: list[dict],
    max_items: int = 20,
    max_per_domain: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Rank evidence by quality and select top items with diversity.

    1. Score every item
    2. Stable sort descending by score
    3. Walk sorted list, capping at max_per_domain per domain
    4. Take first max_items passing the domain cap

    Returns (selected, dropped) where dropped includes reason annotations.
    """
    if len(evidence) <= max_items:
        return evidence, []

    # Score all items
    scored = []
    for ev in evidence:
        total, breakdown = score_evidence(ev)
        scored.append((total, breakdown, ev))

    # Stable sort descending (preserves discovery order for equal scores)
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    dropped = []
    domain_counts: Counter = Counter()

    for total, breakdown, ev in scored:
        url = ev.get("source_url", "") or ""
        domain = extract_domain(url) if url else "unknown"

        if len(selected) >= max_items:
            dropped.append({"evidence": ev, "score": total, "reason": "cap"})
            continue

        if domain_counts[domain] >= max_per_domain:
            dropped.append({"evidence": ev, "score": total, "reason": "domain_cap"})
            continue

        domain_counts[domain] += 1
        ev["_rank_score"] = total
        selected.append(ev)

    return selected, dropped


def format_ranking_log(
    selected: list[dict],
    dropped: list[dict],
) -> dict:
    """Build a structured log dict for debug-level ranking output."""
    sel_entries = []
    for ev in selected:
        sel_entries.append({
            "url": ev.get("source_url", "N/A"),
            "score": ev.get("_rank_score", "?"),
        })

    drop_entries = []
    for d in dropped:
        ev = d["evidence"]
        drop_entries.append({
            "url": ev.get("source_url", "N/A"),
            "score": d["score"],
            "reason": d["reason"],
        })

    # Domain distribution in selected set
    domains: Counter = Counter()
    for ev in selected:
        url = ev.get("source_url", "") or ""
        domains[extract_domain(url) if url else "unknown"] += 1

    return {
        "selected": sel_entries,
        "dropped": drop_entries,
        "domain_distribution": dict(domains),
        "unique_domains": len(domains),
    }
