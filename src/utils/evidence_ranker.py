"""Evidence and source quality scoring.

Two use cases:
  1. Seed ranking (research phase): score_url() and tier_label() rank
     ~80-100 seed URLs by quality so the agent sees curated top-30 seeds.
  2. Judge prompt capping: score_evidence() and rank_and_select() rank
     evidence by quality when the research agent returns more than
     MAX_JUDGE_EVIDENCE items.

Scoring uses only cached MBFC data + URL heuristics — zero network calls.
Political bias is deliberately NOT a scoring signal.
"""

from collections import Counter

from src.tools.source_ratings import extract_domain, get_source_rating_sync

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


def score_url(url: str) -> tuple[int, dict]:
    """Score a URL on source quality signals (no content needed).

    Uses MBFC cache + domain heuristics. Zero network calls.
    Returns (score, breakdown) — same shape as score_evidence
    but without content_richness and source_type components.

    Max score: 55 (factual=30 + gov_tld=15 + credibility=10)
    """
    domain = extract_domain(url) if url else ""
    rating = get_source_rating_sync(url) if url else None
    breakdown = {}

    # MBFC factual (0-30)
    factual = rating.get("factual_reporting") if rating else None
    if factual and factual in FACTUAL_SCORES:
        breakdown["factual"] = FACTUAL_SCORES[factual]
    elif _tld_score(domain) > 0:
        breakdown["factual"] = FACTUAL_UNRATED_GOV
    else:
        breakdown["factual"] = FACTUAL_UNRATED

    # Gov/edu TLD (0-15)
    breakdown["gov_tld"] = _tld_score(domain)

    # MBFC credibility (0-10)
    credibility = rating.get("credibility") if rating else None
    if credibility and credibility in CREDIBILITY_SCORES:
        breakdown["credibility"] = CREDIBILITY_SCORES[credibility]
    else:
        breakdown["credibility"] = CREDIBILITY_UNRATED

    return sum(breakdown.values()), breakdown


def tier_label(url: str) -> str:
    """Human-readable tier label for a URL. Used in seed annotations.

    TIER 1: gov/edu/mil TLD, or MBFC very-high/high factual
    TIER 2: MBFC mostly-factual
    Empty string for unknown/low-quality sources.
    """
    _, breakdown = score_url(url)
    tld = breakdown.get("gov_tld", 0)
    factual = breakdown.get("factual", 0)

    if tld >= GOV_TLD_SCORE:
        return "TIER 1 (government)"
    if tld >= EDU_TLD_SCORE:
        return "TIER 1 (academic)"
    if factual >= 30:  # very-high
        return "TIER 1 (very high factual)"
    if factual >= 24:  # high
        return "TIER 2 (high factual)"
    if factual >= 16:  # mostly-factual
        return "TIER 2 (mostly factual)"
    if factual >= FACTUAL_UNRATED_GOV:  # unrated but gov-like
        return "TIER 1 (institutional)"
    return ""


def score_evidence(ev: dict) -> tuple[float, dict]:
    """Score a single evidence item on quality signals.

    Returns (score, breakdown) where breakdown maps component names
    to their individual scores for debugging.
    """
    url = ev.get("source_url", "") or ""
    source_type = ev.get("source_type", "web")
    content = ev.get("content", "") or ""

    # URL-based components (factual, gov_tld, credibility)
    total, breakdown = score_url(url)

    # Source type (0-30)
    if _is_legiscan_url(url):
        breakdown["source_type"] = 28
    else:
        breakdown["source_type"] = SOURCE_TYPE_SCORES.get(source_type, 10)

    # Content richness (0-15)
    breakdown["content"] = _content_score(content)

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
