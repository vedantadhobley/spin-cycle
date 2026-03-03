"""Tests for evidence quality ranking."""

import pytest
from unittest.mock import patch

from src.utils.evidence_ranker import (
    score_evidence,
    rank_and_select,
    format_ranking_log,
    _is_legiscan_url,
    _tld_score,
    _content_score,
)


def _make_ev(url="https://example.com/article", source_type="web", content="x" * 300):
    """Helper to build a minimal evidence dict."""
    return {
        "source_url": url,
        "source_type": source_type,
        "content": content,
        "title": "Test Article",
    }


# ---------- Mock ratings lookup ----------

MOCK_RATINGS = {
    "reuters.com": {
        "factual_reporting": "very-high",
        "credibility": "high",
        "bias": "center",
    },
    "foxnews.com": {
        "factual_reporting": "mixed",
        "credibility": "medium",
        "bias": "right",
    },
    "eia.gov": {
        "domain": "eia.gov",
        "is_government": True,
        "factual_reporting": None,
        "credibility": None,
        "bias": None,
    },
    "example.edu": {
        "factual_reporting": None,
        "credibility": None,
        "bias": None,
    },
}


def _mock_rating(url_or_domain):
    """Return mock MBFC rating or government fallback."""
    from src.tools.source_ratings import extract_domain
    domain = extract_domain(url_or_domain)
    return MOCK_RATINGS.get(domain)


@pytest.fixture(autouse=True)
def patch_ratings():
    with patch("src.utils.evidence_ranker.get_source_rating_sync", side_effect=_mock_rating):
        yield


# ---------- Unit tests: scoring helpers ----------

class TestHelpers:
    def test_legiscan_url_detected(self):
        assert _is_legiscan_url("https://legiscan.com/US/bill/HR1234")
        assert not _is_legiscan_url("https://reuters.com/article")
        assert not _is_legiscan_url("")
        assert not _is_legiscan_url(None)

    def test_tld_gov(self):
        assert _tld_score("eia.gov") == 15
        assert _tld_score("army.mil") == 15
        assert _tld_score("data.gov.uk") == 15

    def test_tld_edu(self):
        assert _tld_score("mit.edu") == 10

    def test_tld_normal(self):
        assert _tld_score("reuters.com") == 0
        assert _tld_score("") == 0

    def test_content_score_tiers(self):
        assert _content_score("x" * 3000) == 15
        assert _content_score("x" * 1000) == 10
        assert _content_score("x" * 300) == 5
        assert _content_score("x" * 50) == 0
        assert _content_score("") == 0
        assert _content_score(None) == 0


# ---------- Unit tests: score_evidence ----------

class TestScoreEvidence:
    def test_wikipedia_scores_higher_than_unrated_web(self):
        wiki = _make_ev(url="https://en.wikipedia.org/wiki/Test", source_type="wikipedia")
        web = _make_ev(url="https://unknown-blog.com/post", source_type="web")
        wiki_score, _ = score_evidence(wiki)
        web_score, _ = score_evidence(web)
        assert wiki_score > web_score

    def test_reuters_high_score(self):
        ev = _make_ev(url="https://reuters.com/article/test", content="x" * 2500)
        score, breakdown = score_evidence(ev)
        assert breakdown["factual"] == 30  # very-high
        assert breakdown["credibility"] == 10  # high
        assert breakdown["content"] == 15  # rich
        assert score >= 65

    def test_gov_domain_gets_tld_bonus(self):
        ev = _make_ev(url="https://eia.gov/data/electricity", content="x" * 2500)
        score, breakdown = score_evidence(ev)
        assert breakdown["gov_tld"] == 15
        # .gov gets elevated unrated default (trustworthy without MBFC)
        assert breakdown["factual"] == 20
        assert breakdown["credibility"] == 2

    def test_legiscan_source_type_score(self):
        ev = _make_ev(url="https://legiscan.com/US/bill/HR5678")
        _, breakdown = score_evidence(ev)
        assert breakdown["source_type"] == 28

    def test_unrated_short_snippet(self):
        ev = _make_ev(url="https://random-blog.net/post", content="x" * 100)
        score, breakdown = score_evidence(ev)
        assert breakdown["factual"] == 4    # unknown domain — low default
        assert breakdown["credibility"] == 2  # unknown domain — low default
        assert breakdown["content"] == 0     # too short
        assert score < 20

    def test_breakdown_sums_to_total(self):
        ev = _make_ev(url="https://reuters.com/article/x", content="x" * 1000)
        score, breakdown = score_evidence(ev)
        assert score == sum(breakdown.values())


# ---------- Unit tests: rank_and_select ----------

class TestRankAndSelect:
    def test_noop_under_limit(self):
        """No ranking when evidence fits within max_items."""
        items = [_make_ev(url=f"https://example.com/{i}") for i in range(5)]
        selected, dropped = rank_and_select(items, max_items=10)
        assert len(selected) == 5
        assert len(dropped) == 0

    def test_caps_at_max_items(self):
        items = [_make_ev(url=f"https://site{i}.com/article") for i in range(30)]
        selected, dropped = rank_and_select(items, max_items=20)
        assert len(selected) == 20
        assert len(dropped) == 10

    def test_domain_cap_enforced(self):
        """A domain with 5 items should only keep max_per_domain."""
        items = []
        # 5 items from same domain
        for i in range(5):
            items.append(_make_ev(url=f"https://reuters.com/article/{i}", content="x" * 2500))
        # 20 from other domains
        for i in range(20):
            items.append(_make_ev(url=f"https://site{i}.com/article", content="x" * 300))

        selected, dropped = rank_and_select(items, max_items=20, max_per_domain=3)

        reuters_selected = [e for e in selected if "reuters.com" in e.get("source_url", "")]
        assert len(reuters_selected) == 3

        # The 2 dropped reuters items should have reason=domain_cap
        reuters_dropped = [d for d in dropped if "reuters.com" in d["evidence"].get("source_url", "")]
        assert len(reuters_dropped) == 2
        assert all(d["reason"] == "domain_cap" for d in reuters_dropped)

    def test_high_quality_selected_over_low(self):
        """Reuters (very-high factual) should rank above unknown blog."""
        items = []
        # Low quality items first (discovery order)
        for i in range(20):
            items.append(_make_ev(url=f"https://blog{i}.net/post", content="x" * 100))
        # High quality item last
        items.append(_make_ev(url="https://reuters.com/important", content="x" * 2500))

        selected, dropped = rank_and_select(items, max_items=20)

        selected_urls = [e.get("source_url") for e in selected]
        assert "https://reuters.com/important" in selected_urls

    def test_gov_domain_survives_cap(self):
        """Government data sources should survive the quality cap."""
        items = []
        for i in range(22):
            items.append(_make_ev(url=f"https://blog{i}.net/post", content="x" * 100))
        # Gov source added late
        items.append(_make_ev(url="https://eia.gov/data/electricity", content="x" * 3000))

        selected, dropped = rank_and_select(items, max_items=20)

        selected_urls = [e.get("source_url") for e in selected]
        assert "https://eia.gov/data/electricity" in selected_urls

    def test_stable_sort_preserves_order_for_ties(self):
        """Items with equal scores keep their original discovery order."""
        items = [
            _make_ev(url=f"https://site{i}.com/article", content="x" * 300)
            for i in range(25)
        ]
        selected, _ = rank_and_select(items, max_items=20)

        # All have equal scores (same content length, all unrated).
        # First 20 in discovery order should be selected.
        selected_urls = [e.get("source_url") for e in selected]
        for i in range(20):
            assert f"https://site{i}.com/article" in selected_urls

    def test_diversity_minimum_domains(self):
        """With max_per_domain=3 and max_items=20, need at least 7 unique domains."""
        items = []
        # 10 domains, 4 items each = 40 items
        for d in range(10):
            for i in range(4):
                items.append(_make_ev(
                    url=f"https://domain{d}.com/article/{i}",
                    content="x" * 300,
                ))

        selected, _ = rank_and_select(items, max_items=20, max_per_domain=3)
        domains = set()
        for ev in selected:
            from src.tools.source_ratings import extract_domain
            domains.add(extract_domain(ev["source_url"]))
        assert len(domains) >= 7


# ---------- Unit tests: format_ranking_log ----------

class TestFormatRankingLog:
    def test_structure(self):
        items = [_make_ev(url=f"https://site{i}.com/a") for i in range(25)]
        selected, dropped = rank_and_select(items, max_items=20)
        result = format_ranking_log(selected, dropped)

        assert "selected" in result
        assert "dropped" in result
        assert "domain_distribution" in result
        assert "unique_domains" in result
        assert len(result["selected"]) == 20
        assert len(result["dropped"]) == 5

    def test_dropped_includes_reason(self):
        items = []
        for i in range(5):
            items.append(_make_ev(url="https://same-domain.com/article"))
        for i in range(20):
            items.append(_make_ev(url=f"https://other{i}.com/a"))

        selected, dropped = rank_and_select(items, max_items=20, max_per_domain=3)
        result = format_ranking_log(selected, dropped)

        reasons = {d["reason"] for d in result["dropped"]}
        assert "domain_cap" in reasons
