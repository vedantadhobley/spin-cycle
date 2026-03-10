# Seed Relevance Filtering

## Problem

The seed ranking pipeline (`score_url()` in `evidence_ranker.py`) scores results
purely on **source reputation** (MBFC factual rating + TLD heuristics). There is
zero topical/semantic relevance signal. This means a TIER 1 page about the wrong
topic beats a TIER 2 page about the exact right topic every time.

### Observed failures (2026-03-10 run)

| Sub-claim | Garbage result | Score | Why it ranked high |
|-----------|---------------|-------|--------------------|
| "did Pelosi outperform all hedge funds" | `pmc.ncbi.nlm.nih.gov` (DID medical paper) | 37 (TIER 1 gov) | `.gov` TLD bonus |
| "did Pelosi outperform all hedge funds" | `psychologytoday.com` (DID article) | ~20 | decent MBFC rating |
| "did Pelosi outperform all hedge funds" | `d-id.com`, `did.co` (tech companies) | 6 | unknown but not blocked |
| "Supreme Court precedents 2021-2026" | `history.state.gov/frus1917/d881` (1917 diplomatic cable) | 37 (TIER 1 gov) | `.gov` TLD bonus |
| "Supreme Court precedents 2021-2026" | `usa.gov/about-the-us` (generic country info) | 37 (TIER 1 gov) | `.gov` TLD bonus |
| "China renewable energy vs US+EU" | `travelchinaguide.com` | ~6 | not blocked |
| "China renewable energy vs US+EU" | `newworldencyclopedia.org/entry/China` | ~6 | not blocked |

These waste prefetch slots (up to 10) and pollute the agent's context window.

### Root cause: SearXNG multi-engine fan-out

SearXNG sends queries to Google, Bing, DDG, Brave, Mojeek, Qwant, Wikipedia,
and Wikidata simultaneously. Smaller/worse engines keyword-match on individual
terms ("DID" → Dissociative Identity Disorder, "China" → travel guides) and
return off-topic results. SearXNG merges them without relevance filtering.

**SearXNG has been disabled from seed searches** (2026-03-10). Serper + DDG
produce much cleaner results. The SearXNG container remains running for
potential future re-enablement with relevance filtering.

### Secondary root cause: `.gov` over-scoring

`.gov` gets automatic 35 points (FACTUAL_UNRATED_GOV=20 + GOV_TLD_SCORE=15)
even without MBFC data. This is higher than a high-factual MBFC outlet (24+0+2=26).
A `.gov` page about 1917 diplomacy outranks an AP News article about the actual
claim topic.

**TODO**: Reduce `.gov` scoring. Suggested: `GOV_TLD_SCORE=5`, `FACTUAL_UNRATED_GOV=12`.

---

## Proposed Solution: Keyword Gate + Embedding Similarity

Two-layer relevance scoring applied at seed ranking time, before prefetch.
The relevance score acts as a **multiplier** on the existing quality score,
so high-quality + relevant pages rank highest.

### Layer 1: Keyword Overlap (free, instant)

Extract meaningful tokens from the sub-claim, check overlap with each
result's title + snippet. Zero cost, catches the obvious mismatches.

```python
import re

# Simple stopwords — extend as needed
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "has", "have",
              "had", "been", "be", "do", "does", "did", "will", "would",
              "could", "should", "may", "might", "can", "shall", "of", "in",
              "to", "for", "on", "at", "by", "with", "from", "as", "into",
              "than", "that", "this", "it", "its", "and", "or", "but", "not",
              "no", "if", "so", "up", "out", "about", "over", "more", "all",
              "any", "each", "every", "both", "few", "most", "other", "some"}

def keyword_overlap(sub_claim: str, title: str, snippet: str) -> float:
    """Fraction of sub-claim keywords found in result text. 0.0 to 1.0."""
    claim_tokens = {t for t in re.findall(r'\w+', sub_claim.lower())
                    if t not in _STOPWORDS and len(t) > 2}
    if not claim_tokens:
        return 1.0  # can't score, pass through
    result_text = (title + " " + snippet).lower()
    hits = sum(1 for t in claim_tokens if t in result_text)
    return hits / len(claim_tokens)
```

**Expected catches**:
- `psychologytoday.com/dissociative-identity-disorder` → 0.0 overlap with
  "Pelosi stock trading hedge fund" → score drops to near-zero
- `history.state.gov/frus1917` → 0.0 overlap with "Supreme Court precedent"
- `travelchinaguide.com` → maybe 0.1 overlap ("China" matches)

**Gate threshold**: If keyword overlap < 0.15, skip embedding computation
and assign relevance = 0.0 (hard reject from prefetch candidates).

### Layer 2: Embedding Similarity (cheap, semantic)

Use `sentence-transformers/all-MiniLM-L6-v2` (80MB, runs on CPU):
- ~5ms per embedding (single), ~50ms for batch of 30
- Understands synonyms: "coal power plant construction" ≈ "fossil fuel capacity building"
- Understands topic scope: "Supreme Court rulings" ≠ "1917 diplomatic cables"

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module level (lazy)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embedding_similarity(sub_claim: str, texts: list[str]) -> list[float]:
    """Batch cosine similarity between sub-claim and result texts."""
    model = _get_model()
    claim_emb = model.encode(sub_claim, normalize_embeddings=True)
    text_embs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    # Cosine similarity (already normalized)
    similarities = text_embs @ claim_emb
    return similarities.tolist()
```

### Combined Relevance Score

```python
def relevance_score(sub_claim: str, title: str, snippet: str) -> float:
    """Combined relevance score. 0.0 to 1.0."""
    kw = keyword_overlap(sub_claim, title, snippet)
    if kw < 0.15:
        return 0.0  # hard reject — no keyword match at all

    text = f"{title} {snippet}"
    sim = embedding_similarity(sub_claim, [text])[0]

    # Weighted combination: embeddings matter more but keywords gate
    return 0.3 * kw + 0.7 * max(0.0, sim)
```

### Integration Point: `_rank_and_filter_seeds()` in research.py

Currently each seed gets a quality score from `score_url()`. Add relevance
as a multiplier:

```python
# Current:
score, breakdown = score_url(url)

# Proposed:
quality, breakdown = score_url(url)
relevance = relevance_score(sub_claim, title, snippet)
score = quality * (0.2 + 0.8 * relevance)
# Floor of 0.2 so a perfect-quality, low-relevance page isn't zeroed out entirely
# (it might still be useful background), but it won't rank above a
# medium-quality, highly-relevant page.
```

### Batch Optimization

For 30 seed results, compute all embeddings in one batch call (~50ms total
on CPU) rather than 30 individual calls. The keyword gate runs first to
skip obviously irrelevant results before embeddings.

```python
# In _rank_and_filter_seeds():
# 1. Keyword gate — cheap, eliminates ~30-50% of noise
for seed in ranked_seeds:
    kw = keyword_overlap(sub_claim, seed["title"], seed["snippet"])
    if kw < 0.15:
        seed["_relevance"] = 0.0
        continue
    candidates_for_embedding.append(seed)

# 2. Batch embedding — one call for all survivors
texts = [f"{s['title']} {s['snippet']}" for s in candidates_for_embedding]
sims = embedding_similarity(sub_claim, texts)
for seed, sim in zip(candidates_for_embedding, sims):
    seed["_relevance"] = 0.3 * seed["_kw_overlap"] + 0.7 * max(0.0, sim)
```

---

## `.gov` Scoring Adjustment

Separate change, can be done independently of relevance filtering.

### Current scoring (stacks)
- `FACTUAL_UNRATED_GOV = 20` (factual component)
- `GOV_TLD_SCORE = 15` (TLD component)
- Total for unrated `.gov`: **37** (higher than MBFC high-factual at ~26)

### Proposed scoring
- `FACTUAL_UNRATED_GOV = 12` (above unknown=4, below mostly-factual=16)
- `GOV_TLD_SCORE = 5` (small credibility bonus, not a catapult)
- Total for unrated `.gov`: **19** (below MBFC high-factual, above unknown)

This means `eia.gov/todayinenergy` (relevant energy data) and
`history.state.gov/frus1917` (irrelevant 1917 cables) both start at 19,
but with relevance filtering the former would score much higher for an
energy claim.

---

## Dependencies

- `sentence-transformers` (pip) — ~80MB model download on first use
- `torch` — likely already installed for Qwen/ROCm, but the embedding model
  runs on CPU (no GPU needed, avoids competing with Qwen for the AMD GPU)
- No API keys, no network calls after model download

## Performance Impact

- Keyword overlap: <1ms for 30 results
- Model load (first call only): ~2-3s
- Batch embedding (30 results): ~50ms on CPU
- Total added latency per sub-claim: ~50ms (after warm-up)
- This runs during the seed ranking phase which currently takes ~1-2s,
  so 50ms is negligible

## Files to Modify

1. `src/utils/evidence_ranker.py` — add `relevance_score()`, adjust `.gov` constants
2. `src/agent/research.py` — pass sub-claim text to ranking, apply relevance multiplier
3. `requirements.txt` — add `sentence-transformers`
4. `src/utils/relevance.py` (new) — keyword overlap + embedding similarity module
