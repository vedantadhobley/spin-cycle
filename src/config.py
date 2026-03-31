"""Centralized configuration constants for the Spin Cycle pipeline.

All tunable parameters in one place. Grouped by pipeline phase.
Values that are environment-driven (LLAMA_URL, API keys) stay as os.getenv()
at point of use — this file is for code-level constants only.
"""

import os

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

TASK_QUEUE = "spin-cycle-verify"
TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")

# LLM server parallelism — matched to llama.cpp --parallel N.
# Controls: max_concurrent_activities in worker, semaphore in workflows,
# and batch-of-2 parallel patterns in extract_transcript and synthesize.
MAX_CONCURRENT = 2

# ---------------------------------------------------------------------------
# LLM invocation defaults
# ---------------------------------------------------------------------------

# Default temperatures (overridden per-call where needed)
LLM_TEMPERATURE = 0.0
LLM_TEMPERATURE_ON_RETRY = 0.3
LLM_MAX_RETRIES = 2

# Per-step max_tokens (extraction/review/judge need large output windows)
EXTRACTION_MAX_TOKENS = 16384
REVIEW_MAX_TOKENS = 16384
SYNTHESIS_MAX_TOKENS = 4096
NORMALIZE_MAX_TOKENS = 16384
DECOMPOSE_MAX_TOKENS = 16384
JUDGE_MAX_TOKENS = 16384
SYNTHESIZE_MAX_TOKENS = 16384

# Retry pause between LLM attempts (seconds)
LLM_RETRY_DELAY = 1

# ---------------------------------------------------------------------------
# Transcript chunking (Phase 1)
# ---------------------------------------------------------------------------

TARGET_WORDS_PER_CHUNK = 2500
OVERLAP_WORDS = 500

# ---------------------------------------------------------------------------
# Batch classification (between extraction and dedup)
# ---------------------------------------------------------------------------

CLASSIFY_BATCH_SIZE = 50       # claims per classification LLM call
CLASSIFY_MAX_TOKENS = 4096    # generous for ~50 claims × 25 tokens each

# ---------------------------------------------------------------------------
# Embedding-based dedup (Phase 2)
# ---------------------------------------------------------------------------

EMBEDDING_SIMILARITY_THRESHOLD = 0.85  # cosine sim; above this = same claim
EMBEDDING_BATCH_SIZE = 32              # texts per embedding API call
EMBEDDING_TIMEOUT = 60.0               # seconds per embedding batch
NUMERIC_SKELETON_JACCARD_THRESHOLD = 0.7  # word overlap to consider "structurally same"

# ---------------------------------------------------------------------------
# Verification pipeline
# ---------------------------------------------------------------------------

MAX_FACTS = 10  # cap on atomic facts from decompose
MAX_ALL_PARTIES = 40  # cap on merged all_parties list

# Validator minimum field lengths (chars)
# These prevent the LLM from producing empty/stub fields.
MIN_RATIONALE_LENGTH = 10
MIN_STATEMENT_LENGTH = 15
MIN_FIELD_LENGTH_SHORT = 5   # short fields (claim_interpretation, thesis_restatement)
MIN_FIELD_LENGTH_MEDIUM = 10  # medium fields (reasoning, direction_reasoning)
MIN_FIELD_LENGTH_LONG = 15    # long fields (thesis_statement, decompose analysis)
MIN_FIELD_LENGTH_ANALYSIS = 20  # analysis fields (claim_analysis)

# Citation enforcement
MIN_JUDGE_CITATIONS = 3
MIN_SYNTHESIZE_CITATIONS = 5

# Confidence thresholds for validator warnings
LOW_CONFIDENCE_THRESHOLD = 0.3  # strong verdict + low confidence → warning
HIGH_CONFIDENCE_THRESHOLD = 0.8  # unverifiable + high confidence → warning

# Near-duplicate detection in decompose quality check
DECOMPOSE_DEDUP_RATIO = 0.8  # word-set overlap ratio for trivial dedup

# ---------------------------------------------------------------------------
# Temporal activity timeouts (seconds)
# ---------------------------------------------------------------------------

# Verification workflow
TIMEOUT_CREATE_CLAIM = 15
TIMEOUT_DECOMPOSE = 180
TIMEOUT_RESEARCH = 540
TIMEOUT_JUDGE = 300
TIMEOUT_SYNTHESIZE = 300
TIMEOUT_STORE_RESULT = 30
TIMEOUT_START_NEXT = 30
TIMEOUT_NOTIFY_FRONTEND = 10

# Extraction workflow
TIMEOUT_FETCH_TRANSCRIPT = 60
TIMEOUT_EXTRACT_CHUNK = 2700  # 45 min — large chunks on slow model
TIMEOUT_CLASSIFY_CLAIMS = 600  # 10 min — 50-claim batches on local model
TIMEOUT_DEDUP_CLAIMS = 600    # 10 min — per-speaker embedding dedup
TIMEOUT_SYNTHESIZE_CLAIM = 300
TIMEOUT_STORE_CLAIMS = 30
TIMEOUT_UPDATE_STATUS = 15
TIMEOUT_FINISH_TRANSCRIPT = 30

# LLM streaming
LLM_IDLE_TIMEOUT = 90          # seconds — abort if no tokens for this long
LLM_NO_JSON_TOKEN_LIMIT = 4000  # abort if this many tokens produced with no '{' seen

# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------

RESEARCH_MAX_STEPS = 47  # LangGraph recursion limit (~15 tool calls)
RESEARCH_TIMEOUT = 420   # agent soft timeout (seconds)
SEED_MAX_RESULTS = 30
SEED_SEARCH_CONCURRENCY = 8
PREFETCH_MAX = 10
PREFETCH_CONTENT_MAX = 5000  # chars (vs 8000 in page_fetcher — intentional)
PREFETCH_CONCURRENCY = 5

# ---------------------------------------------------------------------------
# Evidence ranking (see evidence_ranker.py for full score tables)
# ---------------------------------------------------------------------------

MAX_JUDGE_EVIDENCE = 20
MAX_PER_DOMAIN = 3
MAX_PER_DOMAIN_TIER1 = 5
MAX_GOV_EVIDENCE = 4

# ---------------------------------------------------------------------------
# External service timeouts (seconds)
# ---------------------------------------------------------------------------

WIKIDATA_TIMEOUT = 10.0
WIKIDATA_SPARQL_TIMEOUT = 15.0
SERPER_TIMEOUT = 15.0
MBFC_SCRAPE_TIMEOUT = 15.0
MBFC_API_TIMEOUT = 30.0
PAGE_FETCH_TIMEOUT = 15
CSPAN_PAGE_TIMEOUT = 60000  # milliseconds (Playwright)
FRONTEND_NOTIFY_TIMEOUT = 5.0

# ---------------------------------------------------------------------------
# Cache TTLs
# ---------------------------------------------------------------------------

MBFC_CACHE_TTL_DAYS = 30
MBFC_INDEX_REFRESH_DAYS = 7
WIKIDATA_CACHE_TTL_DAYS = 7
