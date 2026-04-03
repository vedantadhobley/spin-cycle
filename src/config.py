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

# Qwen3.5-122B-A10B recommended instruct sampling profiles (from model card).
# Two profiles: "general" for structured extraction/classification/routing,
# "reasoning" for evaluation/judgment/analysis.
# DO NOT use greedy decoding (temp=0) — causes degeneration and repetitions.

# General profile — extraction, classification, summarization, tool routing
LLM_GENERAL_TEMPERATURE = 0.7
LLM_GENERAL_TOP_P = 0.8
LLM_GENERAL_TOP_K = 20
LLM_GENERAL_PRESENCE_PENALTY = 1.5

# Reasoning profile — decompose, judge, synthesize verdict
LLM_REASONING_TEMPERATURE = 1.0
LLM_REASONING_TOP_P = 1.0
LLM_REASONING_TOP_K = 40
LLM_REASONING_PRESENCE_PENALTY = 2.0

LLM_MAX_RETRIES = 2

# Max output tokens for all LLM calls. Set once, used everywhere.
# Adjust if the local model's --ctx-size or available memory changes.
LLM_MAX_TOKENS = 16384

# Retry pause between LLM attempts (seconds)
LLM_RETRY_DELAY = 1

# ---------------------------------------------------------------------------
# Transcript chunking (Phase 1)
# ---------------------------------------------------------------------------

TARGET_SENTENCES_PER_CHUNK = 50   # ~50 sentences per chunk
OVERLAP_SENTENCES = 15            # context sentences before/after target range
SPEAKER_CUTOFF_WINDOW = 5        # max sentences to extend past target to hit a speaker boundary

# ---------------------------------------------------------------------------
# Batch classification (between extraction and dedup)
# ---------------------------------------------------------------------------

CLASSIFY_BATCH_SIZE = 50       # claims per classification LLM call

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
TIMEOUT_INJECT_CONTEXT = 600  # 10 min — simpler than extraction, smaller output
TIMEOUT_CLASSIFY_CLAIMS = 600  # 10 min — 50-claim batches on local model
TIMEOUT_DEDUP_CLAIMS = 600    # 10 min — per-speaker embedding dedup
TIMEOUT_SYNTHESIZE_CLAIM = 300
TIMEOUT_ATTRIBUTE_SPEAKERS = 600  # 10 min — scales with unknown turn count
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

# ---------------------------------------------------------------------------
# External service timeouts (seconds)
# ---------------------------------------------------------------------------

WIKIDATA_TIMEOUT = 10.0
WIKIDATA_SPARQL_TIMEOUT = 15.0
SERPER_TIMEOUT = 15.0
MBFC_SCRAPE_TIMEOUT = 15.0
MBFC_API_TIMEOUT = 30.0
PAGE_FETCH_TIMEOUT = 15
PAGE_FETCH_MAX_CONTENT = 8000
CSPAN_PAGE_TIMEOUT = 60000  # milliseconds (Playwright)
FRONTEND_NOTIFY_TIMEOUT = 5.0

# ---------------------------------------------------------------------------
# Cache TTLs
# ---------------------------------------------------------------------------

MBFC_CACHE_TTL_DAYS = 30
MBFC_INDEX_REFRESH_DAYS = 7
WIKIDATA_CACHE_TTL_DAYS = 7
