# Spin Cycle

News claim verification pipeline powered by LangGraph agents and local LLMs.

In a media landscape where misinformation spreads faster than corrections, Spin Cycle automatically ingests news claims, decomposes them into verifiable sub-claims, researches evidence from multiple sources, and delivers structured verdicts with full reasoning chains.

## Why "Spin Cycle"?

Because we're putting the spin through the wringer.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              vedanta-systems                │
                    │            (frontend @ 3100)                │
                    └──────────────────┬──────────────────────────┘
                                       │ HTTP
                    ┌──────────────────▼──────────────────────────┐
                    │         FastAPI Backend (3500/4500)          │
                    │  - Submit claims for verification            │
                    │  - Query verdicts + evidence chains          │
                    └──────┬──────────────────┬───────────────────┘
                           │                  │
              ┌────────────▼────┐    ┌────────▼─────────────┐
              │   Temporal       │    │   PostgreSQL          │
              │   (3501/4501)    │    │   claims, verdicts,   │
              │   Workflow        │    │   evidence, sources   │
              │   orchestration  │    └───────────────────────┘
              └────────┬─────────┘
                       │ activities
              ┌────────▼─────────────────────────────────────────┐
              │           Verification Pipeline                   │
              │                                                   │
              │  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
              │  │Decompose│→│ Research  │→│  Judge &        │  │
              │  │ (LLM)   │  │ (agent)  │  │  Synthesize    │  │
              │  └─────────┘  └──────────┘  └────────────────┘  │
              │                      │                            │
              │               ┌──────▼──────┐                    │
              │               │ DuckDuckGo  │                    │
              │               │ SearXNG     │                    │
              │               │ Wikipedia   │                    │
              │               │ Page Fetch  │                    │
              │               └─────────────┘                    │
              │                      │                            │
              │               ┌──────▼──────┐                    │
              │               │ Programmatic│                    │
              │               │ LegiScan    │                    │
              │               │ Wikidata    │                    │
              │               │ MBFC        │                    │
              │               └─────────────┘                    │
              └─────────────────────────────────────────────────┘
```

## Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | ReAct agent for autonomous evidence gathering |
| LLM toolkit | [LangChain](https://python.langchain.com/) | ChatOpenAI client, tool wrappers, message types |
| Workflow engine | [Temporal](https://temporal.io/) | Durable execution, retries, scheduling, visibility |
| LLM | Qwen3.5-35B-A3B (via llama.cpp/ROCm) | Single instance, ~38 tok/s sustained throughput |
| NER | [SpaCy](https://spacy.io/) (en_core_web_sm) | Entity extraction from claims and evidence (CPU, ~ms) |
| Knowledge graph | [Wikidata](https://www.wikidata.org/) SPARQL | Ownership chains, media holdings, family relationships |
| Source ratings | [MBFC](https://mediabiasfactcheck.com/) | Bias and factual reporting ratings (REST API index bootstrap + cached) |
| Legislation | [LegiScan](https://legiscan.com/) API | US bill search, roll call votes, bill text (Civic API tier) |
| Grammar | [LanguageTool](https://languagetool.org/) (Java, local) | Grammar correction on all LLM outputs (catches quantization artifacts) |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 (async) | Claims, sub-claims, evidence, verdicts, source ratings |
| API | FastAPI | REST endpoints for claim submission and querying |

## How It Works

### 1. Claim Submission

```bash
curl -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin was created by Satoshi Nakamoto in 2009"}'
```

### 2. Verification Pipeline (Temporal workflow)

The claim triggers `VerifyClaimWorkflow` — a flat pipeline of 7 activities:

```
create_claim          Creates DB record (skipped if called via API)
    ↓
decompose_claim       Normalize → Decompose (2 LLM calls, 1 activity)
                      Normalize: 7 transformations (bias neutralization,
                        operationalization, opinion separation, coreference,
                        reference grounding, speculation, rhetorical framing)
                      Decompose: flat atomic facts + thesis extraction
                      Each fact gets categories + LLM-written seed queries
                      Guided by 15-category linguistic pattern taxonomy
                        + 9 extraction rules (decontextualize, entity disambig.,
                          operationalize comparisons)
                      SpaCy NER augments entity extraction
                      Wikidata expands parties → ownership, media, family
    ↓
RESEARCH PHASE (all facts, 2 concurrent):
    research_subclaim   Phase 1a: Programmatic seed search (DuckDuckGo + SearXNG, ~30-50 URLs)
       ↓                Phase 1b: MBFC await → MBFC ownership→Wikidata enrichment
       ↓                  → score → conflict detect → rank → top 30 (tier + conflicts)
       ↓                Phase 2: LangGraph ReAct agent (fetches, follow-up)
       ↓                Phase 3: LegiScan + evidence NER→Wikidata enrichment
       ↓                Returns: evidence + enriched interested parties
    ↓                   (enriched parties merged across sub-claims)
JUDGE PHASE (all facts, 2 concurrent, receives merged parties):
    judge_subclaim      Pre-judge: lightweight NER cleanup (parallel Wikidata)
       ↓                Evidence ranking + MBFC annotation
       ↓                4 conflict-of-interest checks per evidence item
       ↓                LLM evaluates evidence (6-level verdict scale)
    ↓
synthesize_verdict    LLM combines sub-verdicts using the speaker's thesis
                      as primary rubric (not naive fact counting)
    ↓
store_result              Writes results to Postgres
    ↓
start_next_queued_claim   Picks up next queued claim (if any) and starts its workflow
```

Only one claim verifies at a time (to avoid LLM contention). When a claim finishes, the workflow starts the next queued one. Submitting while a claim is running queues it as a DB row.

The **flat facts** approach (matching Google SAFE and FActScore) means the LLM outputs facts directly as strings, guided by a comprehensive **linguistic patterns taxonomy** that catches presuppositions, quantifier scope, temporal boundaries, causation types, and more.

The thesis extraction ensures the synthesizer understands the **intent** of the claim, not just the individual facts. For example, a claim comparing two countries' policies is rated `mostly_false` even though 5/6 sub-facts are true — because one country's data contradicts the speaker's parallel comparison.

### 3. Result

```bash
curl http://localhost:4500/claims/{id}
```

```json
{
  "text": "Bitcoin was created by Satoshi Nakamoto in 2009 and the first block was mined on January 3rd",
  "status": "verified",
  "verdict": "true",
  "confidence": 0.95,
  "sub_claims": [
    {
      "text": "Bitcoin was created by Satoshi Nakamoto in 2009",
      "verdict": "true",
      "confidence": 0.95,
      "reasoning": "Multiple sources, including Wikipedia articles and news summaries, confirm that Bitcoin was created by Satoshi Nakamoto in 2009."
    },
    {
      "text": "The first block of Bitcoin was mined on January 3rd, 2009",
      "verdict": "true",
      "confidence": 0.95,
      "reasoning": "Multiple sources consistently state that the first Bitcoin block was mined on January 3, 2009."
    }
  ]
}
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Access to an LLM server (any OpenAI-compatible endpoint, e.g. llama.cpp via Tailscale)
- (Optional) The `luv-dev` Docker network if running alongside vedanta-systems

### Setup

```bash
# Clone the repo
git clone git@github.com:vedantadhobley/spin-cycle.git
cd spin-cycle

# Create .env from template
cp .env.example .env
# Edit .env to set LLAMA_URL (your LLM endpoint)

# Create the external network (if it doesn't exist)
docker network create luv-dev 2>/dev/null || true

# Start the dev stack (7 containers)
docker compose -f docker-compose.dev.yml up -d

# Verify everything is running
docker compose -f docker-compose.dev.yml ps
```

### Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:4500 | FastAPI backend |
| Temporal UI | http://localhost:4501 | Workflow dashboard |
| Adminer | http://localhost:4502 | Postgres web UI (Dracula theme) |
| SearXNG | http://localhost:4503 | Self-hosted meta-search engine |

Adminer login: Server `spin-cycle-dev-postgres`, User `spincycle`, Password `spin-cycle-dev`, Database `spincycle`.

### Testing Claims

There are three ways to submit claims and observe the pipeline.

#### Via curl (API)

```bash
# Submit a claim for verification
curl -s -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin was created by Satoshi Nakamoto in 2009"}' | python3 -m json.tool

# Returns something like:
# { "id": "abc123...", "text": "...", "status": "pending", ... }

# Wait ~30 seconds for the pipeline to finish, then check the result
curl -s http://localhost:4500/claims/{id} | python3 -m json.tool

# List all claims (with pagination and optional status filter)
curl -s 'http://localhost:4500/claims?limit=10' | python3 -m json.tool
curl -s 'http://localhost:4500/claims?status=verified' | python3 -m json.tool

# Health check
curl -s http://localhost:4500/health
```

Submit multiple claims at once (first starts immediately, rest are queued):
```bash
curl -s -X POST http://localhost:4500/claims/batch \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      {"text": "Claim one to verify", "source_name": "Source A"},
      {"text": "Claim two to verify"},
      {"text": "Claim three to verify"}
    ]
  }' | python3 -m json.tool
```

You can also submit a claim with source attribution:
```bash
curl -s -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Country A spent $50 billion on Project X before cancelling the second phase",
    "source": "https://example.com/article",
    "source_name": "The Example Times"
  }' | python3 -m json.tool
```

#### Via Temporal UI

Open http://localhost:4501 and you can:

1. **Watch workflows execute** — click any workflow to see the full activity history (decompose → research → judge → synthesize → store), including inputs, outputs, and timings for each step.

2. **Start a workflow manually** — click "Start Workflow" in the top right:
   - Workflow Type: `VerifyClaimWorkflow`
   - Workflow ID: anything unique, e.g. `test-1`
   - Task Queue: `spin-cycle-verify`
   - Input: `[null, "The claim text you want to verify"]`

   The first argument is `null` — the workflow will create the claim record in the database automatically. (If you already have a claim ID from the API, you can pass it instead of `null`.)

3. **Replay and debug** — if a workflow fails, you can see exactly which activity failed, what it received, and what error it threw.

#### Via worker logs

Watch the verification pipeline in real-time:
```bash
# Follow worker logs (shows every step as it happens)
docker logs -f spin-cycle-dev-worker

# With LOG_FORMAT=pretty (default in dev), you'll see:
# I [WORKER    ] starting: Connecting to Temporal | temporal_host=... task_queue=spin-cycle-verify
# I [WORKER    ] ready: Worker listening | task_queue=spin-cycle-verify activity_count=7
# I [DECOMPOSE ] normalized: Claim normalized | changes=[...]
# I [DECOMPOSE ] done: Claim decomposed | sub_count=3 thesis=...
# I [WORKFLOW  ] decomposed: Claim decomposed into atomic facts | fact_count=3
# I [RESEARCH  ] start: Starting research agent | sub_claim=...
# I [RESEARCH  ] done: Research complete | evidence_count=16
# I [JUDGE     ] done: Sub-claim judged | verdict=true confidence=0.95
# I [SYNTHESIZE] done: Verdict synthesized | verdict=mostly_true confidence=0.85
# I [STORE     ] done: Result stored in database | claim_id=... verdict=mostly_true

# With LOG_FORMAT=json (default in prod), output is JSON for Grafana Loki
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | (required) | LLM endpoint (Qwen3.5-35B-A3B, unified thinking/non-thinking) |
| `LLAMA_EMBED_URL` | (optional) | Embeddings endpoint (not yet used) |
| `POSTGRES_PASSWORD` | `spin-cycle-dev` | Application Postgres password |
| `LOG_FORMAT` | `json` (prod) / `pretty` (dev) | Log output format — `json` for Grafana Loki, `pretty` for terminal |
| `LOG_LEVEL` | `INFO` | Log level — `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `NEWSAPI_KEY` | (empty) | NewsAPI key (not currently wired — future use) |
| `SERPER_API_KEY` | (empty) | Serper key for Google search evidence |
| `BRAVE_API_KEY` | (empty) | Brave Search API key |
| `LEGISCAN_API_KEY` | (empty) | LegiScan Civic API key (US legislation, votes, bill text) |
| `SEARXNG_URL` | `http://searxng:8080` | SearXNG meta-search endpoint (secondary search, self-hosted) |

## Port Allocation

| Port | Dev | Prod | Service |
|------|-----|------|---------|
| Base | 4500 | 3500 | FastAPI API |
| +1 | 4501 | 3501 | Temporal UI |
| +2 | 4502 | 3502 | Adminer |
| +3 | 4503 | — | SearXNG (dev only) |

## Database

Six tables in PostgreSQL, all with UUID primary keys (except cache tables which use string PKs):

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `claims` | Top-level claims | text, source_url, status (queued→pending→verified), timestamps |
| `sub_claims` | Decomposed sub-claim tree | text, parent_id (self-ref FK), is_leaf, verdict, confidence, reasoning |
| `evidence` | Research results per sub-claim | source_type (web/wikipedia/news_api), content, URL |
| `verdicts` | Overall claim verdict | verdict, confidence, reasoning, reasoning_chain (JSONB) |
| `source_ratings` | Cached MBFC ratings | domain (PK), bias, factual_reporting, ownership, country |
| `wikidata_cache` | Cached Wikidata entity data | entity_name (PK), qid, relationships (JSONB), 7-day TTL |

Relationships: `claims` → has many `sub_claims` → has many `evidence`. `claims` → has one `verdict`. Cache tables are standalone (no FK relationships).

See [ARCHITECTURE.md](ARCHITECTURE.md) for full schema documentation with column types and constraints.

## Project Structure

```
spin-cycle/
├── docker-compose.dev.yml          # Dev stack (4500-4502)
├── docker-compose.yml              # Prod stack (3500-3502)
├── Dockerfile / Dockerfile.dev     # Container images
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
│
├── src/
│   ├── worker.py                   # Temporal worker entrypoint
│   │
│   ├── llm/                        # LLM client layer
│   │   ├── client.py               # ChatOpenAI client setup
│   │   ├── invoker.py              # invoke_llm() — structured output + retries
│   │   ├── parser.py               # Response parsing helpers
│   │   └── validators.py           # Semantic validators (normalize, decompose, synthesize)
│   │
│   ├── utils/                      # Shared utilities
│   │   ├── logging.py              # Structured logging (JSON for Loki, pretty for dev)
│   │   ├── ner.py                  # SpaCy NER — entity extraction (PERSON/ORG)
│   │   ├── quote_detection.py      # Detect claim subject quotes in evidence text
│   │   ├── text_cleanup.py         # Grammar/spell check for LLM output
│   │   └── evidence_ranker.py      # Source + evidence quality scoring, seed ranking, judge capping
│   │
│   ├── api/                        # FastAPI backend
│   │   ├── app.py                  # App + lifespan
│   │   └── routes/
│   │       ├── health.py           # Health check
│   │       └── claims.py           # Claim CRUD
│   │
│   ├── agent/                      # Domain logic (called by Temporal activities)
│   │   ├── decompose.py            # Normalize + extract facts + Wikidata expansion
│   │   ├── research.py             # Seed search + rank + ReAct agent + evidence extraction
│   │   ├── judge.py                # Evidence ranking, annotation, LLM verdict
│   │   ├── synthesize.py           # Verdict synthesis
│   │   └── claim_category.py       # Seed query routing (backend selection)
│   │
│   ├── tools/                      # Evidence gathering + data sources
│   │   ├── source_ratings.py       # MBFC ratings (scrape + cache + parallel await)
│   │   ├── source_filter.py        # Domain blocklist + MBFC cache population
│   │   ├── mbfc_index.py           # MBFC REST API index bootstrap (~10,300 sources)
│   │   ├── media_matching.py       # URL↔media matching, publisher ownership, MBFC owner extraction
│   │   ├── wikidata.py             # Wikidata SPARQL — ownership chains, relationships
│   │   ├── legiscan.py             # LegiScan API — US legislation, votes, bill text
│   │   ├── searxng.py              # SearXNG meta-search (secondary)
│   │   ├── serper.py               # Serper (Google Search API)
│   │   ├── brave.py                # Brave Search API
│   │   ├── web_search.py           # DuckDuckGo (primary search backend)
│   │   ├── wikipedia.py            # Wikipedia API
│   │   └── page_fetcher.py         # URL → text extraction + SpaCy entity metadata
│   │
│   ├── prompts/                    # LLM prompts (heavily documented)
│   │   ├── verification.py         # Decompose, Research, Judge, Synthesize
│   │   └── linguistic_patterns.py  # 15-category linguistic pattern taxonomy
│   │
│   ├── schemas/                    # Data schemas
│   │   ├── api.py                  # Pydantic schemas for API
│   │   ├── llm_outputs.py          # Pydantic schemas for LLM structured output
│   │   └── interested_parties.py   # InterestedPartiesDict TypedDict (pipeline contract)
│   │
│   ├── workflows/
│   │   └── verify.py               # VerifyClaimWorkflow
│   │
│   ├── activities/
│   │   └── verify_activities.py    # Temporal activities (7 activities)
│   │
│   └── db/
│       ├── models.py               # SQLAlchemy models
│       └── session.py              # Async DB sessions
│
└── tests/
    ├── test_health.py
    ├── test_schemas.py
    ├── test_evidence_ranker.py
    ├── regression_claims.py         # Known-answer regression suite
    └── stress_claims.py             # Load/stress testing
```

## What's Next

1. **Alembic migrations** — proper database schema versioning (currently using raw SQL ALTER TABLE)
2. **Extraction pipeline** — automated claim ingestion from RSS feeds via scheduled Temporal workflows
3. **Calibration test suite** — benchmark against known claims to measure accuracy
4. **LangFuse** — self-hosted LLM observability for prompt debugging

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical deep dive, including the extraction pipeline design, database schema details, and how LangChain/LangGraph/Temporal fit together.
