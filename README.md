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
              │               │ Wikipedia   │                    │
              │               │ NewsAPI     │                    │
              │               └─────────────┘                    │
              └─────────────────────────────────────────────────┘
```

## Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | ReAct agent for autonomous evidence gathering |
| LLM toolkit | [LangChain](https://python.langchain.com/) | ChatOpenAI client, tool wrappers, message types |
| Workflow engine | [Temporal](https://temporal.io/) | Durable execution, retries, scheduling, visibility |
| LLM | Qwen3-VL-30B-A3B (on joi via llama.cpp) | Instruct (:3101) for decompose/synthesize, Thinking (:3102) for research/judge |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 (async) | Claims, sub-claims, evidence, verdicts |
| API | FastAPI | REST endpoints for claim submission and querying |

## How It Works

### 1. Claim Submission

```bash
curl -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin was created by Satoshi Nakamoto in 2009"}'
```

### 2. Verification Pipeline (Temporal workflow)

The claim triggers `VerifyClaimWorkflow` — 6 activities run in sequence:

```
create_claim          Creates DB record (skipped if called via API)
    ↓
decompose_claim       LLM splits claim into atomic sub-claims
    ↓
research_subclaim     LangGraph ReAct agent searches DuckDuckGo + Wikipedia
    ↓                 (autonomously decides what to search, loops until satisfied)
judge_subclaim        LLM evaluates evidence — "Do NOT use your own knowledge"
    ↓
synthesize_verdict    LLM combines sub-verdicts into overall verdict
    ↓
store_result          Writes everything to Postgres
```

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
- Access to an LLM server (joi via Tailscale, or any OpenAI-compatible endpoint)
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

Adminer login: Server `spin-cycle-dev-postgres`, User `spincycle`, Password `spin-cycle-dev`, Database `spincycle`.

### Testing Claims

There are three ways to submit claims and observe the pipeline.

#### Via curl (API)

```bash
# Submit a claim for verification
curl -s -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "The Great Wall of China is visible from space"}' | python3 -m json.tool

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

You can also submit a claim with source attribution:
```bash
curl -s -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{
    "text": "UK spent £50 billion on HS2 before cancelling the northern leg",
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
# I [WORKER    ] ready: Worker listening | task_queue=spin-cycle-verify activity_count=6
# I [DECOMPOSE ] start: Decomposing claim into sub-claims | claim=The Great Wall of China...
# I [DECOMPOSE ] done: Claim decomposed | num_sub_claims=1
# I [RESEARCH  ] start: Starting research agent | sub_claim=The Great Wall of China...
# I [RESEARCH  ] done: Research agent complete | evidence_count=3 agent_steps=8
# I [JUDGE     ] done: Sub-claim judged | verdict=false confidence=0.95
# I [SYNTHESIZE] done: Overall verdict synthesized | verdict=false confidence=0.95
# I [STORE     ] done: Result stored in database | claim_id=... sub_claims=1

# With LOG_FORMAT=json (default in prod), output is JSON for Grafana Loki
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://joi:3101` | Instruct LLM endpoint (fast, structured output) |
| `LLAMA_REASONING_URL` | `http://joi:3102` | Thinking LLM endpoint (chain-of-thought reasoning) |
| `LLAMA_EMBED_URL` | `http://joi:3103` | Embeddings endpoint (not yet used) |
| `POSTGRES_PASSWORD` | `spin-cycle-dev` | Application Postgres password |
| `LOG_FORMAT` | `json` (prod) / `pretty` (dev) | Log output format — `json` for Grafana Loki, `pretty` for terminal |
| `LOG_LEVEL` | `INFO` | Log level — `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `NEWSAPI_KEY` | (empty) | NewsAPI key for news search evidence (optional) |
| `SERPER_API_KEY` | (empty) | Serper key for Google search evidence (not yet implemented) |

## Port Allocation

| Port | Dev | Prod | Service |
|------|-----|------|---------|
| Base | 4500 | 3500 | FastAPI API |
| +1 | 4501 | 3501 | Temporal UI |
| +2 | 4502 | 3502 | Adminer |

## Database

Four tables in PostgreSQL, all with UUID primary keys:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `claims` | Top-level claims | text, source_url, status (pending→verified), timestamps |
| `sub_claims` | Decomposed atomic claims | text, verdict, confidence, reasoning |
| `evidence` | Research results per sub-claim | source_type (web/wikipedia/news_api), content, URL |
| `verdicts` | Overall claim verdict | verdict, confidence, reasoning_chain (JSONB) |

Relationships: `claims` → has many `sub_claims` → has many `evidence`. `claims` → has one `verdict`.

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
│   ├── llm.py                      # Shared LLM client → joi (instruct + thinking)
│   │
│   ├── utils/                      # Shared utilities
│   │   └── logging.py              # Structured logging (JSON for Loki, pretty for dev)
│   │
│   ├── api/                        # FastAPI backend
│   │   ├── app.py                  # App + lifespan
│   │   └── routes/
│   │       ├── health.py           # Health check
│   │       └── claims.py           # Claim CRUD
│   │
│   ├── agent/                      # LangGraph agents
│   │   └── research.py             # ReAct agent (DuckDuckGo + Wikipedia)
│   │
│   ├── tools/                      # Evidence gathering tools
│   │   ├── web_search.py           # DuckDuckGo
│   │   ├── wikipedia.py            # Wikipedia API
│   │   └── news_api.py             # NewsAPI
│   │
│   ├── prompts/                    # LLM prompts (heavily documented)
│   │   └── verification.py         # Decompose, Research, Judge, Synthesize
│   │
│   ├── workflows/
│   │   └── verify.py               # VerifyClaimWorkflow
│   │
│   ├── activities/
│   │   └── verify_activities.py    # 6 Temporal activities (including create_claim)
│   │
│   ├── db/
│   │   ├── models.py               # SQLAlchemy models
│   │   └── session.py              # Async DB sessions
│   │
│   └── data/
│       └── schemas.py              # Pydantic schemas
│
└── tests/
    ├── test_graph.py
    ├── test_health.py
    └── test_schemas.py
```

## What's Next

1. **Serper (Google search) tool** — biggest research quality win, SERPER_API_KEY ready in .env
2. **Extraction pipeline** — automated claim ingestion from RSS feeds via scheduled Temporal workflows
3. **Alembic migrations** — proper database schema versioning
4. **LangFuse** — self-hosted LLM observability for prompt debugging

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical deep dive, including the extraction pipeline design, database schema details, and how LangChain/LangGraph/Temporal fit together.
