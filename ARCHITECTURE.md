# Architecture

## System Context

Spin Cycle is an automated news claim verification system. The goal: take verifiable factual claims from the news, decompose them into atomic sub-claims, research real evidence using web tools, and deliver structured verdicts with full reasoning chains.

```
News Sources (RSS, APIs)
    │
    ▼
Temporal (scheduled)  →  LLM extracts claims  →  VerifyClaimWorkflow  →  Postgres
                                                                              │
                                          vedanta-systems (3100)  ←  API  ◄───┘
```

The primary intake is **automated extraction** — Temporal scheduled workflows pull articles from news feeds, the LLM extracts verifiable claims, and each claim is fed into the verification pipeline. The FastAPI backend is a **read layer** for the frontend, with a secondary `POST /claims` for manual submission.

---

## How the Stack Fits Together

There are three major technologies in play, each doing a different job. Understanding what each one handles (and doesn't handle) is key.

### LangChain (foundation layer)

LangChain is the **toolbox**. It provides:

- **`ChatOpenAI`** — the LLM client that talks to joi's OpenAI-compatible API. Every LLM call in the project goes through this class. It handles message formatting, streaming, structured output, and tool calling.
- **LangChain tools** — standardised interfaces for external services. DuckDuckGo search, Wikipedia, NewsAPI are all wrapped as LangChain tools with a common `.invoke()` / `.ainvoke()` API.
- **Message types** — `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`. These are the primitives that make up an LLM conversation.

LangChain does NOT handle orchestration, retries, scheduling, or state persistence. It's the building blocks.

**Where it's used:**
- `src/llm.py` — shared `ChatOpenAI` client configuration
- `src/tools/web_search.py` — `DuckDuckGoSearchResults` tool
- `src/tools/wikipedia.py` — custom `@tool`-decorated async function
- `src/activities/verify_activities.py` — all LLM calls use `SystemMessage`/`HumanMessage`

### LangGraph (agent framework)

LangGraph is the **agent engine**. It builds on LangChain to create state machines with:

- **Cycles**: a node can loop back to a previous node (research → evaluate → need more → research again)
- **Tool calling**: the LLM decides which tools to call, the graph executes them, feeds results back
- **State persistence**: every step reads from and writes to a typed state object

The critical pattern in Spin Cycle is the **ReAct (Reason + Act) agent**:

```
    ┌──────────┐     ┌───────┐
    │  agent   │────▶│ tools │
    │  (LLM)   │◀────│       │
    └────┬─────┘     └───────┘
         │ (no more tool calls)
         ▼
        END
```

1. LLM receives the conversation + tool definitions
2. LLM decides to call a tool → returns an `AIMessage` with `tool_calls`
3. Graph executes the tool → appends `ToolMessage` with results
4. Loop back to LLM — it now sees the tool results
5. LLM decides it has enough → returns a text response → graph ends

This is what makes the research step **agentic** — the LLM autonomously decides what to search, reads results, decides if it needs more, and adapts its strategy.

**Where it's used:**
- `src/agent/research.py` — `create_react_agent()` builds the ReAct agent for evidence gathering
- `src/agent/graph.py` — standalone `StateGraph` verification graph (currently stubbed, may be consolidated)
- `src/agent/state.py` — `VerificationState` TypedDict for the graph
- `src/agent/nodes.py` — stub node functions for the standalone graph

### Temporal (durable workflow orchestration)

Temporal is the **scheduler and reliability layer**. It handles:

- **Durable execution**: if a container crashes mid-workflow, Temporal replays from the last completed activity
- **Retries**: each activity has a `RetryPolicy` (max 3 attempts). If the LLM times out, Temporal retries just that activity
- **Timeouts**: activities have `start_to_close_timeout` (60-120s). Agent loops that run forever get killed
- **Scheduling**: extraction workflows will run on a Temporal cron schedule (every 15 min)
- **Visibility**: Temporal UI shows every workflow, its state, its history. Debug anything

The key insight: **LangGraph runs inside Temporal activities, not instead of them.**

```
Temporal Workflow
└── Activity: research_subclaim (retryable, timeout: 120s)
    └── LangGraph ReAct Agent (cycles between LLM and tools)
        ├── LLM call (via LangChain ChatOpenAI)
        ├── DuckDuckGo search (via LangChain tool)
        ├── Wikipedia search (via LangChain tool)
        ├── LLM call (sees results, decides next action)
        └── ... (until LLM decides it has enough)
```

Temporal handles the **macro orchestration** (decompose → research → judge → synthesize → store). LangGraph handles the **micro orchestration** (search → read → decide → search more).

**Where it's used:**
- `src/workflows/verify.py` — `VerifyClaimWorkflow` definition
- `src/activities/verify_activities.py` — all 6 Temporal activities (including create_claim)
- `src/worker.py` — worker entrypoint that registers workflows + activities
- `docker-compose.dev.yml` — Temporal server + Temporal UI containers

---

## The Verification Pipeline (what's working now)

A claim enters the system (via API or extraction) and goes through 6 activities, orchestrated by Temporal:

### Step 1: decompose_claim

**File:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `DECOMPOSE_SYSTEM` / `DECOMPOSE_USER`

The LLM breaks a complex claim into atomic, independently verifiable sub-claims.

```
Input:  "Bitcoin was created by Satoshi Nakamoto in 2009 and the first block was mined on January 3rd"
Output: ["Bitcoin was created by Satoshi Nakamoto in 2009",
         "The first block of Bitcoin was mined on January 3rd, 2009"]
```

Why decompose? Because each part might have a different truth value. The overall claim is "mostly true" if 2 of 3 sub-claims check out but one has the wrong date.

The prompt uses `/no_think` to disable Qwen3's chain-of-thought — we want clean JSON, not reasoning. If JSON parsing fails, we fall back to the original claim as a single sub-claim.

### Step 2: research_subclaim (the agentic part)

**File:** `src/agent/research.py` → `research_claim()`
**Called from:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `RESEARCH_SYSTEM` / `RESEARCH_USER`

This is where the LangGraph ReAct agent runs. For each sub-claim:

1. Agent receives: "Find evidence about: {sub-claim}"
2. LLM decides what to search → calls `web_search` or `wikipedia_search`
3. Tool executes the search → returns results as text
4. LLM reads results → decides if it needs more → calls another tool or stops
5. Typically: 2-3 tool calls per sub-claim, 6 agent steps, ~8 seconds

**Tools available to the agent:**
- `web_search` — DuckDuckGo search (news, fact-checks, general web). No API key needed.
- `wikipedia_search` — Wikipedia API search (established facts, background). Custom async tool with User-Agent header.

After the agent finishes, we extract evidence from the conversation:
- Each `ToolMessage` becomes an evidence record (source_type: web/wikipedia)
- The agent's final `AIMessage` is captured as an "agent_summary" (not stored in DB, used for context)

**Fallback:** If the ReAct agent fails (tool calling issues, network errors), we fall back to direct tool calls — no LLM reasoning, just search the claim text directly. Less targeted but still produces evidence.

### Step 3: judge_subclaim

**File:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `JUDGE_SYSTEM` / `JUDGE_USER`

The LLM evaluates evidence for a single sub-claim. This is NOT agentic — it's a single LLM call with structured output.

The critical constraint: **"Do NOT use your own knowledge."** The LLM must reason only from the evidence provided. This is what makes verdicts trustworthy — they're grounded in real, citable sources.

```
Input:  sub_claim = "Bitcoin was created by Satoshi Nakamoto in 2009"
        evidence = [Wikipedia excerpt, DuckDuckGo results]
Output: {"verdict": "true", "confidence": 0.95,
         "reasoning": "Multiple sources confirm..."}
```

Sub-claim verdicts: `true` | `false` | `partially_true` | `unverifiable`

If there's no evidence, we short-circuit to "unverifiable" without calling the LLM.

### Step 4: synthesize_verdict

**File:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `SYNTHESIZE_SYSTEM` / `SYNTHESIZE_USER`

The LLM combines sub-claim verdicts into an overall verdict. Also a single LLM call:

```
Input:  claim = "Bitcoin was created by Satoshi Nakamoto in 2009 and..."
        sub_verdicts = [{"verdict": "true", "confidence": 0.95}, ...]
Output: {"verdict": "true", "confidence": 0.95,
         "reasoning": "Both sub-claims are independently verified..."}
```

Why use LLM instead of averaging? Because "X happened in 2019 and cost $50M" where the event DID happen but in 2020 and cost $48M is "mostly true" — the core claim is right, details are slightly off. An LLM makes this nuance call better than math.

Overall verdicts: `true` | `mostly_true` | `mixed` | `mostly_false` | `false` | `unverifiable`

### Step 5: store_result

**File:** `src/activities/verify_activities.py`

Writes everything to Postgres:
- One `SubClaim` row per sub-claim (text, verdict, confidence, reasoning)
- One `Evidence` row per evidence item (source_type, content, URL)
- One `Verdict` row (overall verdict, confidence, reasoning_chain as JSONB)
- Updates `Claim.status` to "verified"

Evidence records with `source_type` not in the DB enum (`web`, `wikipedia`, `news_api`) are filtered out — the agent_summary doesn't get stored.

---

## The Extraction Pipeline (planned, not built)

The verification pipeline works end-to-end. The next major piece is **automated claim extraction** — getting claims into the system without manual submission.

### Design

A new Temporal workflow: `ExtractClaimsWorkflow`, scheduled on a cron (every 15 min):

```
ExtractClaimsWorkflow (cron: every 15 min)
├── fetch_articles         → pull latest from RSS feeds + news APIs
├── for each article:
│   ├── extract_claims     → LLM reads article, extracts verifiable claims
│   └── for each claim:
│       └── start VerifyClaimWorkflow (child workflow)
└── update_source_cursors  → track what we've already processed
```

### New Database Models Needed

```
source_feeds
├── id (uuid)
├── name (e.g. "BBC News - Top Stories")
├── url (RSS feed URL or API endpoint)
├── feed_type (rss | newsapi | custom)
├── enabled (bool)
├── fetch_interval_minutes (default: 15)
├── last_fetched_at
└── created_at

articles
├── id (uuid)
├── url (unique — dedup key)
├── title
├── content (full text or summary)
├── source_feed_id (fk → source_feeds)
├── published_at
├── processed_at
└── created_at
```

### Target News Sources

| Source | Type | Notes |
|--------|------|-------|
| BBC News RSS | rss | Multiple topic feeds (world, politics, science) |
| Reuters RSS | rss | Wire service — high factual density |
| AP News RSS | rss | Wire services are claim-heavy |
| The Guardian RSS | rss | UK-focused, political claims |
| NewsAPI | newsapi | Aggregator — keyword search, requires API key |
| Google News RSS | rss | Aggregated headlines from multiple sources |

RSS is the core — free, no auth, every major outlet has feeds.

### Extraction Prompt Strategy

The `extract_claims` activity sends article text to the LLM:

```
You are a fact-checker's assistant. Read this article and extract all
verifiable factual claims. Each claim should be:
- A single, atomic factual statement
- Independently verifiable (not opinion, prediction, or vague)
- Self-contained (makes sense without the article context)

Return a JSON array of objects:
[{"claim": "...", "context": "brief quote from article where this appeared"}]
```

Key decisions:
- **Atomic claims only** — "UK spent £50B on HS2" not "UK spent £50B on HS2 and cancelled the northern leg"
- **No opinions** — "The government wasted money" is not a verifiable claim
- **Context preserved** — knowing where in the article the claim appeared helps with verification
- **Dedup at insert** — same claim text (or near-duplicate) already exists → skip it

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐       ┌──────────────────┐       ┌────────────────────┐
│     claims      │       │    sub_claims     │       │     evidence       │
│─────────────────│       │──────────────────│       │────────────────────│
│ id (uuid) PK    │───┐   │ id (uuid) PK     │───┐   │ id (uuid) PK      │
│ text (text)     │   │   │ claim_id (uuid)FK │   │   │ sub_claim_id (FK) │
│ source_url      │   └──▶│ text (text)       │   └──▶│ source_type (enum)│
│ source_name     │       │ verdict (enum)    │       │ source_url        │
│ status (enum)   │       │ confidence (float)│       │ content (text)    │
│ created_at      │       │ reasoning (text)  │       │ supports_claim    │
│ updated_at      │       └──────────────────┘       │ retrieved_at      │
└────────┬────────┘                                   └────────────────────┘
         │
         │       ┌──────────────────┐
         │       │    verdicts      │
         │       │──────────────────│
         └──────▶│ id (uuid) PK     │
                 │ claim_id (FK) UQ  │
                 │ verdict (enum)    │
                 │ confidence (float)│
                 │ reasoning_chain   │
                 │   (jsonb)         │
                 │ created_at        │
                 └──────────────────┘
```

### Table: `claims`

The top-level entity. One row per claim submitted (manually or via extraction).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, default uuid4 | Unique identifier |
| `text` | `TEXT` | NOT NULL | The original claim text |
| `source_url` | `VARCHAR(2048)` | nullable | URL where the claim was found |
| `source_name` | `VARCHAR(256)` | nullable | Name of the source (e.g., "BBC News") |
| `status` | `ENUM('pending','processing','verified','flagged')` | NOT NULL, default 'pending' | Workflow state |
| `created_at` | `TIMESTAMPTZ` | default now() | When the claim was submitted |
| `updated_at` | `TIMESTAMPTZ` | default now(), on update | Last modification time |

**Relationships:**
- Has many `sub_claims` (cascade delete)
- Has one `verdict` (cascade delete)

**Status lifecycle:** `pending` → `processing` → `verified` (or `flagged`)

### Table: `sub_claims`

Atomic sub-claims decomposed from the parent claim by the LLM.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, default uuid4 | Unique identifier |
| `claim_id` | `UUID` | FK → claims.id, NOT NULL | Parent claim |
| `text` | `TEXT` | NOT NULL | The sub-claim text |
| `verdict` | `ENUM('true','false','partially_true','unverifiable')` | nullable | LLM's verdict on this sub-claim |
| `confidence` | `FLOAT` | nullable | 0.0 to 1.0 confidence score |
| `reasoning` | `TEXT` | nullable | LLM's explanation of the verdict |

**Relationships:**
- Belongs to one `claim`
- Has many `evidence` (cascade delete)

### Table: `evidence`

Individual pieces of evidence gathered by the research agent for a sub-claim.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, default uuid4 | Unique identifier |
| `sub_claim_id` | `UUID` | FK → sub_claims.id, NOT NULL | Parent sub-claim |
| `source_type` | `ENUM('web','wikipedia','news_api')` | NOT NULL | Where the evidence came from |
| `source_url` | `VARCHAR(2048)` | nullable | URL of the source (often embedded in content) |
| `content` | `TEXT` | nullable | The evidence text/excerpt |
| `supports_claim` | `BOOLEAN` | nullable | Whether this evidence supports the claim (set by judge) |
| `retrieved_at` | `TIMESTAMPTZ` | default now() | When the evidence was gathered |

**Relationships:**
- Belongs to one `sub_claim`

### Table: `verdicts`

The overall verdict for a claim, produced by the synthesize step.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, default uuid4 | Unique identifier |
| `claim_id` | `UUID` | FK → claims.id, UNIQUE, NOT NULL | Parent claim (one verdict per claim) |
| `verdict` | `ENUM('true','mostly_true','mixed','mostly_false','false','unverifiable')` | NOT NULL | Overall verdict |
| `confidence` | `FLOAT` | NOT NULL | 0.0 to 1.0 confidence score |
| `reasoning_chain` | `JSONB` | nullable | Array of reasoning strings from sub-claim judgments |
| `created_at` | `TIMESTAMPTZ` | default now() | When the verdict was produced |

**Relationships:**
- Belongs to one `claim` (one-to-one via unique constraint)

### Enums

| Enum Name | Values | Used By |
|-----------|--------|---------|
| `claim_status` | pending, processing, verified, flagged | claims.status |
| `sub_claim_verdict` | true, false, partially_true, unverifiable | sub_claims.verdict |
| `evidence_source_type` | web, wikipedia, news_api | evidence.source_type |
| `verdict_type` | true, mostly_true, mixed, mostly_false, false, unverifiable | verdicts.verdict |

### ORM Details

All models use SQLAlchemy 2.0 declarative base (`src/db/models.py`):
- UUIDs via `sqlalchemy.dialects.postgresql.UUID(as_uuid=True)`
- Async engine + sessionmaker via `asyncpg` (`src/db/session.py`)
- Tables auto-created on app startup via `Base.metadata.create_all` in the FastAPI lifespan
- No Alembic migrations yet — table changes require dropping and recreating

---

## LLM Integration

### Models

Two models running on joi via llama.cpp, each serving a different role:

| Port | Model | Role | Used By |
|------|-------|------|--------|
| `:3101` | Qwen3-VL-30B-A3B-Instruct | Fast structured output | decompose, synthesize |
| `:3102` | Qwen3-VL-30B-A3B-Thinking | Chain-of-thought reasoning | research, judge |
| `:3103` | (embeddings — not yet used) | Semantic similarity | planned: evidence caching |

Same base architecture (30B params, 3B active MoE), different fine-tunes. The instruct model uses `/no_think` for clean JSON. The thinking model produces `<think>...</think>` blocks that are stripped before parsing.

### Connection Path

```
Container → Docker DNS (127.0.0.11) → systemd-resolved → Tailscale MagicDNS → joi (100.70.38.37)
```

Short hostnames (`joi`) work because Docker is configured to use systemd-resolved as upstream DNS, and Tailscale registers machine names there.

### Configuration

All LLM calls go through `src/llm.py`:

```python
from langchain_openai import ChatOpenAI

def get_llm(temperature=0.1):           # Instruct — fast, structured
    return ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",     # :3101
        model="Qwen3-VL-30B-A3B-Instruct",
        temperature=temperature,
    )

def get_reasoning_llm(temperature=0.2):  # Thinking — slower, better reasoning
    return ChatOpenAI(
        base_url=f"{LLAMA_REASONING_URL}/v1",  # :3102
        model="Qwen3-VL-30B-A3B-Thinking",
        temperature=temperature,
    )
```

### Prompt Design

All prompts live in `src/prompts/verification.py` with extensive inline documentation explaining:
- What each prompt does and why it's designed that way
- The `/no_think` token and when to use it
- Example inputs and outputs
- Design constraints (e.g., "Do NOT use your own knowledge")

Four prompt pairs (system + user):
1. `DECOMPOSE_SYSTEM` / `DECOMPOSE_USER` — break claims into sub-claims
2. `RESEARCH_SYSTEM` / `RESEARCH_USER` — guide the research agent
3. `JUDGE_SYSTEM` / `JUDGE_USER` — evaluate evidence for a sub-claim
4. `SYNTHESIZE_SYSTEM` / `SYNTHESIZE_USER` — combine sub-verdicts

---

## API Layer

### FastAPI Application (`src/api/app.py`)

The app uses a lifespan context manager for startup/shutdown:
- **Startup**: creates DB tables (`Base.metadata.create_all`), connects to Temporal
- **Shutdown**: disposes DB engine
- Temporal client stored in `app.state.temporal` for route access

### Endpoints

| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| `GET` | `/` | Root info | — | `{service, version, status}` |
| `GET` | `/health` | Health check | — | `{status, service, version}` |
| `POST` | `/claims` | Submit a claim | `ClaimSubmit` | `ClaimResponse` (201) |
| `GET` | `/claims/{id}` | Get claim with verdict | — | `VerdictResponse` |
| `GET` | `/claims` | List claims | `?status=&limit=&offset=` | `ClaimListResponse` |

### Pydantic Schemas (`src/data/schemas.py`)

| Schema | Purpose | Key Fields |
|--------|---------|------------|
| `ClaimSubmit` | POST request body | `text` (required, non-empty), `source` (optional URL), `source_name` (optional) |
| `ClaimResponse` | POST response | `id`, `text`, `status`, `created_at` |
| `SubClaimResponse` | Sub-claim in verdict | `text`, `verdict`, `confidence`, `reasoning`, `evidence_count` |
| `VerdictResponse` | Full claim detail | All claim fields + `verdict`, `confidence`, `sub_claims[]` |
| `ClaimListResponse` | Paginated list | `claims[]`, `total`, `limit`, `offset` |

### Claim Lifecycle via API

```
POST /claims {"text": "..."}
  → Insert Claim (status: pending)
  → Start VerifyClaimWorkflow in Temporal
  → Return {id, status: "pending"}

[Temporal runs pipeline: decompose → research → judge → synthesize → store]

GET /claims/{id}
  → Returns {status: "verified", verdict: "true", confidence: 0.95, sub_claims: [...]}
```

---

## Network Architecture

### Docker Containers (dev)

```
spin-cycle-dev-api               :4500  ← FastAPI (hot reload)
spin-cycle-dev-worker                   ← Temporal worker (LangGraph + activities)
spin-cycle-dev-temporal                 ← Temporal server (gRPC :7233, internal)
spin-cycle-dev-temporal-ui       :4501  ← Temporal workflow dashboard
spin-cycle-dev-postgres                 ← Application Postgres (internal)
spin-cycle-dev-temporal-postgres        ← Temporal metadata Postgres (internal)
spin-cycle-dev-adminer           :4502  ← Postgres web UI (Dracula theme)
```

### Port Allocation

| Port | Dev | Prod | Service |
|------|-----|------|---------|
| Base | 4500 | 3500 | FastAPI API |
| +1 | 4501 | 3501 | Temporal UI |
| +2 | 4502 | 3502 | Adminer (Postgres UI) |

### Networks

- `spin-cycle-dev` / `spin-cycle-prod` — internal bridge network
- `luv-dev` / `luv-prod` — external network shared with vedanta-systems for cross-project access

### External Services

- `joi:3101` — LLM chat/vision API (llama.cpp, via Tailscale)
- `joi:3102` — LLM thinking/reasoning API (llama.cpp, via Tailscale)
- `joi:3103` — LLM embeddings API (llama.cpp, via Tailscale)
- DuckDuckGo — web search (no API key)
- Wikipedia API — factual lookups (no API key)
- NewsAPI — news search (requires key, optional)

---

## Logging & Observability

### Architecture

Spin Cycle uses **structured JSON logging** designed for Grafana Loki, matching the logging conventions established in found-footy:

```
Container stdout → Docker json-file → Promtail → Loki → Grafana
```

Every log line is a JSON object with consistent fields:

```json
{"ts":"2025-01-15T12:00:00.123Z","level":"INFO","module":"judge","action":"done","msg":"Sub-claim judged","sub_claim":"Bitcoin was created by...","verdict":"true","confidence":0.95}
```

In development (`LOG_FORMAT=pretty`), logs are human-readable:

```
I [JUDGE     ] done: Sub-claim judged | sub_claim=Bitcoin was created by... verdict=true confidence=0.95
```

### Core Module: `src/utils/logging.py`

| Component | Purpose |
|-----------|---------|
| `StructuredFormatter` | JSON formatter for Loki. Strips Temporal context dicts. Pretty mode for dev. |
| `StructuredLogger` | Singleton (`log`) with `.info()`, `.warning()`, `.error()`, `.debug()` methods |
| `configure_logging()` | Called once at startup. Sets format, level, silences noisy loggers |
| `get_logger()` | Returns a fallback stdlib logger for infrastructure code |

### Usage Pattern

Every log call follows the same signature: `log.level(logger, module, action, msg, **kwargs)`

```python
from src.utils.logging import log

# In Temporal activities — use activity.logger for proper Temporal context
log.info(activity.logger, "decompose", "start", "Decomposing claim",
         claim_id=claim_id, claim=claim_text[:80])

# In Temporal workflows — use workflow.logger
log.info(workflow.logger, "workflow", "started", "Verification started",
         claim_id=claim_id)

# In infrastructure code — use get_logger() fallback
from src.utils.logging import get_logger
logger = get_logger()
log.info(logger, "worker", "ready", "Worker listening", task_queue="spin-cycle-verify")
```

### Standard Fields

Every log line has these fields, which Promtail promotes to Loki labels:

| Field | Description | Example |
|-------|-------------|---------|
| `ts` | ISO 8601 UTC timestamp | `2025-01-15T12:00:00.123Z` |
| `level` | Log level | `INFO`, `WARNING`, `ERROR`, `DEBUG` |
| `module` | Source module | `decompose`, `research`, `judge`, `synthesize`, `store`, `workflow`, `worker`, `api`, `claims`, `tools`, `db`, `llm` |
| `action` | What happened | `start`, `done`, `failed`, `parse_failed`, `no_evidence`, `fallback_start` |
| `msg` | Human-readable message | `"Claim decomposed"` |

Plus arbitrary context kwargs: `claim_id`, `verdict`, `confidence`, `evidence_count`, `error`, `error_type`, etc.

### Action Naming Convention

| Suffix | Meaning |
|--------|---------|
| `*_start` | Beginning of an operation |
| `*_done` | Successful completion |
| `*_failed` | Error or failure |
| `*_skipped` | Intentionally skipped |
| `*_fallback` | Falling back to alternative path |

### Grafana Loki Queries

```logql
# All errors
{project="spin-cycle"} | json | level="ERROR"

# Track a single claim end-to-end
{project="spin-cycle"} | json | claim_id="<uuid>"

# All workflow starts
{project="spin-cycle"} | json | action=~".*start"

# Research agent failures
{project="spin-cycle"} | json | module="research" action="agent_failed"

# Judge verdicts only
{project="spin-cycle"} | json | module="judge" action="done"

# Tool invocations
{project="spin-cycle"} | json | module="tools"

# LLM responses (debug level)
{project="spin-cycle"} | json | action="llm_response"
```

### Noisy Logger Suppression

Third-party libraries are silenced at WARNING to keep logs clean:

| Logger | Level | Why |
|--------|-------|-----|
| `temporalio.worker` | WARNING | Heartbeats, replay logs |
| `temporalio.client` | WARNING | Connection pool noise |
| `langchain`, `langchain_core`, `langchain_openai` | WARNING | LLM request/response at DEBUG |
| `langgraph` | WARNING | Graph transition logs |
| `httpx`, `httpcore`, `urllib3` | WARNING | HTTP request logs |
| `sqlalchemy` | WARNING | Query logs (engine echo=False) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_FORMAT` | `json` | `json` for Loki, `pretty` for development |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

In docker-compose.dev.yml, `LOG_FORMAT` defaults to `pretty`. In docker-compose.yml (prod), it defaults to `json`.

### Promtail Integration

The monitor stack at `~/workspace/monitor/` runs Promtail, which:

1. Discovers spin-cycle containers via Docker socket (`docker_sd_configs`)
2. Extracts `project`, `environment`, `service` from container names
3. Parses JSON log lines from `spin-cycle-*` containers
4. Promotes `level`, `module`, `action` to native Loki labels
5. Ships everything to Loki for queryable, filterable log storage

Config: `~/workspace/monitor/promtail/promtail.yml`

---

## Implementation Status

### What's Working (end-to-end verified)

| Component | Status | Details |
|-----------|--------|---------|
| Docker infrastructure | **Done** | 7 containers, health checks, volume persistence |
| PostgreSQL schema | **Done** | 4 tables: claims, sub_claims, evidence, verdicts |
| FastAPI API | **Done** | POST/GET claims, health check, lifespan management |
| Temporal workflow | **Done** | VerifyClaimWorkflow with 6 activities (including create_claim), retry policies |
| Temporal worker | **Done** | Registers workflow + activities, structured logging configured |
| `decompose_claim` | **Done** | LLM decomposes claims into sub-claims |
| `research_subclaim` | **Done** | LangGraph ReAct agent with DuckDuckGo + Wikipedia |
| `judge_subclaim` | **Done** | LLM evaluates evidence, returns structured verdict |
| `synthesize_verdict` | **Done** | LLM combines sub-verdicts into overall verdict |
| `store_result` | **Done** | Writes all results to Postgres |
| Prompts | **Done** | All 4 prompt pairs documented in `src/prompts/verification.py` |
| LLM connectivity | **Done** | Dual-model: instruct (:3101) + thinking (:3102) on joi |
| Logging | **Done** | Structured JSON logging via `src/utils/logging.py`, Promtail → Loki → Grafana |
| Tests | **Done** | Graph compilation, health endpoint, schema validation |

### What's Planned

See [ROADMAP.md](ROADMAP.md) for the full prioritised improvement plan. Key next steps:

| Component | Status | Details |
|-----------|--------|--------|
| Serper (Google search) tool | **Next** | Biggest research quality win. SERPER_API_KEY in .env, needs `src/tools/serper.py` |
| Alembic migrations | **Next** | Unblocks all future schema changes. Dependency installed, no init |
| Source credibility scoring | **Planned** | Tier system for evidence sources (Reuters > blog) |
| Calibration test suite | **Planned** | 100+ known claims, measure accuracy and confidence calibration |
| RSS feed monitoring | **Planned** | `source_feeds` + `articles` tables, Temporal cron workflow |
| Claim extraction pipeline | **Planned** | ExtractClaimsWorkflow, LLM reads articles → extracts claims |
| Parallel sub-claim processing | **Planned** | Process sub-claims concurrently instead of sequentially |
| LangFuse integration | **Planned** | Self-hosted LLM observability |

---

## Testing & Debugging

### Submitting Claims via API

```bash
# Submit a claim
curl -s -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "The Great Wall of China is visible from space"}' | python3 -m json.tool

# The response includes the claim ID
# { "id": "abc123...", "text": "...", "status": "pending" }

# Wait ~30s for the pipeline to finish, then fetch the result
curl -s http://localhost:4500/claims/{id} | python3 -m json.tool

# List claims with optional filters
curl -s 'http://localhost:4500/claims?status=verified&limit=10' | python3 -m json.tool
```

### Submitting Claims via Temporal UI

The Temporal UI at http://localhost:4501 lets you start workflows directly:

1. Click **Start Workflow** (top right)
2. Fill in:
   - **Workflow Type**: `VerifyClaimWorkflow`
   - **Workflow ID**: any unique string (e.g. `manual-test-1`)
   - **Task Queue**: `spin-cycle-verify`
   - **Input**: `[null, "The claim text you want to verify"]`
3. The first argument is `null` — the workflow creates the claim record in the database automatically. (If you already have a claim ID from the API, pass it as a string instead of `null`.)

From the Temporal UI you can also:
- **Inspect running workflows** — see each activity's input/output/duration in the Event History tab
- **Debug failures** — failed activities show the full stack trace and retry attempts
- **Terminate or cancel** workflows that are stuck

### Watching Worker Logs

```bash
# Stream worker logs — shows every step of the pipeline in real time
docker logs -f spin-cycle-dev-worker

# With LOG_FORMAT=pretty (default in dev), output looks like:
# I [WORKER    ] starting: Connecting to Temporal | temporal_host=spin-cycle-dev-temporal:7233 task_queue=spin-cycle-verify
# I [WORKER    ] ready: Worker listening | task_queue=spin-cycle-verify activity_count=6 workflow_count=1
# I [CREATE    ] start: Creating claim record | claim=The Great Wall of China is visible from ...
# I [DECOMPOSE ] start: Decomposing claim into sub-claims | claim=The Great Wall of China is visible from ...
# I [DECOMPOSE ] done: Claim decomposed | num_sub_claims=1
# I [RESEARCH  ] start: Starting research agent | sub_claim=The Great Wall of China is visible from space
# I [RESEARCH  ] done: Research agent complete | evidence_count=3 agent_steps=8
# I [JUDGE     ] done: Sub-claim judged | verdict=false confidence=0.95
# I [SYNTHESIZE] done: Overall verdict synthesized | verdict=false confidence=0.95
# I [STORE     ] done: Result stored in database | claim_id=abc123... sub_claims=1

# With LOG_FORMAT=json (default in prod), output is JSON for Loki:
# {"ts":"2025-01-15T12:00:00.123Z","level":"INFO","module":"decompose","action":"done","msg":"Claim decomposed","num_sub_claims":2}
```

### Inspecting the Database

Adminer is available at http://localhost:4502:
- **System**: PostgreSQL
- **Server**: `postgres`
- **Username**: `spincycle`
- **Password**: `spincycle`
- **Database**: `spincycle`

Key queries:
```sql
-- See all claims and their status
SELECT id, text, status, overall_verdict, created_at FROM claims ORDER BY created_at DESC;

-- See sub-claims and their verdicts
SELECT sc.text, v.verdict, v.confidence, v.reasoning
FROM sub_claims sc
JOIN verdicts v ON v.sub_claim_id = sc.id
WHERE sc.claim_id = '<claim-id>';

-- See evidence collected for a sub-claim
SELECT source_type, source_url, snippet, relevance_score
FROM evidence
WHERE sub_claim_id = '<sub-claim-id>'
ORDER BY relevance_score DESC;
```

### Running Unit Tests

```bash
# Run from inside the API container
docker exec -it spin-cycle-dev-api pytest -v

# Or locally (needs a running Postgres and env vars set)
pytest -v
```

---

## Project Structure

```
spin-cycle/
├── docker-compose.yml              # Production compose (3500-3502)
├── docker-compose.dev.yml          # Development compose (4500-4502)
├── Dockerfile                      # Production image
├── Dockerfile.dev                  # Dev image (hot reload, volume mount)
├── pyproject.toml                  # Python project config
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
├── .env.example                    # Environment template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── worker.py                   # Temporal worker entrypoint
│   ├── llm.py                      # Shared LLM client (ChatOpenAI → joi)
│   │
│   ├── utils/                      # Shared utilities
│   │   └── logging.py              # Structured logging (JSON for Loki, pretty for dev)
│   │
│   ├── api/                        # FastAPI backend
│   │   ├── app.py                  # App + lifespan (DB + Temporal init)
│   │   └── routes/
│   │       ├── health.py           # GET / and GET /health
│   │       └── claims.py          # POST + GET claims
│   │
│   ├── agent/                      # LangGraph agents
│   │   ├── research.py             # ReAct research agent (DuckDuckGo + Wikipedia)
│   │   ├── graph.py                # Standalone verification graph (stubbed)
│   │   ├── nodes.py                # Graph node functions (stubbed)
│   │   └── state.py                # VerificationState TypedDict
│   │
│   ├── tools/                      # LangChain tools for the agent
│   │   ├── web_search.py           # DuckDuckGo search wrapper
│   │   ├── wikipedia.py            # Wikipedia API with @tool decorator
│   │   └── news_api.py             # NewsAPI client (requires key)
│   │
│   ├── prompts/                    # All LLM prompts with documentation
│   │   └── verification.py         # Decompose, Research, Judge, Synthesize
│   │
│   ├── workflows/                  # Temporal workflow definitions
│   │   └── verify.py               # VerifyClaimWorkflow
│   │
│   ├── activities/                 # Temporal activity implementations
│   │   └── verify_activities.py    # All 6 verification activities (including create_claim)
│   │
│   ├── db/                         # Database layer
│   │   ├── models.py               # SQLAlchemy models (Claim, SubClaim, Evidence, Verdict)
│   │   └── session.py              # Async engine + session factory
│   │
│   └── data/                       # Data schemas
│       └── schemas.py              # Pydantic request/response models
│
├── scripts/
│   └── init_db.py                  # Database initialisation script
│
└── tests/
    ├── test_graph.py               # Graph compiles + has expected nodes
    ├── test_health.py              # Health endpoint tests
    └── test_schemas.py             # Schema validation tests
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >=0.2.0 | Agent state machine framework (ReAct agent) |
| `langchain` | >=0.3.0 | Foundation: message types, tool interfaces |
| `langchain-openai` | >=0.2.0 | `ChatOpenAI` client for joi's API |
| `langchain-community` | >=0.3.0 | `DuckDuckGoSearchResults` tool |
| `temporalio` | >=1.7.0 | Workflow orchestration, workers, activities |
| `fastapi` | >=0.115.0 | REST API framework |
| `uvicorn` | >=0.32.0 | ASGI server |
| `pydantic` | >=2.0 | Request/response validation |
| `sqlalchemy` | >=2.0 | Async ORM (PostgreSQL) |
| `asyncpg` | >=0.30.0 | Async PostgreSQL driver |
| `alembic` | >=1.14.0 | Database migrations (not yet initialised) |
| `httpx` | >=0.28.0 | Async HTTP client (Wikipedia, NewsAPI) |
| `ddgs` | >=7.0.0 | DuckDuckGo search backend |
| `python-dotenv` | >=1.0.0 | .env file loading |
