# Roadmap

Where spin-cycle is, where it needs to go, and what each improvement actually does.

---

## What We Have (v0.2)

A working end-to-end claim verification pipeline with tree decomposition:
- Manual claim submission via API
- LLM decomposes claims into **hierarchical trees** (groups + leaves)
- Leaves researched + judged **in parallel** via `asyncio.gather`
- Groups get intermediate synthesis before final verdict
- LangGraph research agent gathers evidence with dynamically loaded tools
- **Source quality filtering** — domain blocklist (~40 domains) silently drops Reddit, Quora, social media, content farms from all search results
- Thinking model evaluates evidence and renders verdicts
- **Importance-weighted synthesis** — verdicts weighed by significance, not count
- Results stored in Postgres with full tree structure (parent_id, is_leaf) and reasoning chains
- Top-level synthesis reasoning exposed in API responses
- Temporal orchestrates everything with retries and durability (7 activities, 1 workflow)
- Production-grade structured logging (JSON for Loki, pretty for dev)
- LLM max_tokens configured to prevent truncated output (2048 instruct, 4096 reasoning)

**Search tools (env-var gated — set the key to enable):**
- **SearXNG** (self-hosted meta-search, free, aggregates 70+ engines) — `SEARXNG_URL`
- **Serper** (Google index via API, 2,500 one-time free) — `SERPER_API_KEY`
- **Brave Search** (independent index, ~1k queries from $5/mo credit) — `BRAVE_API_KEY`
- **DuckDuckGo** (free fallback, always available, no key needed)
- **Wikipedia** (always available, no key needed)
- **Page fetcher** (reads full page text from URLs, always available)

**What's missing to make this legit:**

---

## Phase 1: Research Quality

The single biggest lever. The verdict is only as good as the evidence.

### 1.1 — Multi-Source Search (DONE)

**Status:** ✅ Implemented

Search tools are dynamically loaded based on configured API keys / services:
- **SearXNG** — Self-hosted meta-search engine. Aggregates Google, Bing, DuckDuckGo, Brave, and dozens more. Free, unlimited, runs as a Docker container in the stack.
- **Serper** — Google results via Serper API. Reliable paid option.
- **Brave Search** — Independent search index. Finds things Google misses.
- **DuckDuckGo** — Free fallback, always available.
- **Wikipedia** — Established facts, always available.
- **Page fetcher** — Fetches and extracts full text from URLs. Lets the agent actually read articles instead of relying on snippets.

The agent uses these tools in combination — it decides which tools to call, reads results, and loops until it has enough evidence.

### ~~1.1b — SearXNG Meta-Search~~ (Merged into 1.1)

SearXNG is now the primary search tool, running self-hosted in the Docker stack.

### 1.2 — News API Integration

**Why:** NewsAPI gives structured access to 150,000+ news sources with date filtering. Critical for claims about recent events where web search might not surface the right articles yet.

**What:** Implement `src/tools/news_api.py` (file exists but isn't wired up). Add date-aware search — if the claim references a date or recent event, bias towards recent articles.

**Effort:** Small. The file stub already exists.

### 1.3 — Source Credibility Scoring

**Why:** Right now sources are filtered (junk domains blocked) but all remaining sources are weighted equally. A Reuters article and a lesser-known outlet both count as "web" evidence. The judge prompt says "reliable sources count more" but the model has to infer credibility from the URL alone.

**Status:** Partially done — domain blocklist filtering is implemented in `src/tools/source_filter.py`, wired into all search tools and the page fetcher. The RESEARCH_SYSTEM prompt explicitly lists acceptable and forbidden source types. What's missing is the **tiered scoring** system below.

**What:** Add a source credibility tier system:

| Tier | Sources | Weight |
|------|---------|--------|
| **1 — Wire services** | Reuters, AP, AFP | Highest |
| **2 — Major outlets** | BBC, NYT, Guardian, WSJ, etc. | High |
| **3 — Official records** | Government sites, court records, legislatures | High |
| **4 — Quality press** | Reputable national/regional outlets | Medium |
| **5 — Other** | Blogs, forums, unknown domains | Low |

Tag each evidence item with its tier. Pass the tier to the judge so it can weigh sources properly.

**Effort:** Medium. Need a domain → tier mapping (can start with top 200 domains) and add `credibility_tier` to the Evidence model.

### 1.4 — Claim-Specific Search Strategy

**Why:** Different claims need different research approaches. A claim about GDP needs official statistics. A claim about what someone said needs the original transcript. A claim about a scientific finding needs the paper.

**What:** Add a pre-research step where the thinking model classifies the claim type and recommends search strategies:

| Claim Type | Strategy |
|-----------|----------|
| Statistical | Search for official reports, government data |
| Quote/attribution | Search for transcript, video, original speech |
| Scientific | Search for paper, peer review, scientific consensus |
| Historical | Wikipedia + academic sources |
| Current event | News API with date filters |
| Policy/legislation | Government websites, legislative records |

**Effort:** Medium. New activity `classify_claim` before research, adds ~5s per claim.

---

## Phase 2: Content Ingestion

Right now you manually POST claims. To actually audit politicians and pundits at scale, you need automated intake.

### 2.1 — RSS Feed Monitoring

**Why:** Most news outlets publish RSS feeds. This is the simplest way to automatically pull articles as they're published.

**What:**
- New `source_feeds` table: feed URL, outlet name, last polled timestamp
- New `articles` table: title, body, source URL, publication date, feed ID
- Temporal scheduled workflow: poll feeds every N minutes, store articles
- Admin API endpoints to manage feeds

**Effort:** Medium. Database tables + a simple Temporal cron workflow.

### 2.2 — Claim Extraction from Articles

**Why:** Articles contain claims but they're buried in prose. You need the LLM to read the article and pull out the verifiable factual assertions.

**What:**
- New `ExtractClaimsWorkflow` — takes an article, uses the LLM to extract claims
- `extract_claims` activity: LLM reads article body, returns list of verifiable claims with their positions in the text (character offsets for frontend highlighting)
- Each extracted claim auto-queues a `VerifyClaimWorkflow`
- Link claims back to articles (many-to-many: same claim can appear in multiple articles)

**Effort:** Large. This is a new workflow + activity + prompt + database relationships. But it's the backbone for automated operation.

### 2.3 — Speech & Debate Transcripts

**Why:** Politicians say things in speeches, press conferences, debates, and parliament. These are often the primary source of claims worth checking.

**What:**
- Ingest transcripts from public sources (C-SPAN, Hansard, Congressional Record, press briefing transcripts)
- Same extraction pipeline as articles — LLM reads transcript, extracts claims
- Track speaker attribution — WHO said it, not just what was said
- Add `speaker` field to claims table

**Effort:** Medium-Large. Transcript sources vary in format. Speaker attribution requires NLP or structured transcripts.

### 2.4 — Social Media Monitoring (Later)

**Why:** Viral claims often originate or accelerate on social media before reaching news.

**What:** Monitor specific accounts (politicians, pundits) for claims. This is politically and legally sensitive — needs careful scoping.

**Effort:** Large. API access is increasingly restricted. Better to start with public transcripts and RSS.

---

## Phase 3: Accuracy & Trust

For this to be legitimate, people need to trust the verdicts. Trust comes from transparency, consistency, and demonstrable accuracy.

### 3.1 — Calibration Testing

**Why:** How accurate is the system actually? You need to measure it. Without benchmarks, you're guessing.

**What:**
- Build a test set of 100+ claims with known ground truth (use existing fact-check databases: ClaimBuster, MultiFC, FEVER dataset)
- Run them through the pipeline periodically
- Track accuracy metrics: precision, recall, F1 by verdict category
- Track confidence calibration: when the system says 0.8 confidence, is it right 80% of the time?

**Effort:** Medium. The pipeline already works — this is about building the test harness and running it.

### 3.2 — Source Citation Quality

**Why:** Right now the `reasoning` field explains the verdict but doesn't always directly quote the evidence. A credible fact-check should cite specific passages.

**What:**
- Update the judge prompt to require direct quotes from evidence
- Store the relevant quote/passage alongside each evidence item
- Add `relevance_score` computation (currently the column exists but isn't populated meaningfully)
- Frontend can show: "Source: Reuters — 'BYD sold 4.2 million EVs in 2025, surpassing Tesla's 3.8 million'"

**Effort:** Small-Medium. Mostly prompt engineering + schema tweaks.

### 3.3 — Verdict Audit Trail

**Why:** Every verdict should be fully reproducible. If someone questions a verdict, you should be able to show exactly what evidence was found, how it was weighed, and why the conclusion was reached.

**What:**
- Store the full LLM conversation for each step (not just the final output)
- For the thinking model: store the `<think>` block alongside the verdict
- Add timestamps to every step for performance analysis
- API endpoint to retrieve the full audit trail for any claim

**Effort:** Medium. Need to expand what's stored in the DB and add audit-specific endpoints.

### 3.4 — Human Review Loop

**Why:** Low-confidence verdicts and controversial claims should be flagged for human review before being published. The system should assist humans, not replace them for borderline cases.

**What:**
- Add `needs_review` flag on claims (triggered when confidence < 0.6 or verdict is "unverifiable")
- Review queue API: list claims needing review, accept/override verdicts
- Track human overrides for calibration (did the system get it wrong? why?)
- Eventually: use human feedback to improve prompts

**Effort:** Medium. Mostly API endpoints + frontend work.

---

## Phase 4: Scale & Performance

### 4.1 — Evidence Caching

**Why:** Multiple claims often reference the same facts. If 10 claims mention "BYD overtook Tesla in EV sales," you don't need to web-search that 10 times.

**What:**
- Cache search results by query (with TTL — evidence goes stale)
- Cache at the evidence level: if we already have high-quality evidence for a sub-claim, skip research
- Use embeddings (joi:3103) to find semantically similar sub-claims that have already been researched

**Effort:** Medium. Need a caching layer (Redis or just Postgres with TTL) and embedding similarity search.

### 4.2 — Parallel Sub-Claim Processing (DONE)

**Status:** ✅ Implemented

Sub-claims are now processed in parallel using `asyncio.gather` within the Temporal workflow. The tree decomposition enables natural parallelism:

- All leaf nodes within a group are researched + judged concurrently
- All top-level nodes are processed in parallel
- Group synthesis waits for children to complete, then runs
- Temporal handles the per-activity retries and timeouts (research: 360s, judge: 120s, synthesis: 60s)

Additionally, tree decomposition groups related sub-claims, enabling intermediate synthesis that produces more nuanced verdicts than flat list processing.

### 4.3 — LangFuse Observability

**Why:** You can't improve what you can't measure. LangFuse gives you traces for every LLM call — latency, token usage, prompt/response pairs, tool calls.

**What:** Self-hosted LangFuse instance, instrument all LLM calls with LangFuse callbacks. Track cost, latency, and quality metrics per claim.

**Effort:** Small-Medium. LangFuse has LangChain integration. Mostly config.

### 4.4 — Alembic Migrations

**Why:** Right now we use `Base.metadata.create_all()` which can only create tables — it can't alter them. Any schema change requires dropping and recreating tables (losing data).

**What:** Initialize Alembic, create initial migration from current schema, use migrations for all future changes.

**Effort:** Small. `alembic init`, generate migration from existing models, swap out `create_all()`.

---

## Phase 5: Product & Distribution

### 5.1 — Article Highlighting UI

The core product vision: read an article and see claims highlighted inline, color-coded by verdict.

```
"The Prime Minister said unemployment fell to [3.2%]{.verdict-true} last quarter,
 while the opposition claimed the government [spent £50 billion on HS2]{.verdict-mostly-true}
 before [cancelling the northern leg]{.verdict-true}."
```

- Green: true / mostly true
- Yellow: mixed / partially true
- Red: false / mostly false
- Grey: unverifiable

Tap a highlighted claim → panel slides out showing:
- Verdict + confidence
- Sub-claim breakdown
- Evidence with source links
- Full reasoning chain

**Requires:** Phase 2.2 (claim extraction with position tracking) and a frontend.

### 5.2 — Public API

**Why:** Other developers, journalists, and researchers should be able to submit claims programmatically.

**What:**
- API key authentication
- Rate limiting
- Webhook notifications when verification completes
- Bulk submission endpoint
- OpenAPI docs (already get this free from FastAPI)

**Effort:** Medium. Auth + rate limiting + webhooks.

### 5.3 — Speaker Profiles

**Why:** Track accuracy over time per person. "This politician's claims have been verified 847 times. 62% true, 18% mostly true, 12% mixed, 8% false."

**What:**
- `speakers` table: name, role, party, bio
- Link claims to speakers
- Aggregate stats per speaker
- Trending: who's making the most false claims this week?

**Effort:** Medium. DB table + aggregation queries + API endpoints.

### 5.4 — Notification System

**Why:** Users want to know when a claim they care about has been verified, or when a politician they follow makes a new claim.

**What:**
- Follow speakers or topics
- Push notifications / email when new verdicts are published
- Daily/weekly digest of verdicts

**Effort:** Large. Needs user accounts, notification infrastructure.

---

## Priority Order

What to build next, in order of impact:

| Priority | Item | Why |
|----------|------|-----|
| **1** | Alembic migrations | Unblocks all future schema changes |
| **2** | Source credibility scoring | Tiered weighting (basic filtering already done) |
| **3** | Calibration test suite | Can't improve without measuring |
| **4** | RSS feed monitoring | First step toward automated intake |
| **5** | Claim extraction from articles | Enables fully automated pipeline |
| **6** | Evidence caching | Reduces redundant work at scale |
| **7** | LangFuse observability | Visibility into LLM performance |
| **8** | Speaker profiles | Product differentiation |
| **9** | Article highlighting UI | Core product experience |
| **10** | Human review loop | Trust + quality assurance |
| **11** | Public API | Distribution |
| **12** | Speech transcripts | Expand beyond articles |

---

## Non-Goals (For Now)

Things that sound useful but aren't worth building yet:

- **Real-time verification** — verifying claims as they're spoken in a live broadcast. The pipeline takes ~2 minutes per claim. Real-time would require a fundamentally different architecture. Batch processing of transcripts after the fact is more practical.
- **Misinformation detection** — spin-cycle verifies specific claims, it doesn't detect misinformation patterns or narrative analysis. That's a different (harder) problem.
- **Opinion analysis** — "The economy is doing terribly" is an opinion, not a verifiable claim. The decomposer should filter these out (it mostly does). We verify facts, not feelings.
- **Image/video verification** — fake images and deepfakes are a real problem but require different tools (reverse image search, forensic analysis). Out of scope for now, though the Qwen3-VL model is multimodal and could eventually help here.
- **Multi-language** — all prompts are English. Supporting other languages means translating prompts, handling non-English search results, and dealing with non-English news sources. Important eventually, but not yet.
