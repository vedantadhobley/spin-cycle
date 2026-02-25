# Roadmap

Where spin-cycle is, where it needs to go, and what each improvement actually does.

---

## What We Have (v0.4)

A working end-to-end claim verification pipeline with **flat decomposition + thesis-aware synthesis**:
- Manual claim submission via API
- LLM decomposes claims into **atomic facts + thesis** in one call — captures the speaker's intent (thesis, structure, key_test)
- Atomic sub-claims researched + judged **in parallel batches** (MAX_CONCURRENT=2) via `asyncio.gather`
- **Thesis-aware synthesis** — evaluates whether the speaker's argument survives the evidence, not just whether a majority of facts are true
- **Unified 6-level verdict scale**: `true | mostly_true | mixed | mostly_false | false | unverifiable` with spirit-vs-substance guidance
- **Single synthesis activity** uses the thesis as primary rubric when available
- LangGraph ReAct research agent gathers evidence with dynamically loaded tools (temperature=0.0 for deterministic queries)
- **Source quality filtering** — domain blocklist (~40 domains) silently drops Reddit, Quora, social media, content farms from all search results
- Thinking model evaluates evidence and renders verdicts (instruct model for everything else)
- **Importance-weighted synthesis** — verdicts weighed by significance, not count
- **Date-aware prompts** — all prompts include `Today's date: {current_date}` so the LLM references current data
- Results stored in Postgres with sub-claims, evidence, and reasoning chains
- Top-level synthesis reasoning + nuance exposed in API responses
- Temporal orchestrates everything with retries and durability (5 activities, 1 workflow)
- Production-grade structured JSON logging (for Grafana Loki, pretty format for dev)
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

**What:** Implement a NewsAPI tool with `@tool` wrapper and register it in `src/agent/research.py`. Needs `NEWSAPI_KEY` env var gating.

**Effort:** Small. Straightforward API integration + tool registration.

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

## Phase 1.5: Verification Methodology

Improvements to the core recursive decomposition and synthesis pipeline. These refine how claims are broken down, researched, and reassembled into verdicts.

### 1.5.1 — Confidence-Weighted Synthesis

**Why:** Right now synthesis sees child verdicts and their confidence scores, but treats them somewhat equally. The LLM prompt mentions importance weighting, but the model has to intuit how much a child's low confidence should affect the parent verdict. A sub-claim with 0.6 confidence should count less than one with 0.95.

**What:**
- Pass explicit confidence scores and their interpretation to the synthesis prompt
- Add structured guidance: "A child verdict with confidence < 0.7 is uncertain — its contribution to the overall assessment should be discounted"
- Consider a pre-synthesis step that computes a weighted score as a hint (LLM still makes the final call)
- Track confidence propagation through the tree — does the final confidence reflect the weakest link?

**Effort:** Small. Mostly prompt engineering. The data is already available in `child_results`.

### 1.5.2 — Adaptive Research Depth

**Why:** Every leaf sub-claim currently gets the same research treatment — one ReAct agent, same timeout (240s agent, 360s total), same tool set. A sub-claim like "The US has a president" doesn't need 4 minutes of web search. Meanwhile, a nuanced economic claim might benefit from deeper research.

**What:**
- Have the decompose step tag each sub-claim with a complexity/controversiality estimate (e.g., `"complexity": "low"` / `"medium"` / `"high"`)
- Low complexity: short agent timeout, fewer search tools, quick verification
- High complexity: full agent timeout, all tools, potentially multiple research rounds
- This reduces total pipeline time for claims with simple sub-parts while allocating more effort where needed

**Effort:** Medium. Requires decompose prompt update, activity parameter changes, and timeout logic in the workflow.

### ~~1.5.3 — Sibling Contradiction Detection~~ (Superseded)

~~Original design referenced recursive tree siblings. With the flat pipeline, all sub-claims are siblings by default.~~ The thesis-aware synthesis now handles this — the synthesizer evaluates whether conflicting sub-verdicts undermine the speaker's thesis, rather than just counting.

### 1.5.4 — Sub-Claim Deduplication & Caching

**Why:** If two different compound nodes decompose into overlapping sub-claims (e.g., "US military spending is increasing" appears as a sub-part of multiple parent claims), the system currently researches them independently. Within a single verification run, this wastes time. Across runs over time, it's even more wasteful.

**What:**
- **Within-run dedup**: Before researching a leaf, check if an identical (or semantically similar) leaf has already been processed in this workflow run. If so, reuse its result.
- **Cross-run caching**: Cache research results by sub-claim text with a TTL (evidence goes stale). Use embeddings (joi:3103) to find semantically similar sub-claims that have already been verified.
- Start simple (exact text match within a run) and evolve to semantic similarity with TTL.

**Effort:** Medium-Large. Within-run dedup is straightforward (dict lookup in workflow). Cross-run caching needs a caching layer and embedding similarity search.

### ~~1.5.5 — Dynamic Depth Budget~~ (Obsolete)

~~The flat pipeline has no recursion or depth. Decompose produces all atomic facts in one pass. This item is no longer applicable.~~

### 1.5.6 — Evidence Quality Signals for Judges

**Why:** The judge currently receives raw evidence text and has to infer source reliability from URLs embedded in the content. Making source quality explicit would help the judge weigh evidence more accurately — especially when evidence from different sources conflicts.

**What:**
- Tag each evidence item with metadata before passing to the judge: `source_domain`, `credibility_tier` (from 1.3), `freshness` (how recent)
- Format evidence in the judge prompt with explicit quality signals: "[Tier 1 — Reuters, 2026-02-24] ..."
- The judge can then explicitly reason about source quality in its verdict
- Pairs well with 1.3 (Source Credibility Scoring) — the tiers become actionable in the judge prompt

**Effort:** Small (once 1.3 is done). Mostly prompt formatting changes.

### ~~1.5.7 — Decomposition Quality Feedback Loop~~ (Partially Addressed)

~~Thesis extraction now captures the speaker's intent at decompose time, which catches the main quality issue (naive fact counting in synthesis). Remaining improvements:~~
- Track when sub-claims are `unverifiable` due to vagueness (not lack of evidence)
- Use accumulated signals to improve the decompose prompt over time

**Effort:** Small. Mostly analytics.

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

### 4.2 — Parallel Batch Processing (DONE)

**Status:** ✅ Implemented

Sub-claims are processed in parallel batches using `asyncio.gather`:

- The workflow decomposes claims into atomic facts in one flat pass (with thesis extraction)
- Facts are processed in batches of MAX_CONCURRENT=2 (matched to GPU `--parallel 2`)
- Each fact goes through research → judge sequentially within its slot
- Two facts are researched/judged simultaneously per batch
- A single `synthesize_verdict` activity combines all judgments using the speaker's thesis as primary rubric
- Temporal handles per-activity retries and timeouts (research: 180s, judge: 300s, synthesis: 60s)

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
| **2** | Confidence-weighted synthesis (1.5.1) | Low effort, immediately improves verdict quality |
| **3** | Source credibility scoring (1.3) | Tiered weighting (basic filtering already done) |
| **4** | Adaptive research depth (1.5.2) | Cuts pipeline time in half for simple claims |
| **5** | Calibration test suite (3.1) | Can't improve without measuring |
| **6** | Evidence quality signals for judges (1.5.6) | Makes source tiers actionable in verdicts |
| **7** | RSS feed monitoring (2.1) | First step toward automated intake |
| **8** | Claim extraction from articles (2.2) | Enables fully automated pipeline |
| **9** | Sub-claim dedup & caching (1.5.4) | Reduces redundant work at scale |
| **10** | LangFuse observability (4.3) | Visibility into LLM performance |
| **11** | Speaker profiles (5.3) | Product differentiation |
| **12** | Article highlighting UI (5.1) | Core product experience |
| **13** | Human review loop (3.4) | Trust + quality assurance |
| **14** | Public API (5.2) | Distribution |
| **15** | Speech transcripts (2.3) | Expand beyond articles |

---

## Non-Goals (For Now)

Things that sound useful but aren't worth building yet:

- **Real-time verification** — verifying claims as they're spoken in a live broadcast. The pipeline takes ~2 minutes per claim. Real-time would require a fundamentally different architecture. Batch processing of transcripts after the fact is more practical.
- **Misinformation detection** — spin-cycle verifies specific claims, it doesn't detect misinformation patterns or narrative analysis. That's a different (harder) problem.
- **Opinion analysis** — "The economy is doing terribly" is an opinion, not a verifiable claim. The decomposer should filter these out (it mostly does). We verify facts, not feelings.
- **Image/video verification** — fake images and deepfakes are a real problem but require different tools (reverse image search, forensic analysis). Out of scope for now, though the Qwen3-VL model is multimodal and could eventually help here.
- **Multi-language** — all prompts are English. Supporting other languages means translating prompts, handling non-English search results, and dealing with non-English news sources. Important eventually, but not yet.
