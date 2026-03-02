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

- **`ChatOpenAI`** — the LLM client that talks to the LLM server's OpenAI-compatible API. Every LLM call in the project goes through this class. It handles message formatting, streaming, structured output, and tool calling.
- **LangChain tools** — standardised interfaces for external services. DuckDuckGo search, Wikipedia, SearXNG, Serper, Brave, and page fetching are all wrapped as LangChain tools with a common `.invoke()` / `.ainvoke()` API.
- **Message types** — `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`. These are the primitives that make up an LLM conversation.

LangChain does NOT handle orchestration, retries, scheduling, or state persistence. It's the building blocks.

**Where it's used:**
- `src/llm.py` — shared `ChatOpenAI` client configuration
- `src/tools/web_search.py` — `DuckDuckGoSearchResults` tool
- `src/tools/wikipedia.py` — custom `@tool`-decorated async function
- `src/activities/verify_activities.py` — all LLM calls via `invoke_llm()` with structured output schemas

### LangGraph (agent framework)

LangGraph is the **agent engine**. It builds on LangChain to create state machines with:

- **Cycles**: a node can loop back to a previous node (research → evaluate → need more → research again)
- **Tool calling**: the LLM decides which tools to call, the graph executes them, feeds results back
- **State persistence**: every step reads from and writes to a typed state object

The critical pattern in Spin Cycle is the **ReAct (Reason + Act) agent** with progress awareness:

```
    ┌────────────┐     ┌──────────┐     ┌───────┐
    │ pre_model  │────▶│  agent   │────▶│ tools │
    │ (progress) │     │  (LLM)   │◀────│       │
    └────────────┘     └────┬─────┘     └───────┘
                            │ (no more tool calls)
                            ▼
                           END
```

1. **Pre-model hook** analyzes the conversation so far — counting tool calls, unique URLs, search queries, engines tried — and injects a progress summary into the LLM's system message (ephemeral, doesn't modify state)
2. LLM receives the conversation + tool definitions + progress note
3. LLM decides to call a tool → returns an `AIMessage` with `tool_calls`
4. Graph executes the tool → appends `ToolMessage` with results
5. Loop back to pre_model → agent. The progress note updates each iteration, giving the agent real-time awareness of what it has
6. LLM decides it has enough → returns a text response → graph ends

This is what makes the research step **agentic** — the LLM autonomously decides what to search, reads results, knows what it's already tried (via progress), and adapts its strategy.

**Where it's used:**
- `src/agent/research.py` — `create_react_agent()` builds the ReAct agent with `pre_model_hook=_research_pre_model_hook`

### Temporal (durable workflow orchestration)

Temporal is the **scheduler and reliability layer**. It handles:

- **Durable execution**: if a container crashes mid-workflow, Temporal replays from the last completed activity
- **Retries**: each activity has a `RetryPolicy` (max 3 attempts). If the LLM times out, Temporal retries just that activity
- **Timeouts**: activities have `start_to_close_timeout` (30-360s). Agent loops that run forever get killed
- **Scheduling**: extraction workflows will run on a Temporal cron schedule (every 15 min)
- **Visibility**: Temporal UI shows every workflow, its state, its history. Debug anything

The key insight: **LangGraph runs inside Temporal activities, not instead of them.**

```
Temporal Workflow
└── Activity: research_subclaim (retryable, timeout: 180s)
    ├── LangGraph ReAct Agent (cycles between LLM and tools)
    │   ├── pre_model_hook (injects progress: queries used, URLs found, engines tried)
    │   ├── LLM call (via LangChain ChatOpenAI + progress awareness)
    │   ├── SearXNG search (via LangChain tool, filtered by source_filter)
    │   ├── DuckDuckGo search (via LangChain tool, filtered by source_filter)
    │   ├── Serper / Brave search (via LangChain tool, filtered by source_filter)
    │   ├── Wikipedia search (via LangChain tool)
    │   ├── Page fetcher (URL → text extraction, blocked URLs rejected)
    │   ├── LLM call (sees results + progress, decides next action)
    │   └── ... (until LLM decides it has enough)
    └── Programmatic enrichment
        └── LegiScan (US legislation: bill details, roll call votes, bill text)
```

Temporal handles the **macro orchestration** (decompose → research → judge → synthesize → store). LangGraph handles the **micro orchestration** (search → read → decide → search more).

**Where it's used:**
- `src/workflows/verify.py` — `VerifyClaimWorkflow` definition
- `src/activities/verify_activities.py` — all 6 Temporal activities
- `src/worker.py` — worker entrypoint that registers workflows + activities
- `docker-compose.dev.yml` — Temporal server + Temporal UI containers

---

## The Verification Pipeline (what's working now)

A claim enters the system (via API or extraction) and is processed as a **flat pipeline of atomic facts**, orchestrated by Temporal with 6 activities. Inspired by Google DeepMind's SAFE (NeurIPS 2024) and FActScore — factual claims are flat structures, not hierarchical trees.

### Model Assignment

All steps use the same **Qwen3.5-35B-A3B** instance on the LLM server, running on **ROCm** for improved AMD GPU performance (~38 tok/s sustained throughput).

Thinking mode is toggled per-request via `chat_template_kwargs`, but currently **disabled for all steps**:

| Step | Mode | Why |
|------|------|-----|
| decompose_claim | `enable_thinking=False` | Structured JSON output, no reasoning needed |
| research_subclaim | `enable_thinking=False` | ReAct tool-routing — picking search queries. Thinking wastes ~25-45s/iteration |
| judge_subclaim | `enable_thinking=False` | Structured prompt guides reasoning. Thinking mode generates 5000-9500 tokens (3-4 min) without improving verdict quality — llama.cpp has no way to limit thinking tokens |
| synthesize_verdict | `enable_thinking=False` | Structured aggregation of sub-verdicts |

**Why thinking is disabled everywhere:**
llama.cpp's `--reasoning-budget` flag only supports `-1` (unlimited) or `0` (disabled) — no intermediate values for token limits. The model generates excessive internal monologue before responding. Until llama.cpp adds proper `max_thinking_tokens` support (or we migrate to vLLM which supports it), thinking mode is impractical.

### Step 1: decompose_claim (normalize → flat facts + linguistic patterns + thesis)

**File:** `src/activities/verify_activities.py`
**Prompts:** `src/prompts/verification.py` → `NORMALIZE_SYSTEM` / `NORMALIZE_USER` + `DECOMPOSE_SYSTEM` / `DECOMPOSE_USER`
**Patterns:** `src/prompts/linguistic_patterns.py` → 15 canonical linguistic categories + decomposition checklist

The decompose activity now runs **two LLM calls** internally:

1. **Normalize** — rewrites the claim in neutral, researchable language (1 LLM call, max_retries=1). Performs 7 transformations grounded in the academic literature:
   - **Bias neutralization** (Pryzant et al. AAAI 2020) — loaded language → neutral equivalents
   - **Operationalization** — vague abstractions → measurable indicators
   - **Normative/factual separation** (VeriScore, GCC taxonomy) — opinions stripped, facts kept
   - **Coreference resolution** — pronouns → explicit referents
   - **Reference grounding** (SAFE decontextualization) — acronyms expanded, dates grounded
   - **Speculative language handling** (AmbiFC ambiguity taxonomy) — predictions flagged
   - **Rhetorical/sarcastic framing** — conditional: only when claim clearly uses irony, rhetorical questions, or sarcasm; converts to literal assertion

   If normalization fails, the raw claim is used as fallback (graceful degradation).

2. **Decompose** — extracts flat atomic facts + thesis from the normalized claim (existing logic).

The normalized claim and list of changes are stored in `thesis_info` for auditability.

The LLM extracts a **flat list of atomic facts** plus **thesis information** that captures the speaker's intent. This approach matches Google SAFE and FActScore — simple, direct fact extraction without template expansion.

```
Input:  "Country A spends over $800B on military, more than Country B at $200B.
         Both countries are increasing military spending while cutting foreign aid."

Output: {
  "thesis": "Country A and Country B both prioritize military over aid, with A spending more",
  "key_test": "A ~$800B, B ~$200B, A > B, AND both must be increasing
               military AND both must be cutting foreign aid",
  "structure": "parallel_comparison",
  "facts": [
    "Country A spends over $800 billion on its military",
    "Country B spends about $200 billion on its military",
    "Country A's military spending is greater than Country B's",
    "Country A is increasing its military spending",
    "Country B is increasing its military spending",
    "Country A is cutting its foreign aid budget",
    "Country B is cutting its foreign aid budget"
  ],
  "interested_parties": {
    "direct": ["Country A Government", "Country B Government"],
    "institutional": ["Country A Ministry of Defense", "Country B Ministry of Defense"],
    "affiliated_media": [],
    "reasoning": "Both governments are subjects of the claim"
  }
}
```

**Why flat facts instead of structured templates?**

The previous approach used `entities + predicates + applies_to` with `{entity}` placeholder templates that code would expand. This was over-engineered:
- Added complexity (template parsing, expansion logic)
- LLM often used wrong placeholder names
- The standard approach (Google SAFE, FActScore) just outputs a flat list

The current approach:
1. LLM outputs facts directly as strings — no templates, no expansion
2. Linguistic patterns module guides decomposition (presuppositions, quantifiers, causation, etc.)
3. Thesis extraction captures speaker intent for synthesis

**Linguistic patterns for decomposition:**

The decompose prompt is augmented with a comprehensive **linguistic pattern taxonomy** (`src/prompts/linguistic_patterns.py`) covering 15 canonical categories from formal semantics:

| Category | What it catches |
|----------|----------------|
| **Presupposition Triggers** | "stopped", "started", "again", "still" — hidden assumptions |
| **Quantifier Scope** | "all" vs "most" vs "some" — different truth conditions |
| **Modality** | "may", "must", "should" — different claim strengths |
| **Evidentiality Markers** | "reportedly", "sources say" — hedging and attribution |
| **Temporal/Aspectual** | "since", "before", "after" — time boundaries |
| **Causation Types** | "caused", "contributed to" — causal vs correlational |
| **Comparison/Degree** | "first", "only", "largest" — superlatives need exhaustive verification |
| **Negation Scope** | "never", "nobody" — proving absence |
| **Speech Acts** | Assertions vs predictions vs opinions |
| **Vagueness/Hedging** | "significant", "many", "experts" — undefined terms |
| **Attribution** | "X said" — verify both attribution AND substance |
| **Conditionals** | "if X then Y" — may be unverifiable |
| **Definition/Category** | "X is a Y" — contested definitions |
| **Generics** | "Politicians lie" — generalizations |
| **Implicature** | Hidden meaning beyond literal text |

These patterns are appended to `DECOMPOSE_SYSTEM` at runtime.

**Extraction rules 6-9** (added alongside normalization) address missing capabilities from the literature:

| Rule | What it does | Source |
|------|-------------|--------|
| **6. Decontextualize** | Each fact must stand alone — no dangling pronouns or implicit references | Google SAFE, Molecular Facts (Gunjal et al. 2024) |
| **7. Extract underlying question** | Loaded phrasing → factual question being asked | ClaimDecomp (Chen et al. EMNLP 2022) |
| **8. Entity disambiguation** | Add minimum context for unique identification | Molecular Facts (Gunjal et al. 2024) |
| **9. Operationalize comparisons** | Define comparison groups by shared trait, not vague similarity | — |

The **decomposition checklist** now includes action directives (not just detection prompts) for vagueness operationalization, implicature extraction, speech act separation, causation preservation, and a new decontextualization quality check.

**The thesis extraction** captures the speaker's rhetorical intent:
- `thesis` — the argument the speaker is making
- `structure` — `simple`, `parallel_comparison`, `causal`, or `ranking`
- `key_test` — what must ALL be true for the thesis to hold

This is passed to the synthesizer so it evaluates whether the speaker's **argument** survives the evidence. Without this, a claim comparing two countries could be rated `mostly_true` if 5 of 6 facts check out — even if the one false fact (e.g., Country B NOT cutting aid) completely invalidates the speaker's parallel comparison.

**Interested parties extraction and expansion:**

The decompose step identifies parties with potential conflicts of interest through two layers:

1. **LLM extraction** — identifies direct parties, institutional connections, and reasoning
2. **SpaCy NER augmentation** — `en_core_web_sm` runs on the claim text to catch PERSON/ORG entities the LLM missed (deterministic, CPU-only, milliseconds)
3. **Wikidata expansion** — each party is programmatically expanded via SPARQL to discover:
   - Corporate ownership chains (subsidiaries, parent companies)
   - Media holdings (critical for source independence)
   - Political affiliations
   - Family relationships (2-hop: e.g., Person A → Spouse → Father-in-law)
   - Family members' corporate roles (founder, CEO, chairperson)

The expanded parties object includes:
- `direct`: Entities directly mentioned in the claim
- `institutional`: Parent organizations, governing bodies
- `affiliated_media`: Media outlets owned by or connected to interested parties
- `all_parties`: Full deduplicated list (used by judge for conflict detection)
- `wikidata_context`: Formatted text injected into judge and research prompts

**File:** `src/agent/decompose.py` → `expand_interested_parties()`
**File:** `src/tools/wikidata.py` → `get_ownership_chain()`, `collect_all_connected_parties()`
**File:** `src/utils/ner.py` → `extract_entities()` (SpaCy NER)

**LanguageTool grammar correction** runs on all LLM text outputs (facts, thesis, reasoning) to catch grammar oddities from quantized model outputs. Quantized LLMs sometimes produce valid-word substitutions that spell checkers miss (e.g., "priming" instead of "primary") — LanguageTool catches these because they create grammatically odd phrases even though each word is valid. The Java server lazy-loads on first use and runs locally inside the worker container.

**File:** `src/utils/text_cleanup.py` → `cleanup_text()` (LanguageTool)
**Applied in:** `decompose_claim` (facts, thesis), `extract_evidence` (content, title), `judge_subclaim` (reasoning), `synthesize_verdict` (reasoning)

All prompts include `Today's date: {current_date}` (formatted at call time) so the LLM knows the current date when evaluating temporal claims.

### Step 2: research_subclaim (the agentic part)

**File:** `src/agent/research.py` → `research_claim()`
**Called from:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `RESEARCH_SYSTEM` / `RESEARCH_USER`

This is where the LangGraph ReAct agent runs. For each **atomic fact**:

1. **Pre-model hook** injects a progress note into the system message — tool call count, unique URLs found, search queries used, engines tried, strategic suggestions (e.g., "try Brave for source diversity", "fetch full articles from your best URLs")
2. Agent receives: "Find evidence about: {sub-claim}" + progress awareness
3. LLM decides what to search → calls any of the available tools
4. Tool executes the search → returns results as text
5. Loop back to pre_model → agent. Progress note updates each iteration
6. LLM reads results + progress → decides if it needs more → calls another tool or stops
7. Typically: 8-12 tool calls per sub-claim, ~38 max agent steps
8. Agent timeout: 120s (soft), 180s (Temporal hard limit)
9. Max steps: 28 (allows ~14 tool calls before hard stop)

**Streaming evidence collection:** The agent uses `astream()` with `stream_mode="updates"` instead of `ainvoke()`. Messages are collected incrementally as the agent works. If the agent hits its step limit (`GraphRecursionError`) or times out, we keep ALL evidence gathered up to that point instead of losing everything. This replaced a direct `ainvoke()` call that would return nothing on interruption.

The research agent uses **thinking=off**. The ReAct loop is pure tool-routing — picking search queries and deciding when to stop. Thinking mode wastes ~25-45s per iteration generating `<think>` blocks that nobody reads, just to produce an 8-token tool call. With thinking off, the same search queries are produced in ~3s per iteration.

**Tools available to the agent (dynamically loaded based on API keys):**
- `searxng_search` — SearXNG meta-search (news, general web). Self-hosted, no API key.
- `web_search` — DuckDuckGo search (fallback/supplementary). No API key needed.
- `serper_search` — Google search via Serper API. Requires `SERPER_API_KEY`.
- `brave_search` — Brave Search API. Requires `BRAVE_API_KEY`.
- `wikipedia_search` — Wikipedia API search (established facts, background).
- `page_fetcher` — Fetches and extracts text from URLs found in search results.

**Programmatic enrichment (NOT agent tools — runs after the agent finishes):**
- **LegiScan** — US legislation search. If the subclaim matches any legislation, appends bill details (sponsors, status, history), roll call votes (individual member positions), and bill text (the actual legislative language). The bill text enables the judge to detect "poison pills" — provisions slipped into otherwise popular bills that explain otherwise puzzling voting patterns. Requires `LEGISCAN_API_KEY`.

All search tools pass results through `source_filter.py` before returning — low-quality sources (Reddit, Quora, social media, content farms, etc.) are silently dropped. See **Source Quality Filtering** below.

**MBFC cache population** runs as fire-and-forget background tasks during research. When search results come back, `populate_mbfc_cache()` launches async MBFC scrapes for uncached domains without blocking the agent. A `_mbfc_pending` dedup set prevents the same domain from being scraped multiple times across concurrent searches. By the time the judge runs, the cache is warm.

**Page fetcher entity extraction:** When the agent fetches a full article, SpaCy NER extracts PERSON/ORG entities from the content and includes them in the tool output (e.g., "Entities mentioned: Person A, Person B, Organization X"). This gives the agent visibility into who is quoted/mentioned without an additional LLM call.

The RESEARCH_SYSTEM prompt explicitly instructs the agent to prefer authoritative sources: government databases, wire services, established news outlets, academic institutions, official statistics agencies. It also includes three strategic search directives:
- **Search both sides** — after finding evidence leaning one direction, search for the opposite perspective
- **Comparative claims** — search for each side of a comparison independently instead of searching for the comparison as a whole (which produces opinion pieces instead of factual data)
- **Resolve position titles** — when a claim references a title ("head of Agency X"), first search to resolve who currently holds that position, then use the name in subsequent searches

After the agent finishes, we extract evidence from the conversation:
- Each `ToolMessage` becomes an evidence record (source_type: web/wikipedia)
- The agent's final `AIMessage` is NOT included — it's the agent's own interpretation, not primary evidence
- LegiScan enrichment appends legislative evidence items (no URL dedup against agent evidence — LegiScan returns structured data fundamentally different from web search)

**Fallback:** If the ReAct agent fails (tool calling issues, network errors) with no evidence gathered, we fall back to direct tool calls — no LLM reasoning, just search the claim text directly. Less targeted but still produces evidence. If the agent fails WITH partial evidence (common with step limit or timeout), we use that partial evidence instead of falling back.

### Step 3: judge_subclaim

**File:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `JUDGE_SYSTEM` / `JUDGE_USER`

The LLM evaluates evidence for a single sub-claim. This is NOT agentic — it's a single LLM call with structured output. Uses **thinking=off** — the structured prompt guides reasoning explicitly, and thinking mode generates 5000-9500 tokens (3-4 min) without improving verdict quality.

The critical constraint: **"Do NOT use your own knowledge."** The LLM must reason only from the evidence provided. This is what makes verdicts trustworthy — they're grounded in real, citable sources.

**Evidence quality ranking** (`src/utils/evidence_ranker.py`) runs when the research agent returns more than 20 evidence items. Instead of naively taking the first 20 (discovery order), evidence is scored 0-100 on quality signals and sorted before capping:

| Component | Range | Signals |
|-----------|-------|---------|
| Source type | 0-30 | Wikipedia=30, LegiScan=28, web=10 |
| MBFC factual | 0-30 | very-high=30, high=24, mostly-factual=16, unrated=12 |
| Gov/institutional TLD | 0-15 | .gov/.mil=15, .edu=10 |
| Content richness | 0-15 | >2000 chars=15, >800=10, >200=5 |
| MBFC credibility | 0-10 | high=10, medium=5, unrated=4 |

A domain diversity cap (max 3 items per domain) ensures at least 7 unique source domains in the final 20. Political bias is deliberately NOT a scoring signal — factual quality matters, political lean doesn't. Unrated sources get generous defaults (12/30 factual) because they include .gov data portals, academic papers, and international sources outside MBFC coverage. Scoring uses `get_source_rating_sync()` — cache-only, zero network calls.

**Pre-judge enrichment** runs before the LLM sees any evidence, in two passes:

1. **Entity enrichment (SpaCy NER → Wikidata):** All evidence content is concatenated, SpaCy extracts PERSON/ORG entities, new entities not already in `all_parties` are Wikidata-expanded (capped at 8). If a newly discovered entity connects to an existing interested party, it's added to `all_parties` and its media holdings are added to `affiliated_media`.

2. **Publisher enrichment (domain → Wikidata):** Unique source domains are extracted from evidence URLs and Wikidata-expanded to discover ownership chains. This catches cases where a news outlet is owned by an interested party.

**4 conflict-of-interest checks** run per evidence item during formatting:

| Check | What it detects | Example |
|-------|----------------|---------|
| **Affiliated media** | Source URL matches media owned by interested party | Outlet X when its owner is in `all_parties` |
| **Quoted interested party** | Evidence content quotes statements from claim subjects | "FBI stated that..." when claim is about FBI conduct |
| **Publisher ownership** | Source publisher owned by interested party (via MBFC ownership field) | Outlet X when its owner is in `all_parties` |
| **Sub-source MBFC** | Evidence references another publication with poor factual rating or extreme bias | "according to [outlet]" → outlet has Mixed factual rating |

Each check adds a `⚠️` warning to the evidence header that the LLM sees. The judge prompt has extensive instructions on how to handle self-serving statements, circular evidence, and interested party quotes — including specific patterns to reject and when to verdict "unverifiable."

**Source rating tags** from MBFC (Media Bias/Fact Check) are added to each evidence item:
- `[Center | Very High factual]` — bias and factual reporting rating
- `[Unrated source]` — domain not in MBFC database
- Bias distribution tracking warns if evidence skews heavily left or right

```
Input:  sub_claim = "Bitcoin was created by Satoshi Nakamoto in 2009"
        evidence = [Wikipedia excerpt, DuckDuckGo results]
Output: {"verdict": "true", "confidence": 0.95,
         "reasoning": "Multiple sources confirm..."}
```

Sub-claim verdicts: `true` | `mostly_true` | `mixed` | `mostly_false` | `false` | `unverifiable`

The judge uses a **6-level verdict scale** with spirit-vs-substance guidance:
- `true` — core assertion and key details are correct
- `mostly_true` — spirit is right, minor details off (e.g., "$50B" when the real figure is $48B)
- `mixed` — substantial parts both confirmed and contradicted
- `mostly_false` — core thrust is wrong, but contains some accurate elements
- `false` — directly contradicted by evidence
- `unverifiable` — insufficient evidence to determine

If there's no evidence, we short-circuit to "unverifiable" without calling the LLM.

### Step 4: synthesize_verdict (thesis-aware synthesis)

**File:** `src/activities/verify_activities.py`
**Prompt:** `src/prompts/verification.py` → `SYNTHESIZE_SYSTEM` / `SYNTHESIZE_USER`

A single synthesis activity that combines sub-claim verdicts into an overall verdict. When thesis info is available (from the decompose step), the synthesizer evaluates whether the **speaker's argument** survives the sub-verdicts — not just whether a majority of facts are true.

```
Input:  claim_text = "Country A and Country B are both increasing military..."
        child_results = [
            {"sub_claim": "Country A increasing military spending", "verdict": "mostly_true", ...},
            {"sub_claim": "Country B increasing military spending", "verdict": "true", ...},
            {"sub_claim": "Country A cutting foreign aid", "verdict": "mostly_true", ...},
            {"sub_claim": "Country B cutting foreign aid", "verdict": "false", ...}
        ]
        thesis_info = {
            "thesis": "Country A and Country B are prioritizing military spending over foreign aid.",
            "structure": "parallel_comparison",
            "key_test": "Both countries must show increased military spending AND decreased foreign aid."
        }
        is_final = True
Output: {"verdict": "mostly_false", "confidence": 0.85,
         "reasoning": "The thesis requires both countries to be cutting foreign aid. Country B is actually increasing it, which undermines the core argument. Country A's part holds — increased military spending and reduced foreign aid, but the parallel comparison fails because Country B contradicts the thesis."}
```

The thesis is injected as a `SPEAKER'S THESIS` block in the synthesis prompt. The synthesizer is instructed to use the thesis as its **primary rubric** — evaluating whether THAT ARGUMENT survives the sub-verdicts, not whether a numerical majority of facts are true.

Why use LLM instead of averaging? Because "X happened in 2019 and cost $50M" where the event DID happen but in 2020 and cost $48M is "mostly true" — the core claim is right, details are slightly off. An LLM makes this nuance call better than math.

### Step 5: store_result

**File:** `src/activities/verify_activities.py`

Takes the result dict and writes it to Postgres:
- One `SubClaim` row per atomic fact (all `is_leaf=True` in the flat pipeline)
- One `Evidence` row per evidence item (source_type, content, URL) — linked to leaf sub-claims
- One `Verdict` row (overall verdict, confidence, reasoning)
- Updates `Claim.status` to "verified"

Evidence records with `source_type` not in the DB enum (`web`, `wikipedia`, `news_api`) are filtered out — the agent_summary doesn't get stored.

### Workflow Orchestration (flat pipeline)

The workflow processes claims in a flat pipeline — one decompose call (with thesis extraction), then research+judge in parallel batches, then thesis-aware synthesis.

```
VerifyClaimWorkflow
├── create_claim (if needed)
├── decompose_claim (90s timeout) → {facts: [...], thesis_info: {thesis, normalized_claim, normalization_changes, structure, key_test}}
│   ├── normalize (internal LLM call, max_retries=1, graceful fallback)
│
├── For each batch of MAX_CONCURRENT=2 facts:
│   └── asyncio.gather(
│       ├── research_subclaim (180s timeout)
│       │   ├── ReAct agent (streaming evidence collection)
│       │   └── LegiScan enrichment (programmatic, after agent)
│       │   → judge_subclaim (300s timeout)
│       └── research_subclaim (180s timeout)
│           ├── ReAct agent (streaming evidence collection)
│           └── LegiScan enrichment (programmatic, after agent)
│           → judge_subclaim (300s timeout)
│       )
│
├── IF 1 fact: skip synthesis, use judgment directly
├── IF 2+ facts: synthesize_verdict (60s timeout, is_final=True, thesis_info passed)
│
└── store_result (30s timeout)
```

Key properties:
- **Flat, not recursive** — no tree, no recursion, no MAX_DEPTH. One decompose call produces flat facts + thesis.
- **Direct fact extraction** — LLM outputs facts directly as strings, guided by linguistic patterns taxonomy. No template expansion.
- **Thesis-aware** — the decompose step extracts the speaker's intent (thesis, structure, key_test) and passes it to synthesis. The synthesizer evaluates whether the argument survives the evidence, not just whether a majority of facts are true.
- **No hard fact limit** — fact count is driven by claim complexity, not an arbitrary cap. Complex claims get full coverage.
- **MAX_CONCURRENT = 2** — limits parallel research+judge pipelines to match GPU bandwidth. Two research agents run simultaneously.
- **Streaming evidence** — agent uses `astream()` to collect evidence incrementally. Timeout or step limit preserves all evidence gathered so far.
- **Programmatic enrichment** — after the agent finishes, LegiScan searches for matching legislation and appends structured evidence (bill details, roll call votes, bill text).
- **Single synthesis** — `synthesize_verdict` combines all fact-level judgments into one final verdict. Single-fact claims skip synthesis entirely.
- **Temporal retries per activity** — if one research call fails, only that activity retries (max 3 attempts).
- **Date-aware** — all prompts include `Today's date: {current_date}` so the LLM references current data, not training cutoff data.

### GPU Compute Constraints

The LLM runs via llama.cpp with **ROCm backend** (AMD GPU optimization). `--parallel N` slots multiplex concurrent requests onto a single GPU — it does NOT parallelize them. N concurrent requests = each takes ~Nx longer, total throughput is constant (~38 tok/s sustained).

| Service | Port | `--parallel` | Backend | Notes |
|---------|------|-------------|---------|-------|
| Qwen3.5 | `:3101` | 4 | ROCm | Unified model, thinking toggled per-request |
| Embedding | `:3103` | 4 | Vulkan | Fast, low-latency |

`MAX_CONCURRENT=2` limits parallel research+judge pipelines. Higher concurrency doesn't improve wall-clock time — it just increases per-request latency.

---

## Source Quality Filtering

**File:** `src/tools/source_filter.py`

All search results pass through a domain blocklist before reaching the research agent. This is a hard filter — blocked domains are silently dropped.

### Why?

Search engines return Reddit comments, Quora answers, Medium blogs, and other user-generated content that isn't citable for fact-checking. The LLM prompt also instructs the agent to prefer authoritative sources, but the code-level filter catches what the LLM might miss.

### Blocked Categories (~70 domains)

| Category | Examples | Reason |
|----------|----------|--------|
| Social media / forums | reddit.com, quora.com, twitter.com, facebook.com | User-generated, unvetted |
| Content farms | ehow.com, answers.com, reference.com | SEO-driven, not authoritative |
| Video platforms | youtube.com, vimeo.com, tiktok.com | Not citable text sources |
| Fact-check sites | snopes.com, politifact.com, factcheck.org | We do our own verification |
| Blog platforms | medium.com, substack.com | Mostly unvetted |
| AI aggregators | perplexity.ai, you.com | Not primary sources |
| Tabloids | dailymail.co.uk, thesun.co.uk, nypost.com, tmz.com | Sensationalist, unreliable |
| Partisan outlets | breitbart.com, infowars.com, dailywire.com, occupydemocrats.com | Ideological bias |
| State propaganda | rt.com, sputniknews.com | State-controlled media |

### How It's Wired

- `filter_results(results)` — called in all 4 search tools (SearXNG, Serper, Brave, DuckDuckGo) on the result list before returning
- `is_blocked(url)` — called in `page_fetcher.py` to reject blocked URLs before fetching
- Handles subdomains: `old.reddit.com` matches the `reddit.com` block
- Search tools request extra results (e.g., 15 instead of 10) to compensate for filtering losses

### Prompt-Level Reinforcement

The `RESEARCH_SYSTEM` prompt explicitly lists acceptable and forbidden source types, ranked in a **3-tier hierarchy**:

| Tier | Sources | Weight |
|------|---------|--------|
| **Tier 1 — Primary documents** | Treaties, charters, legislation, court filings, UN resolutions, official data (World Bank, SIPRI, BLS), academic papers | Strongest |
| **Tier 2 — Independent reporting** | Wire services (Reuters, AP), major outlets (BBC, NYT, Guardian), Wikipedia, think tanks | Strong |
| **Tier 3 — Interested-party statements** | Government websites (whitehouse.gov, state.gov, kremlin.ru), press releases, politician statements | Weakest — treated as claims, not facts |

The judge prompt mirrors this hierarchy: primary documents outweigh reporting, and both outweigh political statements. Government websites are explicitly flagged as communications arms of political actors, not neutral sources.

- **NEVER USE:** Reddit, Quora, social media, personal blogs, forums, YouTube comments, AI-generated summaries, fact-check sites (Snopes, PolitiFact)

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
- **Atomic claims only** — "Country A spent $50B on Project X" not "Country A spent $50B on Project X and cancelled the second phase"
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
│ text (text)     │   │   │ claim_id (uuid)FK│   │   │ sub_claim_id (FK) │
│ source_url      │   └──▶│ parent_id (FK)   │   └──▶│ source_type (enum)│
│ source_name     │       │ is_leaf (bool)   │       │ source_url        │
│ status (enum)   │       │ text (text)      │       │ content (text)    │
│ created_at      │       │ verdict (enum)   │       │ supports_claim    │
│ updated_at      │       │ confidence (float)│       │ retrieved_at      │
└────────┬────────┘       │ reasoning (text) │       └────────────────────┘
         │                   └──────────────────┘
         │       ┌──────────────────┐      parent_id is self-referential:
         │       │    verdicts      │      compound nodes link to their parent,
         │       │──────────────────│      leaves link to their parent node,
         └──────▶│ id (uuid) PK     │      top-level nodes have parent_id = NULL
                 │ claim_id (FK) UQ  │
                 │ verdict (enum)    │
                 │ confidence (float)│
                 │ reasoning (text)  │
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

Atomic sub-claims and compound nodes decomposed from the parent claim by the LLM. Forms a tree structure via self-referential `parent_id`.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, default uuid4 | Unique identifier |
| `claim_id` | `UUID` | FK → claims.id, NOT NULL | Parent claim |
| `parent_id` | `UUID` | FK → sub_claims.id, nullable | Parent compound node (NULL for top-level nodes) |
| `is_leaf` | `BOOLEAN` | NOT NULL, default true | Leaf (researched+judged) vs compound (synthesized from children) |
| `text` | `TEXT` | NOT NULL | Leaf: verifiable assertion. Compound: decomposed text |
| `verdict` | `ENUM(...)` | nullable | LLM's verdict on this sub-claim (7-level scale) |
| `confidence` | `FLOAT` | nullable | 0.0 to 1.0 confidence score |
| `reasoning` | `TEXT` | nullable | LLM's explanation of the verdict |

**Relationships:**
- Belongs to one `claim`
- Has many `evidence` (cascade delete)
- Self-referential: has optional `parent` (compound node) and many `children`

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
| `reasoning` | `TEXT` | nullable | Top-level synthesis reasoning explaining the verdict |
| `reasoning_chain` | `JSONB` | nullable | Array of reasoning strings from sub-claim judgments |
| `created_at` | `TIMESTAMPTZ` | default now() | When the verdict was produced |

**Relationships:**
- Belongs to one `claim` (one-to-one via unique constraint)

### Enums

| Enum Name | Values | Used By |
|-----------|--------|---------|
| `claim_status` | pending, processing, verified, flagged | claims.status |
| `sub_claim_verdict` | true, false, partially_true, unverifiable, mostly_true, mixed, mostly_false | sub_claims.verdict |
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

One unified model running via llama.cpp, with thinking toggled per-request:

| Port | Model | Mode | Used By |
|------|-------|------|--------|
| `:3101` | Qwen3.5-35B-A3B | `enable_thinking=False` | decompose, research, synthesize |
| `:3101` | Qwen3.5-35B-A3B | `enable_thinking=True` | judge |
| `:3103` | (embeddings — not yet used) | — | planned: evidence caching |

30B params, 3B active MoE. Thinking mode is toggled via `chat_template_kwargs` in the request body. When thinking is enabled, the model produces `<think>...</think>` blocks that are stripped before parsing.

### Connection Path

```
Container → Docker DNS (127.0.0.11) → Tailscale FQDN → LLM server
```

The `LLAMA_URL` env var points to the LLM server's Tailscale FQDN (e.g. `http://host.tailf424db.ts.net:3101`).

### Configuration

All LLM calls go through `src/llm.py`:

```python
from langchain_openai import ChatOpenAI

def get_llm(temperature=0.1):           # thinking=off — used for ALL pipeline steps
    return ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",     # :3101
        model="Qwen3.5-35B-A3B",
        temperature=temperature,
        max_tokens=2048,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )

# get_reasoning_llm() also exists (thinking=on) but is UNUSED — kept for experiments.
# llama.cpp lacks max_thinking_tokens support, so thinking mode is impractical.
```

`max_tokens` is set explicitly to prevent llama.cpp's default `n_predict` from cutting off LLM output mid-JSON.

### Prompt Design

All prompts live in `src/prompts/verification.py` with extensive inline documentation explaining:
- What each prompt does and why it's designed that way
- Why thinking mode is disabled for all steps (llama.cpp limitation)
- Example inputs and outputs
- Design constraints (e.g., "Do NOT use your own knowledge")

Four prompt pairs (system + user):
1. `DECOMPOSE_SYSTEM` / `DECOMPOSE_USER` — flat fact extraction with linguistic patterns taxonomy
2. `RESEARCH_SYSTEM` / `RESEARCH_USER` — guide the research agent (includes source quality rules)
3. `JUDGE_SYSTEM` / `JUDGE_USER` — evaluate evidence for a sub-claim
4. `SYNTHESIZE_SYSTEM` / `SYNTHESIZE_USER` — combine child verdicts (importance-weighted, adapts via `is_final` parameter)

Plus the linguistic patterns module (`src/prompts/linguistic_patterns.py`) which is appended to `DECOMPOSE_SYSTEM` at runtime.

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
| `SubClaimResponse` | Sub-claim in verdict | `text`, `verdict`, `confidence`, `reasoning`, `evidence_count`, `children[]` (recursive) |
| `VerdictResponse` | Full claim detail | All claim fields + `verdict`, `confidence`, `reasoning`, `sub_claims[]` (tree) |
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

- `LLAMA_URL` — LLM API (llama.cpp Qwen3.5-35B-A3B, unified thinking/non-thinking, via Tailscale)
- `LLAMA_EMBED_URL` — LLM embeddings API (llama.cpp, via Tailscale)
- SearXNG — self-hosted meta-search (configured via `SEARXNG_URL`)
- DuckDuckGo — web search (no API key)
- Serper — Google search API (requires `SERPER_API_KEY`)
- Brave Search — web search API (requires `BRAVE_API_KEY`)
- Wikipedia API — factual lookups (no API key)

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
| PostgreSQL schema | **Done** | 4 tables: claims, sub_claims (with tree structure), evidence, verdicts |
| FastAPI API | **Done** | POST/GET claims, health check, lifespan management |
| Temporal workflow | **Done** | VerifyClaimWorkflow with 5 activities, flat pipeline, thesis-aware synthesis, retry policies |
| Temporal worker | **Done** | Registers workflow + 6 activities, structured logging configured |
| `decompose_claim` | **Done** | LLM decomposes text into flat facts (guided by linguistic patterns) + thesis (structure, key_test) in one pass |
| `research_subclaim` | **Done** | LangGraph ReAct agent with SearXNG + DuckDuckGo + Serper + Brave + Wikipedia + page_fetcher |
| `judge_subclaim` | **Done** | LLM evaluates evidence, returns structured verdict |
| `synthesize_verdict` | **Done** | Thesis-aware synthesis — evaluates whether speaker's argument survives sub-verdicts (importance-weighted, not count-based) |
| `store_result` | **Done** | Writes flat result to Postgres (all sub-claims as leaves) |
| Source quality filtering | **Done** | Domain blocklist (~40 domains) filters all search results + page fetches |
| Prompts | **Done** | 4 prompt pairs documented in `src/prompts/verification.py` |
| LLM connectivity | **Done** | Unified Qwen3.5 on :3101 — `enable_thinking` toggled per-request (off: decompose/research/synthesize, on: judge) |
| Logging | **Done** | Structured JSON logging via `src/utils/logging.py`, Promtail → Loki → Grafana |
| Tests | **Done** | Health endpoint, schema validation |

### What's Planned

See [ROADMAP.md](ROADMAP.md) for the full prioritised improvement plan. Key next steps:

| Component | Status | Details |
|-----------|--------|--------|
| Alembic migrations | **Next** | Unblocks all future schema changes. Dependency installed, no init |
| Source credibility hierarchy | **Done** | 3-tier system in prompts: primary docs > independent reporting > political statements |
| Calibration test suite | **Planned** | 100+ known claims, measure accuracy and confidence calibration |
| RSS feed monitoring | **Planned** | `source_feeds` + `articles` tables, Temporal cron workflow |
| Claim extraction pipeline | **Planned** | ExtractClaimsWorkflow, LLM reads articles → extracts claims |
| LangFuse integration | **Planned** | Self-hosted LLM observability |

---

## Testing & Debugging

### Submitting Claims via API

```bash
# Submit a claim
curl -s -X POST http://localhost:4500/claims \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin was created by Satoshi Nakamoto in 2009"}' | python3 -m json.tool

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
# I [CREATE    ] start: Creating claim record | claim=Bitcoin was created by Satoshi Nakamoto in ...
# I [DECOMPOSE ] start: Decomposing claim | claim=Bitcoin was created by Satoshi Nakamoto in ...
# I [DECOMPOSE ] done: Claim decomposed | sub_count=1
# I [WORKFLOW  ] leaf_start: Processing leaf sub-claim | sub_claim=Bitcoin was created by Satoshi Nakamoto in 2009 depth=1
# I [RESEARCH  ] start: Starting research agent | sub_claim=Bitcoin was created by Satoshi Nakamoto in 2009
# I [RESEARCH  ] done: Research complete | evidence_count=6
# I [JUDGE     ] done: Sub-claim judged | verdict=false confidence=0.95
# I [SYNTHESIZE] done: Verdict synthesized | verdict=false confidence=0.95
# I [STORE     ] done: Result stored in database | claim_id=abc123... verdict=false

# With LOG_FORMAT=json (default in prod), output is JSON for Loki:
# {"ts":"2025-01-15T12:00:00.123Z","level":"INFO","module":"decompose","action":"done","msg":"Claim decomposed","sub_count":4}
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
│   ├── llm.py                      # Shared LLM client (ChatOpenAI → LLM server)
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
│   │   └── research.py             # ReAct research agent (multi-source search)
│   │
│   ├── tools/                      # LangChain tools for the agent
│   │   ├── source_filter.py        # Domain blocklist — filters junk sources from all tools
│   │   ├── searxng.py              # SearXNG meta-search (self-hosted)
│   │   ├── serper.py               # Serper (Google Search API)
│   │   ├── brave.py                # Brave Search API
│   │   ├── web_search.py           # DuckDuckGo search wrapper
│   │   ├── wikipedia.py            # Wikipedia API with @tool decorator
│   │   └── page_fetcher.py         # URL → text extraction (respects blocklist)
│   │
│   ├── prompts/                    # All LLM prompts with documentation
│   │   ├── verification.py         # Decompose, Research, Judge, Synthesize
│   │   └── linguistic_patterns.py  # 15-category linguistic pattern taxonomy
│   │
│   ├── workflows/                  # Temporal workflow definitions
│   │   └── verify.py               # VerifyClaimWorkflow
│   │
│   ├── activities/                 # Temporal activity implementations
│   │   └── verify_activities.py    # All 6 verification activities
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
    ├── test_health.py              # Health endpoint tests
    └── test_schemas.py             # Schema validation tests
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >=0.2.0 | Agent state machine framework (ReAct agent) |
| `langchain` | >=0.3.0 | Foundation: message types, tool interfaces |
| `langchain-openai` | >=0.2.0 | `ChatOpenAI` client for LLM server API |
| `langchain-community` | >=0.3.0 | `DuckDuckGoSearchResults` tool |
| `temporalio` | >=1.7.0 | Workflow orchestration, workers, activities |
| `fastapi` | >=0.115.0 | REST API framework |
| `uvicorn` | >=0.32.0 | ASGI server |
| `pydantic` | >=2.0 | Request/response validation |
| `sqlalchemy` | >=2.0 | Async ORM (PostgreSQL) |
| `asyncpg` | >=0.30.0 | Async PostgreSQL driver |
| `httpx` | >=0.28.0 | Async HTTP client (Wikipedia, Serper, Brave, SearXNG) |
