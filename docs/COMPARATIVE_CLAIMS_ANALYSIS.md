# Comparative Claims Problem: Analysis & Proposed Solutions

**Date**: 2026-03-01
**Status**: Resolved — Option 1 (research strategy) implemented, tested, working

---

## The Problem

After implementing normalization + decomposition improvements (rules 1-9, checklist updates, semantic validators), **comparative subclaims consistently return `unverifiable`**.

The AIPAC/FARA claim has been run **6 times** across multiple iterations. The pattern is identical every time:

| Subclaim | Verdict | Confidence | Consistent? |
|----------|---------|------------|-------------|
| "AIPAC is not registered under FARA" | `true` | 0.85-0.95 | Every run |
| "Organizations lobbying for foreign govts are subject to FARA" | `true` | 0.95 | Every run |
| Any comparative subclaim (see variants below) | `unverifiable` | 0.35 | Every run |

### Comparative subclaim variants across runs (all `unverifiable @ 0.35`)

1. "AIPAC is treated differently than organizations lobbying on behalf of foreign governments regarding FARA exemptions"
2. "AIPAC has received FARA exemptions or regulatory treatment that differs from comparable organizations"
3. "Similar organizations to AIPAC are registered under FARA"
4. "Comparable organizations to AIPAC have different FARA exemption statuses"
5. "The special exceptions provided to AIPAC are inconsistent with the treatment of similar organizations"
6. "The special exceptions provided to AIPAC result in different treatment compared to similar organizations"

**Every formulation fails.** The decomposer has been improved significantly (Rule 9 operationalizes comparisons, normalization neutralizes loaded language), but the underlying problem persists.

---

## Root Cause Analysis

**This is not a decomposition problem. It's a research problem.**

The subclaims are well-formed. The judge's reasoning is consistent and correct: it wants specific comparative data (e.g., "Saudi lobby IS registered, Turkish lobby IS registered, AIPAC is NOT") and the evidence doesn't contain it.

### What the research agent finds

For the comparative subclaim "AIPAC is treated differently than organizations lobbying on behalf of foreign governments regarding FARA exemptions", the agent gathered 16 pieces of evidence:

| Source | Relevance | Contains comparative data? |
|--------|-----------|---------------------------|
| trackaipac.com | Relevant (AIPAC/FARA) | No — advocacy piece |
| wvnexus.org | Relevant (opinion) | No — argues AIPAC should register |
| cov.com (Covington) | Relevant (FARA legal guide) | No — general FARA exemptions |
| congress.gov (CRS) | Relevant (FARA overview) | No — mentions 400+ registrants but doesn't name them |
| quincyinst.org | Relevant (foreign lobbying) | Partial — mentions US subsidiaries of foreign businesses |
| opensecrets.org/fara | Relevant (FARA data) | No — general FARA description |
| **drfaraplasticsurgery.com** | **GARBAGE** | N/A — plastic surgeon named "Dr. Fara" |
| legalclarity.org | Relevant (FARA explainer) | No — general legal description |
| hashem.faith | Somewhat relevant | No — legal analysis of AIPAC specifically |
| scholarship.law.upenn.edu | Relevant (law review) | No — general FARA scholarship |
| kslaw.com | Relevant (FARA enforcement) | No — DOJ enforcement rollback |
| **efile.fara.gov** (PDF) | **PRIMARY SOURCE** | Partial — it's a single registrant filing, not a comparison |
| scholarship.law.duke.edu | Relevant (law review) | No — general FARA scholarship |
| penncerl.org | Relevant (FARA paper) | No — general political activities definition |
| **farahandfarah.com** | **GARBAGE** | N/A — "Farah & Farah" law firm |
| congress.gov (failed) | N/A | HTTP 403 |

### The key insight

The evidence to verify this claim **exists in the real world**:
- **efile.fara.gov** is the official FARA registrant database. It lists every registered foreign agent.
- Searching it would show: Saudi Arabia's lobby (registered), Turkey's lobby (registered), UAE's lobby (registered), while AIPAC is absent.
- Law review articles (Robinson 2020, Atieh 2010) discuss this comparison but the fetched excerpts don't include the comparative tables.

**The research agent never searches for the comparison group independently.** It searches for the comparative claim as a whole ("AIPAC treated differently") instead of searching for each side ("who IS registered under FARA" vs "is AIPAC registered").

---

## Proposed Solutions

### Option 1: Better research strategy for comparative claims (RECOMMENDED)

**Cost**: 0 extra LLM calls, prompt change only
**File**: `src/prompts/verification.py` — `RESEARCH_SYSTEM`
**Scope**: General-purpose fix that helps ALL comparative claims, not just FARA

**What to add**: Guidance in the research prompt that teaches the agent to decompose comparative searches:

```
COMPARATIVE CLAIMS — SEARCH BOTH SIDES INDEPENDENTLY:
When researching a claim about differential treatment, comparison, or inconsistency:
1. Search for evidence about Side A (e.g., "AIPAC FARA registration status")
2. Search for evidence about Side B (e.g., "FARA registered foreign agents list" or
   "organizations registered under FARA")
3. Search for direct comparisons (e.g., "FARA registration comparison lobbying groups")

Searching only for the comparison as a whole ("AIPAC treated differently") produces
opinion pieces. Searching for each side independently produces the factual data needed
to MAKE the comparison.

For regulatory claims specifically: search for the official registry or database.
Most regulatory frameworks have public registrant lists.
```

**Why this works**:
- The research agent already has a "SEARCH BOTH SIDES" directive for supporting vs contradicting evidence
- This extends it to comparative claims: search for evidence about each entity in the comparison
- It would lead the agent to search "FARA registered agents list" → find efile.fara.gov → find actual registrants
- General-purpose: helps any "X is treated differently than Y" claim

**Risks**:
- Uses tool budget on more searches (but comparative claims need it)
- Agent may still not find structured comparative data if it doesn't exist online

---

### Option 2: Decompose comparisons into independently verifiable atomic facts

**Cost**: 0 extra LLM calls, prompt change only
**File**: `src/prompts/verification.py` — `DECOMPOSE_SYSTEM` (rules or examples)

**What to change**: Add guidance that comparative claims should be split into their component sides when possible:

```
COMPARATIVE CLAIMS — SPLIT INTO VERIFIABLE SIDES:
When a claim asserts differential treatment or inconsistency between entities:
- Extract a fact about how Entity A is treated
- Extract a fact about how the comparison group is treated
- The comparison itself becomes the thesis (verified by synthesizer from sub-verdicts)

Example:
"X is treated differently than similar entities regarding [regulation]"
→ Fact 1: "X is not subject to [regulation]"
→ Fact 2: "Entities that [shared trait] are subject to [regulation]"
→ Thesis: "X's regulatory treatment is inconsistent with similar entities"

The synthesizer can then determine if differential treatment exists
from the individual verdicts, WITHOUT needing a single article that
makes the comparison explicitly.
```

**Why this works**:
- We already verify "AIPAC not registered" (true) and "orgs lobbying for foreign govts are subject to FARA" (true)
- If both are true, differential treatment is a logical inference the synthesizer can make
- Doesn't require finding a single article that says "AIPAC is treated differently"

**Why this might NOT work**:
- The synthesizer currently says `mostly_false @ 0.65` even when 2/3 subclaims are true — it's conservative
- The logical gap between "A is not registered" + "similar entities are registered" → "A is treated differently" may be too much for the judge to infer without explicit evidence
- The decomposer can't name specific comparison entities without using LLM knowledge

**However**: Looking at the data, we're ALREADY decomposing this way in some runs. The last run produced exactly:
1. "AIPAC is not registered under FARA" → true @ 0.85
2. "Organizations lobbying on behalf of foreign governments are subject to FARA" → true @ 0.95
3. "AIPAC is treated differently than orgs lobbying on behalf of foreign govts regarding FARA exemptions" → unverifiable @ 0.35

The problem isn't the decomposition — it's that subclaim 3 still exists and still can't be independently verified. Even if we remove it, the synthesizer needs to bridge the gap from (1) + (2) → "differential treatment exists."

---

### Option 3: Improve the synthesizer's inferential reasoning

**Cost**: 0 extra LLM calls, prompt change only
**File**: `src/prompts/verification.py` — `SYNTHESIZE_SYSTEM`

**What to change**: Add guidance that logical implications from verified sub-facts count as evidence:

```
LOGICAL IMPLICATIONS:
When individual sub-claims, taken together, logically imply a conclusion
stated in the thesis, that implication is valid evidence for the thesis —
even if no single source explicitly states the conclusion.

Example:
- Sub-claim 1: "Entity A is not registered under [regulation]" → TRUE
- Sub-claim 2: "Entities that [share A's trait] are subject to [regulation]" → TRUE
- Thesis: "Entity A is treated inconsistently with similar entities"
→ The thesis follows logically from the verified sub-claims.
  Verdict should reflect that the factual basis is established,
  even if no source explicitly draws the comparison.
```

**Why this might help**:
- The synthesizer already weighs by importance, not by count
- Adding explicit guidance about logical inference could push the final verdict from `mostly_false` to `mostly_true`

**Risk**:
- This is a slippery slope — we don't want the synthesizer making inferential leaps that aren't warranted
- The difference between "logical implication" and "hallucinated inference" is subjective

---

### Option 4: Add FARA registrant database as a programmatic source (like LegiScan)

**Cost**: New code, new API integration
**File**: New `src/tools/fara.py` + wire into research enrichment

The FARA database at `efile.fara.gov` is public and structured. We could:
1. Scrape/query the registrant list programmatically
2. Pass it as context to the judge (like LegiScan data)
3. The judge would see: "Registered: Saudi Arabia → Hogan Lovells, Turkey → Mercury Public Affairs, UAE → Akin Gump... AIPAC: NOT FOUND"

**Why this works**:
- Definitively answers the comparative question with primary-source data
- Same pattern as LegiScan (programmatic enrichment after research)

**Why NOT to do this now**:
- Very specific to FARA claims — doesn't generalize
- Adds maintenance burden for a niche data source
- The general-purpose fix (Option 1 or 2) should be tried first

---

## Recommendation

**Do Option 1 (research strategy) + Option 2 (decompose into sides) together.**

They're complementary and both zero-cost:
- Option 2 ensures we don't produce an un-researchable comparative subclaim
- Option 1 ensures the research agent searches for evidence about each side independently
- Together, they should produce: "AIPAC not registered" (true) + "Saudi/Turkish/UAE lobbies are registered" (true, found via efile.fara.gov) → synthesizer can infer differential treatment

Option 3 (synthesizer inference) is worth considering as a follow-up if the synthesizer still produces conservative verdicts after Options 1+2 improve the evidence quality.

Option 4 (FARA database) is overkill for now but could be revisited if FARA-related claims are common.

---

## Resolution (2026-03-02)

**Option 1 (comparative research strategy) was implemented and tested.**

Added to `RESEARCH_SYSTEM` in `src/prompts/verification.py`:
- "COMPARATIVE CLAIMS — SEARCH EACH SIDE INDEPENDENTLY" guidance
- Teaches the agent to search for Side A and Side B separately instead of searching for the comparison as a whole

Option 2 (decompose into sides) was already partially addressed by extraction Rule 9 (operationalize comparisons), which defines comparison groups by shared traits rather than vague similarity.

### Test Results

| Metric | Before (6 runs) | After |
|--------|-----------------|-------|
| Comparative subclaim verdict | `unverifiable` | **`mostly_true`** |
| Comparative subclaim confidence | 0.35 | **0.75** |
| Final verdict | `mostly_false` | **`mostly_true`** |
| Final confidence | 0.65 | **0.85** |
| Comparative evidence items | 16 | **31** |

Option 3 (synthesizer inference) was not needed — the improved research quality was sufficient.
Option 4 (FARA database) was not needed — the agent found the relevant data on its own.

---

## Test Plan

After implementing:
1. Restart worker container (`docker restart spin-cycle-dev-worker`)
2. Submit fresh AIPAC claim: "AIPAC should have to register under FARA, and the special exceptions which have been provided to it are inconsistent with other similar organizations."
3. Check:
   - Normalization still works (loaded language neutralized)
   - Subclaims don't include a monolithic comparative claim
   - Research agent searches for "FARA registered agents" / "who is registered under FARA" independently
   - Evidence includes actual registrant data (efile.fara.gov or similar)
   - No garbage results (drfaraplasticsurgery.com, farahandfarah.com)
   - Final verdict reflects the logical conclusion from verified sub-facts

---

## Context: What's Already Been Done

All of the following were implemented in this session and are live in the codebase:

1. **NormalizeOutput schema** — `src/schemas/llm_outputs.py`
2. **NORMALIZE_SYSTEM/USER prompts** — `src/prompts/verification.py` (6 transformations)
3. **Extraction rules 6-9** — decontextualize, extract underlying question, entity disambiguation, operationalize comparisons
4. **Checklist updates** — 4 items updated with action directives + 1 new decontextualization item
5. **Semantic validators** — validate_normalize, enhanced validate_decompose (dedup with length ratio, min fact length)
6. **Normalization wired into decompose_claim** — graceful fallback on failure
7. **Timeout bumped** — 60s → 90s for decompose activity
8. **All examples genericized** — no political references in prompts
9. **Docs updated** — ARCHITECTURE.md, README.md, ROADMAP.md

The normalization improvements are working correctly. The decomposition is producing better subclaims. The remaining problem is specifically about comparative claims where the research agent doesn't find the evidence needed to verify differential treatment.
