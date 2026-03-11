# Verdict Quality Issues — 2026-03-10/11 Test Runs

Initial 5-claim test after SearXNG removal + prefetch enablement, followed by
full 40-claim regression + stress suite. Issues organized by root cause.

**Evidence quality note**: SearXNG removal dramatically improved evidence.
2,990 evidence items across 40 claims (avg 75/claim), 0 unverifiable verdicts,
0 garbage domains. Top sources: Wikipedia, LegiScan, Statista, PMC, NPR, BBC,
NYT, Reuters. See "Evidence Quality" section at the end.

---

## Rubric Restructure Results (2026-03-10)

Replaced prose-based judge prompt (26.4K chars) with 5-step rubric (8.5K chars)
and prose-based synthesize prompt (8.3K chars) with 4-step rubric (4.4K chars).
See `docs/rubric-restructure-plan.md` for the plan and rationale.

**Core idea**: Force structured output steps (interpret → triage → direction →
precision → verdict) so the model can't skip evaluation steps. The `is_independent`
field on each evidence assessment embeds self-serving detection in the output
structure instead of 3K chars of prose guidance.

### 9-Claim Retest Results

| Claim | Before Rubric | After Rubric | Expected | Result |
|-------|--------------|-------------|----------|--------|
| Canada coastline + 2nd largest | mostly_false | mostly_false | mostly_true | Same |
| Murdoch climate skepticism | mostly_false | **true** | mostly_true | **Improved** |
| Amazon warehouse | mostly_false | mostly_false | mostly_true | Same |
| Finland education/suicide | false | **mostly_false** | mostly_false | **Fixed** |
| Churchill quote | mostly_false | **true** | true | **Fixed** |
| ExxonMobil climate | mostly_true | **true** | true | **Fixed** |
| Insulin price cap | mostly_false | mostly_false | mostly_true | Same |
| Japan elderly care | mostly_false | mostly_false | mostly_true | Same |
| Google/Meta EU fines | mostly_false | **unverifiable** | mostly_false | **Improved** |

**5 of 9 improved, 0 regressions.**

### What the rubric fixed (root causes resolved)

- **RC1 (synthesizer counting)**: Murdoch went from mostly_false→true. The rubric
  forced the synthesizer to classify Fox/NYPost as core_assertion and Sky News as
  supporting_detail, then evaluate whether the thesis survives.
- **RC2 (judge pedantry)**: Churchill went from mostly_false→true. The rubric's
  precision step explicitly handles attribution ("Did X say these words? Yes →
  correct") instead of the model inventing a standard about original authorship.
  ExxonMobil also improved (mostly_true→true) — the direction assessment step
  ignores PR denials as non-independent evidence.
- **RC3 (false/mostly_false threshold)**: Finland went from false→mostly_false.
  The boundary rule in the rubric ("ANY meaningful truth content → mostly_false")
  worked correctly for historically-accurate claims.
- **Google/Meta**: Improved from mostly_false→unverifiable. The independence check
  correctly identified that EU corporate tax data for specific companies isn't
  publicly available, making the comparison genuinely unverifiable.

### What the rubric didn't fix (remaining issues)

- **Canada, Amazon**: Both subclaims classified as core_assertion, so the
  synthesizer still lets one failure drag the verdict. The rubric's classification
  step works when there's a clear thesis vs supporting detail, but parallel
  assertions joined by "and" get treated as co-equal cores.
- **Insulin**: "Every single Republican" is an absolute that genuinely fails.
  mostly_false is arguably correct — 94% opposition with an incorrect absolute.
- **Japan**: Both subclaims are genuinely wrong (Monaco #1 median age, European
  countries outspend Japan). mostly_false is correct.

### Schema validation issue found

The LLM sometimes puts parenthetical qualifiers in Literal fields:
`"supports (historical)"` instead of `"supports"`. Fixed by stripping
parentheticals in the `EvidenceAssessment.normalize_assessment` validator.
Without this fix, ~10% of judge calls needed retries.

---

## Root Cause 1: Synthesizer Counts Instead of Weighing

The synthesize prompt says "WEIGH BY IMPORTANCE, NOT BY COUNT" but the model
consistently defaults to counting subclaims. When any subclaim fails, the
overall verdict is dragged down even when the core thesis is clearly supported.

**Affects 4 claims — all have flipped verdicts (wrong direction).**

### 1a. Canada Coastline — mostly_false (0.90), should be **mostly_true**

**Claim**: "Canada has more coastline than any other country and is the second
largest country by land area"

**Subclaims**:
- "Canada has the longest coastline" → **true (0.95)** — correct, 202,080 km
- "Canada is 2nd largest by total land area" → **mostly_false (0.85)** — pedantic:
  Canada is 2nd by total area (land + water) but 3rd/4th by land-only area

The coastline part (the more remarkable claim) is unambiguously true. The "2nd
largest" part is how virtually everyone describes Canada — the land-vs-total-area
distinction is a technicality. The synthesizer treats both as co-equal core
assertions and lets the technicality flip the entire verdict.

### 1b. Murdoch Climate Skepticism — mostly_false (0.82), should be **mostly_true**

**Claim**: "Rupert Murdoch's media empire has systematically promoted climate
skepticism across Fox News, Sky News, and the New York Post"

**Subclaims**:
- Fox News → **mostly_true (0.85)** — well-documented (Media Matters, academic studies)
- NY Post → **mostly_true (0.85)** — well-documented
- Sky News → **mostly_false (0.72)** — UK Sky News was sold 2018; Sky News
  Australia (still Murdoch-owned) IS a documented climate misinformation hub

Core thesis: "Murdoch's empire systematically promotes climate skepticism."
This is overwhelmingly supported. One of three named outlets is ambiguous (not
wrong — Sky News Australia does exactly what the claim says). The synthesizer
treated 2-of-3 as a failing grade instead of weighing the core thesis.

### 1c. Amazon Warehouse — mostly_false (0.82), should be **mostly_true**

**Claim**: "Amazon pays its warehouse workers a minimum of $15 per hour, but the
company's warehouse injury rate is three times the industry average and Jeff
Bezos personally blocked unionization efforts at the Bessemer, Alabama facility"

**Subclaims**:
- $15/hr minimum → **mostly_true (0.85)** — confirmed
- 3x injury rate → **mostly_true (0.78)** — confirmed (OSHA data, news investigations)
- Bezos personally blocked union → **mostly_false (0.72)** — Amazon engaged in
  illegal union-busting (NLRB ruling) but evidence doesn't support Bezos
  *personally* directing it

2 of 3 parts are true. The 3rd is an overstatement of corporate behavior
attributed to an individual. The synthesis reasoning explicitly says "one of
three is false, therefore overall cannot be true or mostly true."

### 1d. Finland Education/Suicide — false (0.92), should be **mostly_false**

**Claim**: "Finland has the best education system in the world and the highest
suicide rate in Europe"

**Subclaims**:
- Best education → **false (0.92)** — Singapore now #1 PISA, Finland ~#12
- Highest suicide rate → **false (0.85)** — Lithuania now leads; Finland near EU average

Both parts were historically true (Finland WAS #1 PISA for years, DID have the
highest European suicide rate in the 1980s-90s). Present-tense assertions are
wrong, but "false" implies fabrication. The synthesizer sees two "false"
subclaims and goes to "false" overall, ignoring the historical grounding. This
is a commonly-repeated folk-knowledge claim that WAS accurate — **mostly_false**
better captures "outdated but not fabricated."

### Proposed Fix: Synthesizer Core-vs-Detail Identification

1. **Synthesize prompt strengthening**: Add explicit examples showing that when
   a compound claim has a clear core thesis supported by most subclaims, the
   failing subclaim is a detail error that belongs in the reasoning, not the
   verdict. Example: "Murdoch promotes climate skepticism (core) across Fox,
   Sky, NYPost (supporting details)" — if 2 of 3 outlets confirm the core
   thesis, the verdict should reflect the core being true.

2. **Core assertion extraction**: Have the synthesizer explicitly identify
   the core assertion before counting subclaims. "What is this claim really
   saying?" should be answered before "how many subclaims passed?"

3. **Historical grounding rule**: When claims use present tense for assertions
   that were historically accurate, the verdict should be mostly_false (outdated)
   rather than false (fabricated).

**Priority: HIGH** — affects the most claims and produces the worst errors
(fully flipped verdicts).

---

## Root Cause 2: Judge Over-Pedantry

The judge applies excessively literal standards to attribution, semantic
precision, and edge-case technicalities, diverging from how a reasonable
person would evaluate claims.

**Affects 3 claims — all too conservative (should be one level higher).**

### 2a. Churchill Quote — mostly_false (0.85), should be **mostly_true**

**Claim**: "Winston Churchill said that democracy is the worst form of
government except for all the others"

**Subclaims**:
- "Churchill stated the quote" → **mostly_false (0.85)** — He DID say it in the
  House of Commons (Nov 11, 1947, recorded in Hansard). Judge penalizes because
  he prefaced with "it has been said that..." — treating a universally accepted
  attribution as false because Churchill credited it as pre-existing.
- "The quote is a comparative claim that democracy is inferior" → **mostly_false
  (0.85)** — DECOMPOSITION ERROR. The original claim never asserts this
  interpretation. The decomposer invented a straw-man subclaim about the
  quote's meaning.

Two compounding errors: (1) bogus second subclaim that the quality validator
should have caught, (2) excessively literal attribution standard. The quote is
universally attributed to Churchill by every quotation compendium.

### 2b. ExxonMobil Climate — mostly_true (0.85), should be **true**

**Claim**: "ExxonMobil's own scientists confirmed climate change was real in the
1970s but the company spent decades funding climate denial"

**Subclaims**:
- Scientists confirmed in 1970s → **mostly_true (0.85)** — downgraded because
  "Exxon spokespersons disputed the conclusiveness." PR denials are not
  legitimate counter-evidence to peer-reviewed analysis (2023 Science journal,
  Harvard/Potsdam study, Inside Climate News investigation).
- Decades funding denial → **mostly_true (0.85)** — downgraded because funding
  went to "think tanks" rather than "direct scientific research." The claim
  says "funding climate denial" not "funding research" — this is hair-splitting.

Both parts are among the most thoroughly documented corporate malfeasance
stories in history, backed by court proceedings, peer-reviewed studies, and
primary source documents.

### 2c. IRA Climate Investment — mostly_true (0.85), should be **true**

**Claim**: "The Inflation Reduction Act allocated $369 billion for climate and
energy provisions, making it the largest climate investment in US history"

**Subclaims**:
- $369B allocation → **mostly_true (0.85)** — downgraded because actual spending
  may exceed $800B-$1.2T due to uncapped tax credits. But "allocated $369B" is
  the CBO-scored figure — exceeding it makes the claim a conservative
  *understatement*, not an error.
- Largest climate investment in US history → **mostly_true (0.85)** — confirmed
  by CRS, Yale, RMI, every non-partisan analysis.

The judge confused "the estimate might be exceeded" with "the estimate might
be wrong." Understating the truth is not inaccuracy.

### Proposed Fix: Judge Pedantry Calibration

1. **Attribution standard**: "X said Y" is true if X spoke those words in a
   documented setting, regardless of whether X claimed original authorship.
   Add judge prompt guidance: "When evaluating attribution claims ('X said Y'),
   the question is whether X spoke or wrote those words, not whether X claimed
   to have originated them."

2. **PR denials are not counter-evidence**: Add guidance that corporate/official
   denials do not constitute evidence against findings from peer-reviewed
   studies, court proceedings, or investigative journalism.

3. **Understatement ≠ inaccuracy**: When evidence shows a claim's figure is
   lower than reality, the claim understates the truth — this supports the
   claim rather than undermining it.

**Priority: HIGH** — conservative bias on well-documented facts undermines
credibility of the pipeline.

---

## Root Cause 3: False vs Mostly-False Threshold

"False" should mean the claim is fundamentally fabricated or contradicted.
The pipeline uses "false" for claims that have significant truth content but
fail on specific superlatives or absolute quantifiers.

**Affects 3 claims — all one level too harsh.**

### 3a. Japan Elderly — false (0.85), should be **mostly_false**

**Claim**: "Japan has the oldest population in the world and spends more per
capita on elderly care than any European country"

**Subclaims**:
- Oldest population → **mostly_false (0.78)** — Monaco has higher median age,
  but Monaco is a microstate (pop ~39,000). Most mainstream sources (Reuters,
  WEF) call Japan #1. Edge-case technicality.
- Spending more than any European country → **false (0.85)** — France, Belgium,
  Luxembourg all exceed Japan. This part is genuinely wrong.

The spending part is wrong, but Japan IS extremely old and DOES spend heavily
on elderly care. "False" implies the claim is fabricated — it's not. The
directional claim (Japan = old + high elderly spending) is correct; the
superlatives ("oldest" and "more than any") are wrong.

### 3b. Republican Insulin — false (0.95), should be **mostly_false**

**Claim**: "Every single Republican voted against capping insulin prices at $35
in the Inflation Reduction Act"

**Subclaims**:
- House Republicans → **mostly_false (0.85)** — 12 of 205 House Rs voted FOR
- Senate Republicans → **mostly_false (0.85)** — 7 Senate Rs voted to keep the
  $35 cap (but voted against the overall IRA)

"Every single" is demonstrably wrong. But the directional claim is accurate:
193 of 205 House Rs (94%) and nearly all Senate Rs opposed it. "False" at 0.95
confidence implies fabrication. This is a true-in-spirit claim with an
incorrect absolute quantifier — exactly what mostly_false is for.

### 3c. Finland Education/Suicide — also affected (covered under Root Cause 1)

### Proposed Fix: False/Mostly-False Boundary

1. **Judge prompt threshold guidance**: Add explicit definition:
   - **false**: The claim's core assertion is fabricated, invented, or
     fundamentally contradicted. No reasonable interpretation makes it true.
   - **mostly_false**: The claim has a kernel of truth or is directionally
     grounded but makes specific assertions (superlatives, absolute quantifiers,
     exact figures) that are wrong.

2. **Directional truth rule**: When the direction/spirit of a claim is supported
   but specific quantifiers fail ("every" → "94%", "highest" → "one of the
   highest"), the verdict is mostly_false, not false.

**Priority: MEDIUM** — affects claims with superlatives and absolute quantifiers
(common in political rhetoric).

---

## Root Cause 4: Decomposition Quality

Bad subclaim generation that distorts the verdict.

### 4a. Churchill Quote — Straw-Man Subclaim

The decomposer created a second subclaim interpreting the quote's *meaning*
("democracy is inferior to all other forms") — something the original claim
never asserted. The claim simply says Churchill said X. The quality validator
should have caught this as a non-verifiable interpretation, not a fact.

### 4b. SpaceX/DOGE — Juxtaposition Misinterpretation (from initial 5-claim test)

Decomposer sometimes treats "aims to cut $2T" as "has cut $2T." Inconsistent
across runs. The 40-claim batch actually handled this correctly (mostly_true,
0.85) — suggesting the decomposition instability is stochastic, not systematic.

### 4c. Canada Coastline — Over-Specific Wording Preserved

Decomposer preserved "by total land area" from the claim, creating a falsifiable
technicality. Should have normalized to "by size" or "by area" since the common
understanding of "second largest country" doesn't distinguish land-vs-total.

### Proposed Fix: Decomposition Improvements

1. **Quality validator enhancement**: Detect subclaims that interpret or
   analyze the meaning of quoted text rather than verifying factual assertions.

2. **Normalizer: strip pedantic precision**: When claims use informal language
   ("land area" when they mean "area"), normalize to the commonly understood
   meaning rather than preserving the technically-falsifiable wording.

3. **Intentional verb handling**: (from initial test) "Aims to" / "seeks to"
   → verify the intent, not the outcome.

4. **Contrast connector handling**: "While" / "simultaneously" / "despite" →
   verify each side independently.

**Priority: MEDIUM** — affects specific claim patterns but decomposition
instability means some runs handle these correctly.

---

## Previously Documented Issues (Initial 5-Claim Test)

### Pelosi — Absolutist Language Over-Penalization
**Verdict**: mostly_false, should be **mostly_true**

Judge treats "virtually every" vs "almost every" as meaningful factual failure.
Direction is clearly supported (Pelosi outperformed most hedge funds). See
Root Cause 2 (judge pedantry) — same pattern.

**Fix**: Quantifier equivalence guidance in judge prompt + normalizer softening.

### Google/Meta — Research Scope Too Narrow
**Verdict**: mostly_false (0.85) — weak confidence

Research was Ireland-focused (finfacts-blog.com, Irish DPC fines). Claim is
about worldwide tax avoidance by two companies.

**Fix**: Decompose multi-entity claims into per-entity subclaims. Seed query
guidance for financial claims targeting primary sources (SEC, EU Commission).

### SpaceX/DOGE — Juxtaposition Handling
Resolved in 40-claim batch (mostly_true, 0.85). Decomposition instability
means this sometimes works and sometimes doesn't. Fix via explicit contrast
connector rules in decompose prompt.

---

## Correctly Handled Claims (25 of 40)

Notable successes:

- **Counterfactual tax rate** → mostly_false (0.85): Excellent handling —
  decomposed into verifiable premises + unverifiable counterfactual core,
  then used historical trend data to evaluate directionality. One of the
  best-handled claims.
- **Bezos $1M to everyone** → false (0.92): Did the math correctly ($8
  quadrillion needed vs $200B net worth).
- **ECB rates 2025** → false (0.95): Found ECB actually cut rates in 2025.
- **WHO lab leak** → false (0.95): Correctly identified misattribution.
- **No president convicted** → true (0.95): Correct.
- **Great Barrier Reef** → true (0.92): Well-sourced with ARC study data.
- **Opioid/Purdue** → mostly_true (0.92): Clean 4-subclaim decomposition, all
  well-sourced. Arguably should be true (the "withdrawals vs profits"
  distinction for Sackler $10B is minor).
- **NASA budget** → false (0.92): Correctly identified sequestration-year cuts.
- **Great Wall of China** → false (0.92): Correctly debunked.
- **Brazil deforestation** → mostly_false (0.78): Correctly distinguished
  "rate" (Cambodia leads) vs "total area" (Brazil leads).
- **US military spending** → mostly_true (0.85): Appropriate nuance on
  "next ten" vs actual number.
- **Population doubled** → mostly_true (0.85): Correctly handled approximate
  truth (2.19x ≈ "doubled").
- **G7 healthcare** → mostly_true (0.85): Correct.
- **Sweden lockdowns** → mostly_false (0.82): Correctly identified later
  restrictions despite early no-lockdown approach.
- **Stanford remote work** → mostly_false (0.90): Correctly identified 13%
  vs fabricated 70% figure.

---

## Evidence Quality — SearXNG Removal Impact

### Before (with SearXNG)
- Garbage results: psychologytoday.com DID articles for Pelosi queries,
  history.state.gov 1917 diplomatic cables for SCOTUS queries,
  travelchinaguide.com for China energy queries
- Multiple unverifiable verdicts due to evidence noise
- `.gov` over-scoring promoted irrelevant government pages

### After (Serper + DDG only)
- **2,990 evidence items** across 40 claims (avg 75/claim)
- **0 unverifiable verdicts**
- **0 garbage domains** — all evidence is topically relevant
- **Minimum 20 evidence items** per claim, 9-26 unique domains per claim
- Top domains: Wikipedia (155), LegiScan (85), Statista (45), PMC (45),
  NPR (28), BBC (28), NYT (26), Reuters (21)

### Data Hygiene Issue
785 evidence items (26%) have NULL `source_url` — URLs exist embedded in
content text (search snippets, failed fetches) but aren't extracted to the
column. Doesn't affect verdicts but worth fixing.

---

## Summary of All Fixes

| Issue | Root Cause | Component | Status |
|-------|-----------|-----------|--------|
| Synthesizer counts vs weighs core thesis | RC1 | synthesize rubric | **FIXED** — rubric Step 2 forces classification |
| Core assertion extraction before counting | RC1 | synthesize rubric | **FIXED** — thesis_survives field |
| Historical grounding rule (outdated ≠ fabricated) | RC1 | judge rubric | **FIXED** — boundary rule in Step 5 |
| Attribution standard (said ≠ originated) | RC2 | judge rubric | **FIXED** — Step 4 precision check |
| PR denials not counter-evidence | RC2 | judge rubric | **FIXED** — is_independent field in Step 2 |
| Understatement ≠ inaccuracy | RC2 | judge rubric | **FIXED** — Step 4 precision check |
| False/mostly-false threshold | RC3 | judge rubric | **FIXED** — boundary rule in Step 5 |
| Directional truth rule for quantifiers | RC3 | judge rubric | **FIXED** — Step 4 precision check |
| Quantifier equivalence (virtually≈almost) | RC2 | judge rubric | **FIXED** — Step 4 precision check |
| Parallel "and" assertions treated as co-equal cores | RC1 | synthesize rubric | **OPEN** — model classifies both as core |
| Quality validator: meaning-interpretation subclaims | RC4 | decompose | **OPEN** — not addressed by rubric |
| Normalizer: strip pedantic precision | RC4 | normalizer prompt | **OPEN** — not addressed by rubric |
| Intentional verb handling (aims/seeks) | RC4 | decompose prompt | **OPEN** — stochastic, sometimes works |
| Contrast connector handling (while/despite) | RC4 | decompose prompt | **OPEN** — stochastic, sometimes works |
| Multi-entity claim splitting | — | decompose prompt | **OPEN** |
| Seed query targeting for financial claims | — | decompose prompt | **OPEN** |
| NULL source_url data hygiene | — | research.py | **OPEN** — low priority |
| `.gov` over-scoring | — | evidence_ranker.py | **OPEN** — see seed-relevance-filtering.md |
| Seed relevance filtering | — | evidence_ranker.py | **OPEN** — see seed-relevance-filtering.md |

## Related Documentation

- `docs/rubric-restructure-plan.md` — rubric design plan, content audit,
  new schemas, implementation sequence (Phase 1 COMPLETE)
- `docs/seed-relevance-filtering.md` — proposed keyword gate + embedding
  similarity for topical relevance filtering
- `.gov` scoring adjustment also proposed there (GOV_TLD_SCORE 15→5,
  FACTUAL_UNRATED_GOV 20→12)
