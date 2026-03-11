# 60-Claim Batch Analysis — March 11, 2026

Model: Qwen3.5-122B-A10B (Q4_K_M) | Prompts: Rubric-based (Phase 1)

## Scorecard

| Metric | Count |
|--------|-------|
| Total claims | 60 |
| Correct verdicts | 50 (83%) |
| Wrong verdicts | 10 |
| Parse failures (unverifiable/0.0) | 3 |
| Flagged (pipeline crash) | 3 (Mar 10 batch, resubmitted) |
| Duplicate subclaims | 1+ |

### Verdict Distribution

| Verdict | Count |
|---------|-------|
| true | 21 |
| mostly_true | 4 |
| mixed | 4 |
| mostly_false | 10 |
| false | 17 |
| unverifiable | 4 |

### Compared to Mar 4 (pre-rubric, 40 overlapping claims)

| Change | Count |
|--------|-------|
| Improved | 5 |
| Regressed | 4 |
| Stable | 31 |

---

## Issue Categories

### A. Parse Failures (3 claims)

LLM output failed JSON extraction or Pydantic validation after 3 attempts.
All default to `unverifiable` with `confidence=0.0`.

These are the HIGHEST priority — the pipeline produced no usable output.

---

#### A1. "Since China stopped manipulating its currency in 2015, the trade deficit with the US has only gotten worse"

- **Verdict:** unverifiable (0.0)
- **Failure:** "Failed to parse LLM judgment after 3 attempts" on ALL 3 subclaims,
  then "Failed to synthesize verdict after 3 attempts"
- **Evidence collected:** 36, 42, 33 per subclaim — plenty of evidence
- **Subclaims:** (1) China ceased currency manipulation in 2015, (2) The cessation
  caused the increase, (3) The trade deficit increased after 2015

**Root cause:** The judge is producing output that fails JSON extraction or schema
validation 3 times in a row. With 30+ evidence items, the context may be large
enough to cause the model to produce malformed structured output.

**What it should be:** `mostly_false` — the presupposition that China "stopped
manipulating" is debatable, and the causal link is unsubstantiated. The trade
deficit data itself is verifiable.

---

#### A2. "Organic food has been proven to be more nutritious than conventionally grown food"

- **Verdict:** unverifiable (0.0)
- **Failure:** "Failed to parse LLM judgment after 3 attempts" at judge level
- **Evidence collected:** 33 items
- **Subclaims:** Single subclaim — "Organic food is more nutritious than
  conventionally grown food"

**Root cause:** Same parse failure pattern. Evidence is ample.

**What it should be:** `mostly_false` or `false` — major meta-analyses (Stanford
2012, Baranski 2014) reach different conclusions; no clear proof of "more
nutritious" as blanket claim. "Proven" is the key word — nothing is "proven."

---

#### A3. "Amazon pays its warehouse workers a minimum of $15/hr, but the company's warehouse injury rate is three times the industry average and Jeff Bezos personally blocked unionization efforts at the Bessemer, Alabama facility"

- **Verdict:** unverifiable (0.0) (one run); mostly_false (0.85) (another run)
- **Failure:** "Failed to parse LLM judgment after 3 attempts" on ALL 3 subclaims
  in one run. Succeeded on resubmission.
- **Evidence collected:** 30, 30, 21 per subclaim

**Root cause:** Intermittent parse failure. This claim also flagged (crashed) on
the Mar 10 run, suggesting it's a consistent trouble spot — possibly due to long
multi-part evidence contexts.

**What it should be:** `mostly_true` or `mixed` — $15/hr is true, injury rate is
~1.8x not 3x (mostly_true), Bezos "personally blocked" is oversimplified but
Amazon corporate did fight unionization (mostly_true).

**Note on the successful run's verdict:** The `mostly_false` verdict on the
resubmission is also wrong — it overcorrects on the injury rate ("not 3x
therefore false") and underweights the two substantially true sub-claims.

---

### B. Negation / Polarity Bugs (2 claims)

The decompose step rewrites the claim in a way that loses or inverts the
original polarity, and synthesize fails to account for the flip.

---

#### B1. "Lightning never strikes the same place twice"

- **Verdict:** true (0.95)
- **Should be:** false
- **Subclaim:** "There are documented cases of lightning striking the same
  location more than once." → true (0.95)
- **Reasoning:** "The claim is supported by overwhelming evidence from both
  government agencies (NOAA) and independent news organizations. Specific,
  documented cases are cited, such as the Empire State Building being struck
  approximately 100 times annually..."

**Root cause — DECOMPOSE POLARITY INVERSION:** The decomposer rewrote a
NEGATIVE claim ("never strikes twice") as a POSITIVE assertion ("there are
documented cases of it striking twice"). The judge correctly evaluated the
positive assertion as TRUE. But the synthesizer then carried `true` forward
without realizing the positive subclaim CONTRADICTS the original negative claim.

The reasoning text even says "The evidence directly confirms that lightning
frequently strikes the same location repeatedly, making the claim factually
accurate" — the model doesn't see the contradiction because by the time it
reaches synthesis, it's lost the original polarity.

**Fix approach (rubric):**
- **Decompose prompt:** Add explicit rule: "NEVER invert the polarity of the
  original claim. If the claim says 'X never happens', the subclaim must be
  'X never happens', NOT 'X has been documented to happen.' The judge must
  evaluate the ORIGINAL assertion, not its logical inverse."
- **Synthesize prompt:** Add rubric step: "Before rendering verdict, check
  whether each subclaim's truth value SUPPORTS or CONTRADICTS the original
  claim's assertion. A subclaim that says 'X exists' being TRUE means an
  original claim of 'X never exists' is FALSE."

---

#### B2. "Sweden never implemented any lockdown measures during the COVID-19 pandemic"

- **Verdict:** true (0.95)
- **Should be:** false or mostly_false
- **Subclaim:** "Sweden did not implement nationwide lockdown measures during the
  COVID-19 pandemic." → true (0.95)
- **Reasoning:** "The distinction between 'restrictions' and 'lockdowns' is
  consistently maintained across sources, validating the claim's specific
  assertion about the absence of lockdowns."

**Root cause — DECOMPOSE QUALIFIER INJECTION:** The decomposer added
"nationwide" to the subclaim, which wasn't in the original. The original says
"never implemented ANY lockdown measures" — absolute language. By softening
to "nationwide lockdown measures", the decomposer allowed the judge to
distinguish between "restrictions" and "lockdowns" and find the weaker claim
true.

Sweden imposed: ban on gatherings >8 people, high school closures, mandatory
business restrictions. These ARE lockdown measures even if not a full
stay-at-home order. The claim's "never...any" language makes it falsifiable
by any single restriction.

**Fix approach (rubric):**
- **Decompose prompt:** Add explicit rule: "NEVER add qualifiers, hedges, or
  scope narrowers that aren't present in the original claim. If the original
  says 'never implemented ANY lockdown measures', the subclaim must preserve
  the absolute scope. Do not soften to 'nationwide lockdown measures' or
  'full lockdown' — evaluate the claim AS STATED."
- **Normalize prompt:** May also need attention — should flag absolute language
  ("never", "any", "all", "every") as precision-critical and instruct
  downstream stages to preserve it.

---

### C. Temporal / Contextual Misreads (2 claims)

The pipeline fails to recognize temporal context that changes the truth value.

---

#### C1. "India will surpass China as the world's most populous country by 2030"

- **Verdict:** mostly_false (0.95)
- **Should be:** true or mostly_true
- **Subclaim:** "India's population is projected to surpass China's population
  by the year 2030" → mostly_false (0.95)
- **Reasoning:** "The claim that India's population is projected to surpass
  China's 'by 2030' is contradicted by the most recent and authoritative
  evidence. Multiple independent sources confirm that India already surpassed
  China in April 2023."

**Root cause — JUDGE LOGIC ERROR:** The judge FOUND the evidence that India
already surpassed China in 2023. It then rated the claim `mostly_false`
because "the specific timeline is incorrect" — but a prediction of "by 2030"
that already came true in 2023 is MORE true than predicted, not false. The
judge treated "already happened" as "prediction was wrong" instead of
"prediction was fulfilled early."

**Fix approach (rubric):**
- **Judge prompt:** Add to the QUANTITATIVE / TEMPORAL section: "If a
  prediction has already been fulfilled before the stated deadline, the claim
  is TRUE (the predicted event occurred within the timeframe). A prediction of
  'X will happen by 2030' that already happened in 2023 is true — it happened
  before the deadline, not after."

---

#### C2. "Switzerland has been neutral in every military conflict since 1815"

- **Verdict:** false (0.95)
- **Should be:** mostly_true or true
- **Subclaim:** "Switzerland has not participated in any military conflict
  since 1815" → false (0.95)
- **Reasoning:** "Historical evidence confirms that in July 1815, the Swiss
  army invaded the French region of Franche-Comté and fought French soldiers
  as part of the coalition against Napoleon. This military campaign occurred
  in the same year the claim's timeline begins."

**Root cause — JUDGE BOUNDARY PEDANTRY:** The judge found a military action
in July 1815, which is technically within "since 1815". But the claim's
obvious intent is about the post-Congress of Vienna era. Switzerland has
maintained formal neutrality for 200+ years since late 1815. Rating this
`false` at 0.95 over a boundary-month technicality is overcorrection.

More importantly: the 2022 Russia sanctions (breaking with tradition) are a
stronger challenge to the claim, but the judge fixated on the 1815 boundary
instead.

**Fix approach (rubric):**
- **Judge prompt:** Add to the PRECISION section: "When evaluating temporal
  boundary claims ('since 1815', 'for the past decade'), consider the
  charitable interpretation of the boundary date unless the claim specifically
  hinges on an exact start date. A claim about Swiss neutrality 'since 1815'
  refers to the post-Congress of Vienna settlement, not events during the
  Napoleonic Wars that the settlement concluded."
- This is more of a calibration issue than a structural bug — the judge
  needs to weigh materiality of exceptions, not just find any counterexample.

---

### D. Judge Factual Errors (2 claims)

The judge reaches a wrong factual conclusion despite having evidence available.

---

#### D1. "Canada has more coastline than any other country and is the second largest country by land area"

- **Verdict:** mostly_false (0.75)
- **Should be:** true
- **Subclaims:** "Canada has the longest coastline" → true |
  "Canada is the second largest country by land area" → mostly_false
- **Reasoning:** "While Canada ranks second globally when measuring 'total
  area' (which includes inland water bodies), it does not hold the second
  spot when measuring strictly 'land area.' In terms of land mass alone,
  Canada ranks behind other nations such as the United States or China."

**Root cause — JUDGE FACTUAL ERROR:** This reasoning is WRONG. Canada's land
area is 9,093,507 km² — second only to Russia (16,377,742 km²). The US
(9,147,420 km²) is larger in TOTAL area but SMALLER in land area than Canada
when water is excluded — or the rankings are essentially tied depending on
the source. The judge hallucinated a false distinction.

The claim says "land area" and Canada IS the second largest by land area.
This is a basic geographic fact the model got wrong.

**Fix approach:** This is a factual reasoning error, not a prompt structure
issue. The research gathered evidence, but the judge misinterpreted or
hallucinated the land area rankings. Potential fixes:
- **Evidence quality:** Check if the evidence actually contained clear land
  area figures. If not, the research step may need better seed queries for
  simple geographic facts.
- **Judge prompt:** Could add a rule: "When evidence provides specific
  numbers, state the numbers explicitly in your reasoning before drawing
  conclusions. Do not rely on general knowledge to interpret rankings."

---

#### D2. "Exposed workers at a nuclear weapons facility had lower cancer rates than the general population, and this finding has been replicated across multiple studies in different countries"

- **Verdict:** mostly_false (0.75)
- **Should be:** mostly_true or true
- **Subclaims:** "Workers had lower cancer rates" → mostly_true (0.75) |
  "Finding replicated across multiple countries" → mostly_false (0.75)
- **Reasoning:** "The claim's assertion that this finding is replicated across
  multiple countries is contradicted by broader international evidence. Major
  multinational studies, such as the INWORKS cohort and research involving
  Russian workers, have found positive dose-response relationships."

**Root cause — JUDGE CONFLATION:** The healthy worker effect IS one of the
most well-documented biases in occupational epidemiology. The judge confused
two different things: (1) the finding that nuclear workers have lower OVERALL
cancer/mortality rates than the general population (the healthy worker effect
— extensively replicated), and (2) dose-response studies that show radiation
increases SPECIFIC cancer risks. Both are true simultaneously — the HWE is
about overall rates, the dose-response is about specific cancers.

**Fix approach (rubric):**
- **Judge prompt:** This is a nuance issue. The judge needs to evaluate
  whether the SPECIFIC claim being made is supported, not whether a related
  but different finding exists. "When evidence shows conflicting findings,
  determine whether they address the SAME specific question or different
  aspects of a broader topic. Conflicting findings about different questions
  do not automatically contradict each other."

---

### E. Overclaiming / Calibration Issues (4 claims)

The pipeline reaches a defensible-direction verdict but overshoots to
`true`/`false` when `mostly_true`/`mostly_false` would be more appropriate.
These are calibration issues, not structural bugs.

---

#### E1. "Every single Republican voted against capping insulin prices at $35 in the Inflation Reduction Act"

- **Verdict:** true (0.95)
- **More appropriate:** mostly_true
- **Mar 4 verdict:** mostly_false (0.92) — found the 7 Rs who voted for the
  amendment
- **Subclaims:** (DUPLICATE) Two near-identical subclaims with only a trailing
  period difference: one got mostly_true (0.75), the other got true (0.95)

**Issue:** The Mar 11 reasoning collapsed "voted against the bill" with "voted
against the insulin cap provision." In reality, 7 Republican senators voted FOR
the separate amendment to extend the $35 cap to private insurance (it failed
57-43). The Mar 4 reasoning correctly caught this nuance.

The subclaim duplication is a separate bug — the quality validator should catch
subclaims that differ only by punctuation.

**Fix approach:**
- **Decompose quality validator:** Normalize trailing punctuation before
  comparing subclaims for semantic duplicates. Currently uses LLM comparison
  which apparently doesn't catch period-only differences.
- **Verdict calibration:** The `true` verdict isn't indefensible (all Rs did
  vote against the final bill), but the reasoning should acknowledge the
  amendment vote nuance. This is a depth-of-analysis issue.

---

#### E2. "Israel's treatment of Palestinians in the West Bank meets the legal definition of apartheid under international law"

- **Verdict:** true (0.95)
- **More appropriate:** mostly_true (contested legal category)
- **Mar 4 verdict:** mostly_true (0.82) — correctly hedged

**Issue:** The claim involves a contested legal category. Multiple authoritative
bodies (Amnesty, HRW, UNHCHR) have applied the label, but no binding court
judgment exists. `mostly_true` with appropriate hedging is more accurate than
`true` at 0.95 for an actively debated legal classification.

**Fix approach (rubric):**
- **Judge prompt:** The CONTESTED CATEGORIES guidance may need strengthening:
  "For claims about contested legal/political classifications (apartheid,
  genocide, terrorism), where authoritative expert bodies disagree or where
  no binding judicial determination exists, cap confidence at 0.85 and prefer
  `mostly_true`/`mostly_false` over `true`/`false`."

---

#### E3. "The United States spends more on its military than the next ten countries combined"

- **Verdict:** true (0.95)
- **More appropriate:** mostly_true (0.85)
- **Mar 4 verdict:** mostly_true (0.82)

**Issue:** The exact count fluctuates by year. SIPRI data shows it's closer to
"next 7-9" depending on the year and whether you use purchasing power parity.
The Mar 4 run actually did the math ($997B vs ~$782B for next 10) and
appropriately hedged. The Mar 11 run cited sources that repeat the "next 10"
talking point without independent verification.

**Fix approach:** Minor calibration issue. The judge relied on secondary sources
repeating the claim rather than doing primary arithmetic.

---

#### E4. "Rupert Murdoch's media empire has systematically promoted climate skepticism across Fox News, Sky News, and the New York Post"

- **Verdict:** mostly_true (0.85)
- **More appropriate:** true or mostly_true — this is actually fine
- **Mar 4 verdict:** mostly_true (0.85)

**No action needed.** The verdict is reasonable. The hedging on Sky News evidence
is appropriate.

---

### F. Flagged / Pipeline Crashes (3 claims, Mar 10 batch)

These claims crashed the pipeline and had to be resubmitted. No verdict was
produced. Need to check Temporal failure history for root cause.

---

#### F1. "Canada has more coastline than any other country and is the second largest country by land area"

Flagged on Mar 10 resubmission. Succeeded on later retry (with wrong verdict —
see D1).

#### F2. "Rupert Murdoch's media empire has systematically promoted climate skepticism across Fox News, Sky News, and the New York Post"

Flagged on Mar 10 resubmission. Succeeded on later retry (see E4).

#### F3. "Amazon pays its warehouse workers a minimum of $15 per hour..."

Flagged on Mar 10 AND parse-failed on one Mar 11 run (see A3). This claim is
consistently problematic — likely due to its length and 3-part structure
generating large evidence contexts.

---

### G. Questionable but Defensible (3 claims)

Not wrong per se, but worth monitoring.

---

#### G1. "UC Berkeley's graduate admissions were biased against women, with men admitted at a significantly higher rate overall"

- **Verdict:** true (0.95)
- **Notes:** Classic Simpson's paradox. The AGGREGATE statistic (44% M vs 35% F)
  is true. The pipeline correctly evaluated the claim AS STATED — the overall
  rate was higher for men. The reasoning doesn't surface the Simpson's paradox
  context, which would be ideal for a comprehensive analysis, but the verdict
  is technically correct.

**No fix needed for verdict.** Synthesize reasoning could be richer.

---

#### G2. "The US national debt has increased under every president since Jimmy Carter"

- **Verdict:** mixed (0.65)
- **Notes:** True in nominal terms, but the judge found conflicting metrics
  (nominal vs GDP share for Carter). `mixed` is defensible given the metric
  ambiguity. Could also be `mostly_true` or `true` if you stick to nominal.

**No fix needed.** The low confidence (0.65) appropriately reflects uncertainty.

---

#### G3. "Coconut vs shark deaths"

- **Verdict:** unverifiable (0.65)
- **Notes:** Correctly identified the 150-coconut-deaths figure as an urban
  legend with no primary source. `unverifiable` is a good call here.

**No fix needed.** This is a success case for the pipeline.

---

## Fix Plan — Rubric Changes

### Priority 1: Parse Failures (3 claims broken)

**Problem:** Judge produces output that fails JSON extraction 3× in a row.
Affects claims with large evidence contexts (30-42 items).

**Investigation needed:**
- Pull the raw LLM output from logs for these failures
- Determine if it's JSON formatting, schema mismatch, or context overflow
- May need to increase evidence truncation or split large contexts

**Potential fixes:**
1. Log the raw LLM output on parse failure (if not already) for diagnosis
2. Add evidence count cap before judge (e.g., rank_and_select already caps,
   but verify the cap is working for these claims)
3. Consider increasing max_retries from 2 to 3 for judge specifically
4. Test whether the model's structured output degrades with context length

---

### Priority 2: Decompose Polarity & Qualifier Injection (2 claims broken)

**Problem:** Decompose either inverts claim polarity (B1: lightning) or injects
softening qualifiers (B2: Sweden "nationwide").

**Fix — Decompose prompt additions:**

Add to the SUBCLAIM EXTRACTION rules:

```
POLARITY PRESERVATION (CRITICAL):
- NEVER invert the polarity of the original claim when creating subclaims.
- If the claim asserts "X never happens", the subclaim MUST be "X never
  happens" — NOT "X has been documented to happen."
- If the claim asserts "No country does X", the subclaim MUST be "No country
  does X" — NOT "Countries have been found to do X."
- The judge must evaluate the ORIGINAL assertion. If you rephrase a negative
  claim as a positive one, the judge will evaluate the positive version and
  the final verdict will be inverted.

QUALIFIER PRESERVATION (CRITICAL):
- NEVER add qualifiers, hedges, or scope narrowers not present in the
  original claim.
- If the claim says "never implemented ANY lockdown measures", do NOT soften
  to "did not implement NATIONWIDE lockdown measures" or "did not implement
  FULL lockdown measures."
- Absolute language ("never", "any", "all", "every", "no") is
  precision-critical. Preserve it exactly. The judge needs to evaluate the
  claim's actual strength, not a weakened version.
```

---

### Priority 3: Temporal Prediction Logic (1 claim broken)

**Problem:** Judge treats an already-fulfilled prediction as "wrong timeline"
instead of "fulfilled early."

**Fix — Judge prompt addition:**

Add to the TEMPORAL / QUANTITATIVE section:

```
PREDICTIONS AND DEADLINES:
- If a claim predicts "X will happen by [date]" and X has ALREADY happened
  before that date, the claim is TRUE — the prediction was fulfilled ahead
  of schedule.
- Do NOT rate such claims as false or mostly_false because "the specific
  date was wrong." The claim set an upper bound, and reality beat it.
- Example: "India will surpass China in population by 2030" — if India
  surpassed China in 2023, the claim is TRUE (it happened before 2030).
```

---

### Priority 4: Judge Calibration — Overclaiming (3 claims affected)

**Problem:** Model swings to `true`/`false` at 0.95 when nuanced middle
verdicts are more appropriate.

**Fix — Judge prompt additions:**

Add to the VERDICT CALIBRATION section:

```
CONTESTED CATEGORIES:
- For claims involving contested legal, political, or academic
  classifications (e.g., apartheid, genocide, recession) where authoritative
  bodies disagree or no binding judicial/regulatory determination exists:
  prefer mostly_true/mostly_false over true/false, and cap confidence at 0.85.

BOUNDARY TECHNICALITIES:
- When a temporal claim ("since 1815", "for the past decade") is
  substantially true across the claimed period but violated by a minor
  boundary case, do NOT rate it false.
- Weigh the MATERIALITY of exceptions. A 200-year neutrality record broken
  by a military action in the boundary month is mostly_true, not false.
- Ask: "Would a reasonable, informed person consider this claim true?"

PRECISION OF COMPARATIVE CLAIMS:
- For claims like "more than the next 10 combined" where the exact number
  fluctuates: if the DIRECTION is clearly true but the exact figure is
  approximate, use mostly_true rather than true.
```

---

### Priority 5: Duplicate Subclaim Detection

**Problem:** Quality validator misses near-duplicate subclaims that differ only
by trailing punctuation (insulin claim: two subclaims, only difference is a
period, got different verdicts).

**Fix — Quality validator (`decompose.py`):**
- Normalize subclaim text before comparison: strip trailing punctuation,
  collapse whitespace
- Or: add string-similarity check (e.g., Levenshtein ratio > 0.95 = duplicate)
  alongside the existing LLM-based semantic comparison

---

### Priority 6: Judge Factual Reasoning (2 claims affected)

**Problem:** Judge misinterprets evidence (Canada land area) or conflates
related but distinct findings (HWE replication).

**Fix — Judge prompt additions:**

```
EXPLICIT NUMBERS RULE:
- When evidence provides specific figures (areas, populations, dollar amounts),
  you MUST state the actual numbers in your reasoning before drawing
  conclusions about rankings or comparisons.
- Do NOT rely on general knowledge to interpret evidence. If evidence says
  "Canada: 9,093,507 km²" and "Russia: 16,377,742 km²", state both numbers
  and compare them directly.

DISTINGUISHING RELATED FINDINGS:
- When evidence shows apparently conflicting findings, determine whether they
  address the SAME specific question or DIFFERENT aspects of a broader topic.
- "Nuclear workers have lower overall mortality" and "radiation increases
  specific cancer risk" are NOT contradictory — they address different
  statistical measures (all-cause vs. specific-cause).
```

---

## Implementation Order

| # | Fix | Claims Fixed | Effort | Risk |
|---|-----|-------------|--------|------|
| 1 | Decompose: polarity + qualifier preservation rules | B1, B2 | Low (prompt edit) | Low |
| 2 | Judge: temporal prediction logic | C1 | Low (prompt edit) | Low |
| 3 | Judge: calibration rules (contested, boundary, precision) | C2, E1, E2, E3 | Low (prompt edit) | Medium — may overcorrect |
| 4 | Judge: explicit numbers + distinguishing findings | D1, D2 | Low (prompt edit) | Low |
| 5 | Decompose: duplicate subclaim detection (punctuation) | E1 (dupes) | Low (code) | Low |
| 6 | Parse failures: investigate + fix | A1, A2, A3 | Medium (diagnosis needed) | Unknown |

Total prompt additions: ~500 words across decompose + judge prompts.
No synthesize changes needed (issues originate upstream).
