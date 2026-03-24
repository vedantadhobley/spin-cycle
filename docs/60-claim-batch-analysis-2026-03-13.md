# 60-Claim Batch Analysis — March 13, 2026

Model: Qwen3.5-122B-A10B (Q4_K_M) | Prompts: Rubric-based (Phase 1) + evidence digest citations

Changes since March 11: structured evidence digest for synthesize, [N] citation
format in synthesis reasoning, reasoning depth guidance replacing vague
"contextual reasoning" prompt section.

## Scorecard

| Metric | Mar 13 | Mar 11 | Delta |
|--------|--------|--------|-------|
| Total claims | 60 | 60 | — |
| Correct verdicts | 38 (63%) | 50 (83%) | -12* |
| Debatable verdicts | 14 (23%) | — | — |
| Wrong verdicts | 8 (13%) | 10 (17%) | -2 |
| Parse failures | **0** | 3 | **-3** |
| Pipeline crashes | 0 | 3 | -3 |
| Duplicate subclaims | 0 | 1+ | fixed |

*Note: Mar 11 counted "debatable" as "correct." Using the same methodology
(correct + debatable), Mar 13 scores 52/60 (87%) vs Mar 11's 50/60 (83%).

### Verdict Distribution

| Verdict | Mar 13 | Mar 11 | Delta |
|---------|--------|--------|-------|
| true | 19 | 21 | -2 |
| mostly_true | 10 | 4 | +6 |
| mixed | 0 | 4 | -4 |
| mostly_false | 14 | 10 | +4 |
| false | 20 | 17 | +3 |
| unverifiable | 2 | 4 | -2 |

The shift from `mixed`/`unverifiable` toward `mostly_true`/`mostly_false` shows
better calibration — the pipeline is committing to directional verdicts instead
of punting to middle categories.

### Compared to Mar 11

| Change | Count |
|--------|-------|
| Fixed (broken → correct) | 7 |
| Improved (better calibration) | 2 |
| Stable | 47 |
| Regressed | 2 |
| New failure (persistent bug) | 2 |

---

## Changes from March 11

### Fixed (7 claims)

#### F1. China currency/trade deficit — parse failure → false (0.95)

Was a complete parse failure on Mar 11 (judge output failed JSON extraction 3×).
Now correctly identifies the false presupposition ("China stopped manipulating"
is demonstrably false — designated a currency manipulator in 2019), verifies the
trade deficit did increase, and rejects the causal link. Clean decomposition into
3 subclaims, all evaluated correctly.

#### F2. Organic food nutritious — parse failure → false (0.92)

Another Mar 11 parse failure. Now correctly evaluates: major meta-analyses
(Stanford 2012, Baranski 2014) reach conflicting conclusions, and the word
"proven" makes the claim definitively false. No blanket proof exists.

#### F3. Lightning never strikes twice (advanced suite) — true → false (0.95)

Mar 11 had a **polarity inversion bug**: decomposer rewrote "Lightning never
strikes the same place twice" as "There are documented cases of lightning
striking the same location more than once." Judge found that true, synthesize
carried it forward as `true` without realizing the inversion.

Mar 13: decomposer preserves the negative framing ("Lightning never strikes the
same location more than once"), judge evaluates it directly as false, citing
Empire State Building (~100 strikes/year), NOAA data, and park ranger Roy
Sullivan (struck 7 times).

#### F4. India surpasses China by 2030 — mostly_false → true (0.95)

Mar 11 had a **temporal logic error**: judge found India surpassed China in 2023
but rated the claim `mostly_false` because "the specific timeline is incorrect."
The prediction "by 2030" was fulfilled early — that makes it MORE true, not
false.

Mar 13: judge correctly recognizes the prediction was fulfilled ahead of schedule.
Reasoning cites UN data confirming India surpassed China in April 2023.

#### F5. Texas power grid — false → mostly_false (0.85)

Better calibration. The grid DID fail, Texas IS independent from federal
interconnections. The causal claim ("because they refused to connect") is the
oversimplification. `mostly_false` correctly captures that the individual facts
are true but the causal mechanism is wrong, vs Mar 11's blanket `false` which
didn't credit the true components.

#### F6. Unemployment "real number is 25%" — false → mostly_false (0.95)

Better handling of concession structure ("Sure, unemployment is low, but...").
The pipeline now acknowledges the valid premises (unemployment IS officially low,
discouraged workers DO exist) while correctly rejecting the 25% figure (U-6 is
~7-8%).

#### F7. Opioid/Purdue/Sacklers — true → mostly_true (0.92)

Better causal calibration. "Started because" oversimplifies a multi-factor
epidemic (overprescription culture, regulatory failures, illicit fentanyl).
`mostly_true` acknowledges Purdue's well-documented role while flagging the
causal oversimplification. Specific figures ($10B, hundreds of thousands of
deaths) verified correctly.

### Improved calibration (2 claims)

#### I1. US national debt under every president — mixed (0.65) → true (0.95)

Was defensibly uncertain on Mar 11 due to metric ambiguity (nominal vs GDP
share). Now confidently correct — debt DID increase under every president from
Carter through Biden in nominal terms, including Clinton (surpluses, but debt
increased due to trust fund accounting). Well-sourced from FRED data.

#### I2. Switzerland neutrality since 1815 — false (0.95) → mostly_false (0.85)

Less harsh. Mar 11 fixated on a July 1815 boundary-month military action. Mar 13
cites the 1847 Sonderbund War (a genuine internal civil war) as a more
substantial counterexample. Still arguably too strong — "neutral in every
military conflict" typically refers to international conflicts, and Switzerland's
200-year international neutrality record is genuine — but the direction of change
is correct.

### Regressed (2 claims)

#### R1. Coconut vs shark deaths — unverifiable (0.65) → mostly_true (0.75)

Mar 11 correctly identified the "150 coconut deaths/year" figure as unverifiable
(traces to a single unverified source). Mar 13 accepts it at face value and
repeats the viral statistic. The pipeline should trace statistics to primary
sources — this is a regression in source-criticality.

#### R2. Nuclear HWE + replication — mostly_false (0.75) → false (0.85)

Both runs are wrong (should be `mostly_true` or `true`), but Mar 13 is more
wrong. The healthy worker effect IS well-documented and replicated internationally.
The judge conflates dose-response studies (showing radiation increases specific
cancer risk) with overall rate comparisons (which do show lower rates due to HWE).
These address different statistical measures and are not contradictory.

---

## Persistent Bugs (not fixed since March 11)

### P1. Sweden lockdowns — true (0.95), should be false/mostly_false

**Root cause: decomposer qualifier injection.** The decomposer softens "never
implemented ANY lockdown measures" to something more evaluable, losing the
absolute scope. The judge then finds the softened claim true. Sweden imposed
gathering bans (>8 people), high school closures, and business restrictions —
these ARE lockdown measures even if not a full stay-at-home order. The claim's
"never...any" language is falsifiable by any single restriction.

This was identified as Priority 2 in the Mar 11 fix plan (polarity + qualifier
preservation rules) but has not been implemented.

### P2. Canada coastline + land area — mostly_false (0.85), should be true

**Root cause: judge factual hallucination.** Both subclaims are true (Canada:
202,080 km coastline, second largest by land area at 9.09M km²). The judge
hallucinates that Canada ranks 4th by land area, behind Russia, China, and USA.
This is factually wrong — Canada IS second to Russia. The evidence likely
contains the correct figures, but the judge's reasoning overrides them with
incorrect model knowledge.

Mar 11 identified this as Priority 6 (explicit numbers rule) — not implemented.

---

## New Failures (not in March 11 analysis)

### N1. Fluoride/cancer — mostly_false (0.75), should be true/mostly_true

**Root cause: judge can't handle "absence of evidence" claims.** The claim
("There is no scientific evidence that fluoride... causes cancer") reflects the
WHO/CDC/ADA scientific consensus position. The judge finds a few isolated studies
with methodological issues and concludes "no evidence" is too absolute, rating
the claim `mostly_false`. But the scientific consensus IS that there is no
established causal link at recommended levels.

**Pattern:** Claims asserting the absence of something are structurally hard for
the pipeline. It finds any counter-signal and overweights it vs the consensus.

### N2. Sitting US president convicted — unverifiable (0.55), should be true

**Root cause: judge can't connect evidence to claim.** The evidence contains
Trump's May 2024 conviction, which explicitly describes him as "former
president." The claim asks about "sitting" presidents. The judge should conclude
no sitting president has been convicted (true), but instead punts to
`unverifiable` because it can't confidently assert the universal negative.

**Pattern:** Same absence-of-evidence issue as N1. The pipeline struggles with
claims that require confirming nothing exists.

### N3. Nuclear HWE + "proving safe" — mostly_true (0.75), should be mostly_false

**Root cause: decomposer fails to split embedded fallacy.** The claim has two
parts: (1) the statistical finding (workers had lower cancer rates — true, due
to HWE) and (2) the causal conclusion ("proving that low-level radiation exposure
is safe" — a non-sequitur). The decomposer treats the entire claim as one unit
instead of separating the factual claim from the fallacious reasoning, so the
`mostly_true` verdict validates the fallacious causal conclusion.

Contrast with the multi-step fallacy claim (France nuclear → Germany cancer),
which WAS correctly decomposed into 3 subclaims and correctly rated `false`.
The difference may be that the France/Germany claim was longer and more obviously
multi-part.

---

## Full Claim-by-Claim Results

### Regression Suite (20 claims)

| # | Claim | Verdict | Conf | SC Count | Assessment | Mar 11 |
|---|-------|---------|------|----------|------------|--------|
| 1 | Biden open border policy | mostly_false | 0.85 | 2 | CORRECT | mostly_false (0.95) |
| 2 | Pharma price-gouging | mostly_true | 0.90 | 2 | CORRECT | mostly_true (0.85) |
| 3 | America's infrastructure worst | mostly_false | 0.75 | 3 | CORRECT | mostly_false (0.90) |
| 4 | Texas power grid Uri | mostly_false | 0.85 | 3 | CORRECT (fixed) | false (0.95) |
| 5 | ExxonMobil climate denial | true | 0.95 | 2 | CORRECT | true (0.95) |
| 6 | Japan oldest population/elderly care | unverifiable | 0.55 | 2 | DEBATABLE | false (0.95) |
| 7 | Every Republican voted against insulin cap | true | 0.95 | 1 | DEBATABLE | true (0.95) |
| 8 | China currency/trade deficit | false | 0.95 | 3 | CORRECT (fixed) | parse failure |
| 9 | No Republican reduced deficit since Eisenhower | false | 0.95 | 1 | DEBATABLE | true (0.85) |
| 10 | Unemployment real number 25% | mostly_false | 0.95 | 3 | CORRECT (fixed) | false (0.95) |
| 11 | SpaceX $15B + Musk DOGE | true | 0.95 | 2 | CORRECT | true (0.95) |
| 12 | Murdoch climate skepticism | mostly_true | 0.85 | 1 | CORRECT | mostly_true (0.85) |
| 13 | WHO said COVID lab leak | false | 0.95 | 1 | CORRECT | false (0.95) |
| 14 | Israel apartheid international law | true | 0.95 | 3 | DEBATABLE | true (0.95) |
| 15 | Gun control 40% reduction | mostly_false | 0.75 | 3 | CORRECT | mostly_false (0.85) |
| 16 | IRA $369B climate investment | true | 0.95 | 2 | CORRECT | true (0.95) |
| 17 | Google/Meta fines vs taxes | mostly_true | 0.75 | 1 | DEBATABLE | unverifiable (0.55) |
| 18 | Amazon $15/hr, injury rate, Bezos | mostly_false | 0.85 | 3 | WRONG | mostly_false (0.85) |
| 19 | Opioid epidemic Purdue/Sacklers | mostly_true | 0.92 | 3 | CORRECT (fixed) | true (0.95) |
| 20 | 91% tax rate counterfactual | mostly_false | 0.85 | 3 | CORRECT | mostly_false (0.75) |

**Regression suite: 14 correct, 5 debatable, 1 wrong. 0 regressions.**

### Stress Suite (20 claims)

| # | Claim | Verdict | Conf | SC Count | Assessment | Mar 11 |
|---|-------|---------|------|----------|------------|--------|
| 21 | NASA budget increasing every year | false | 0.95 | 1 | CORRECT | false (0.95) |
| 22 | Brazil highest deforestation rate | mostly_true | 0.75 | 1 | CORRECT | mostly_true (0.85) |
| 23 | G7 universal healthcare except US | true | 0.95 | 2 | CORRECT | true (0.95) |
| 24 | Microplastics evidence exists | true | 0.95 | 1 | CORRECT | true (0.95) |
| 25 | Finland education + suicide rate | mostly_false | 0.75 | 2 | CORRECT | mostly_false (0.75) |
| 26 | Canada coastline + land area | mostly_false | 0.85 | 2 | WRONG (P2) | mostly_false (0.75) |
| 27 | Fluoride no evidence of cancer | mostly_false | 0.75 | 1 | WRONG (N1) | — |
| 28 | No sitting president convicted | unverifiable | 0.55 | 1 | WRONG (N2) | — |
| 29 | Sweden never any lockdowns | true | 0.95 | 1 | WRONG (P1) | true (0.95) |
| 30 | Great Wall visible from space | false | 0.95 | 2 | CORRECT | false (0.95) |
| 31 | Stanford 70% remote productivity | false | 0.95 | 2 | CORRECT | false (0.95) |
| 32 | Churchill democracy quote | mostly_false | 0.85 | 1 | DEBATABLE | — |
| 33 | Coconut vs shark deaths | mostly_true | 0.75 | 1 | WRONG (R1) | unverifiable (0.65) |
| 34 | US military > next 10 combined | true | 0.95 | 1 | DEBATABLE | true (0.95) |
| 35 | Rainwater collection illegal | false | 0.95 | 1 | CORRECT | false (0.95) |
| 36 | Drinking age 21 since 1980s | true | 0.95 | 2 | DEBATABLE | true (0.95) |
| 37 | Bezos $1M to everyone | false | 0.95 | 1 | CORRECT | false (0.95) |
| 38 | Global population doubled since 1970 | true | 0.95 | 1 | CORRECT | true (0.95) |
| 39 | ECB raised rates every 2025 meeting | false | 0.95 | 1 | CORRECT | false (0.95) |
| 40 | Great Barrier Reef 50% loss | mostly_true | 0.90 | 1 | CORRECT | mostly_true (0.85) |

**Stress suite: 12 correct, 3 debatable, 5 wrong. 1 regression (coconut).**

### Advanced Suite (20 claims)

| # | Claim | Verdict | Conf | SC Count | Assessment | Mar 11 |
|---|-------|---------|------|----------|------------|--------|
| 41 | UC Berkeley admissions bias | true | 0.95 | 1 | DEBATABLE | true (0.95) |
| 42 | Nuclear workers lower cancer (basic) | mostly_true | 0.75 | 1 | CORRECT | — |
| 43 | Chocolate → Nobel Prizes | true | 0.95 | 1 | DEBATABLE | — |
| 44 | Marijuana → traffic fatalities | true | 0.92 | 1 | DEBATABLE | — |
| 45 | Dresden bombing 200K civilians | false | 0.95 | 2 | CORRECT | false (0.95) |
| 46 | Tuskegee deliberately infected | false | 0.95 | 1 | CORRECT | false (0.95) |
| 47 | Organic food proven more nutritious | false | 0.92 | 1 | CORRECT (fixed) | parse failure |
| 48 | Spiders swallowed while sleeping | false | 0.95 | 1 | CORRECT | false (0.95) |
| 49 | US debt increased every president | true | 0.95 | 1 | CORRECT (improved) | mixed (0.65) |
| 50 | Nuclear HWE + "proving safe" | mostly_true | 0.75 | 1 | WRONG (N3) | — |
| 51 | More alive than ever died | false | 0.95 | 1 | CORRECT | false (0.95) |
| 52 | Lightning never strikes twice | false | 0.95 | 1 | CORRECT (fixed) | true (0.95) |
| 53 | Switzerland neutral since 1815 | mostly_false | 0.85 | 1 | DEBATABLE (improved) | false (0.95) |
| 54 | India surpasses China by 2030 | true | 0.95 | 1 | CORRECT (fixed) | mostly_false (0.95) |
| 55 | Nordic happiness + high taxes | true | 0.95 | 2 | CORRECT | true (0.95) |
| 56 | Cuba life expectancy vs US | mostly_false | 0.75 | 2 | DEBATABLE | — |
| 57 | France nuclear → lower cancer than Germany | false | 0.90 | 3 | CORRECT | — |
| 58 | Gender pay gap 84 cents same work | false | 0.95 | 2 | DEBATABLE | — |
| 59 | Mass shootings > days in 2025 | mostly_true | 0.75 | 1 | CORRECT | — |
| 60 | Nuclear HWE + replication | false | 0.85 | 3 | WRONG (R2) | mostly_false (0.75) |

**Advanced suite: 12 correct, 6 debatable, 2 wrong. 1 regression (nuclear HWE).**

---

## Error Pattern Analysis

### Pattern 1: Decomposer qualifier injection (2 claims)

**Affected:** Sweden lockdowns (P1), nuclear HWE + "proving safe" (N3)

The decomposer softens absolute language or fails to split embedded logical
fallacies from factual claims. "Never implemented ANY lockdown measures" gets
weakened; "lower cancer rates, proving radiation is safe" gets treated as one
unit instead of fact + fallacy.

**Evidence:** The France/Germany nuclear claim (3-part, longer) WAS correctly
decomposed, suggesting the issue is specific to shorter embedded fallacies
where the boundary between fact and conclusion is less obvious.

### Pattern 2: Judge factual hallucination (2 claims)

**Affected:** Canada land area (P2), coconut deaths (R1)

The judge overrides evidence with incorrect model knowledge. For Canada, it
hallucinates land area rankings despite evidence likely containing correct
figures. For coconut deaths, it accepts a viral statistic at face value instead
of tracing to (non-existent) primary sources.

### Pattern 3: Absence-of-evidence claims (2 claims)

**Affected:** Fluoride/cancer (N1), sitting president convicted (N2)

Claims asserting something does NOT exist are structurally hard for the pipeline.
The judge finds any counter-signal (a single disputed study, a tangentially
related conviction) and overweights it against the consensus/exhaustive check.
The pipeline lacks a framework for evaluating universal negatives — "no X has
ever Y" requires confidence in exhaustive search, not finding one maybe-example.

### Pattern 4: Nuclear worker effect conflation (2/3 wrong)

**Affected:** HWE + "proving safe" (N3), HWE + replication (R2)

The judge conflates two distinct statistical measures: (1) overall mortality
rates (lower due to healthy worker effect — true) and (2) dose-response
relationships for specific cancers (positive — also true). These are NOT
contradictory but the judge treats dose-response findings as refuting the
overall rate claim.

The basic HWE claim (without causal extension) passes correctly.

### Pattern 5: Overclaiming on contested claims (3 debatable)

**Affected:** Israel apartheid, US military spending, marijuana/traffic

`true (0.95)` for claims where authoritative sources disagree or exact figures
fluctuate. The pipeline needs better calibration for contested categories —
cap confidence at 0.85 and prefer `mostly_true` when reasonable experts
disagree.

---

## Fix Plan — Priority Order

### Priority 1: Decomposer qualifier/polarity preservation

**Claims fixed:** P1 (Sweden), N3 (nuclear "proving safe")
**Effort:** Low (prompt edit)
**Risk:** Low

Add to decompose prompt:

```
QUALIFIER PRESERVATION (CRITICAL):
- NEVER add qualifiers, hedges, or scope narrowers not in the original claim.
- "Never implemented ANY lockdown measures" ≠ "did not implement NATIONWIDE
  lockdown measures." Preserve absolute language exactly.

EMBEDDED CONCLUSIONS:
- When a claim contains both a factual assertion AND a causal/logical
  conclusion ("X happened, proving Y"), decompose into separate subclaims:
  one for the factual assertion, one for the conclusion.
- The judge must evaluate the conclusion independently of the fact.
```

### Priority 2: Judge explicit numbers rule

**Claims fixed:** P2 (Canada land area)
**Effort:** Low (prompt edit)
**Risk:** Low

Add to judge prompt:

```
EXPLICIT NUMBERS RULE:
- When evidence provides specific figures (areas, populations, amounts), state
  the actual numbers in your reasoning before drawing ranking/comparison
  conclusions. Do NOT rely on general knowledge to interpret rankings.
```

### Priority 3: Judge absence-of-evidence framework

**Claims fixed:** N1 (fluoride), N2 (sitting president)
**Effort:** Medium (prompt edit + potentially research changes)
**Risk:** Medium — may cause over-acceptance of absence claims

Add to judge prompt:

```
ABSENCE CLAIMS ("no evidence exists", "no X has ever Y"):
- These require evaluating the QUALITY OF THE SEARCH, not finding counter-
  examples.
- If systematic reviews / authoritative bodies (WHO, CDC, major institutions)
  conclude no evidence exists, that IS evidence supporting the absence claim.
- A single disputed study with methodological concerns does not negate a
  consensus of absence.
- For historical universals ("no president has ever"), treat exhaustive
  historical records as sufficient evidence unless a clear counterexample
  is found. Tangentially related events are not counterexamples.
```

### Priority 4: Judge dose-response vs overall-rate distinction

**Claims fixed:** R2 (nuclear replication), improves N3
**Effort:** Low (prompt edit)
**Risk:** Low

Add to judge prompt:

```
DISTINGUISHING RELATED FINDINGS:
- Conflicting findings about DIFFERENT statistical measures do not
  automatically contradict each other.
- "Workers have lower overall mortality" (healthy worker effect) and
  "radiation increases specific cancer risk" (dose-response) are BOTH
  true simultaneously. They address different questions.
- Evaluate whether evidence addresses the SPECIFIC claim being made, not
  whether a related but different finding exists.
```

### Priority 5: Judge contested-category calibration

**Claims fixed:** Israel apartheid, US military, marijuana/traffic
**Effort:** Low (prompt edit)
**Risk:** Medium — may cause under-confidence on clear claims

Add to judge prompt:

```
CONTESTED CATEGORIES:
- For claims involving contested legal/political classifications where
  authoritative bodies disagree or no binding determination exists:
  prefer mostly_true/mostly_false over true/false, cap confidence at 0.85.
- For claims with fluctuating exact figures ("more than the next 10"):
  if the DIRECTION is clearly true but the exact number varies by source/year,
  use mostly_true rather than true.
```

### Priority 6: Source-criticality for viral statistics

**Claims fixed:** R1 (coconut deaths)
**Effort:** Medium (prompt edit or research-level change)
**Risk:** Low

Add to judge prompt:

```
VIRAL STATISTICS:
- When a specific statistic appears across many sources but all trace to the
  same original claim (no primary study, no institutional data), treat it as
  unverified regardless of how many secondary sources repeat it.
- Ask: "Is there a primary study, government dataset, or institutional report
  that independently measured this?" If not, the statistic is unverifiable.
```

### Priority 7: Decompose non-determinism

**Affected:** Hubble "ineffective" vs "not justified" (observed during pre-batch)
**Effort:** Medium-high (structural change or prompt reinforcement)
**Risk:** Medium

The decomposer produces different subclaim wordings across runs, cascading
into different verdicts. Temperature is 0 but quantized models still have
variance. Options:
- Add stronger decompose prompt rules about preserving claim language
- Add a determinism check (hash claim → seed) if the inference engine supports it
- Accept non-determinism but add guardrails (e.g., subclaim must contain key
  terms from original claim)

---

## Implementation Order

| # | Fix | Claims Fixed | Effort | Risk | Status |
|---|-----|-------------|--------|------|--------|
| 1 | Decompose: qualifier preservation + embedded conclusions | P1, N3 | Low | Low | **DONE** — decompose rules 14+15 |
| 2 | Judge: explicit numbers rule | P2 | Low | Low | **OPEN** |
| 3 | Judge: absence-of-evidence framework | N1, N2 | Medium | Medium | **OPEN** |
| 4 | Judge: dose-response vs overall-rate distinction | R2, N3 | Low | Low | **OPEN** |
| 5 | Judge: contested-category calibration | 3 debatable | Low | Medium | **OPEN** |
| 6 | Judge: viral statistics source-criticality | R1 | Medium | Low | **OPEN** |
| 7 | Decompose: non-determinism mitigation | Hubble variance | Medium-high | Medium | **OPEN** |

Total prompt additions: ~600 words across decompose + judge prompts.
Priorities 1-4 are low-risk, high-impact. Priority 5-6 are calibration tuning.
Priority 7 is a deeper structural issue.

## Implementation Batching Strategy

Changes batched into 3 rounds to maintain cause-and-effect traceability.
Each round targets non-overlapping failure modes so regressions can be
attributed by claim category.

### Round 1 — High-impact, independent failure modes (P1 + P2 + P3)

**Changes:**
- P1: Decompose qualifier preservation + embedded conclusions (decompose prompt)
- P2: Judge explicit numbers rule (judge prompt, quantitative claims)
- P3: Judge absence-of-evidence framework (judge prompt, evidence-gap claims)

**Rationale:** These three fix the most failures (~6 of 10 wrong verdicts),
touch different code paths (decompose vs judge), and target non-overlapping
claim types. If something regresses, the claim category identifies which fix
caused it.

**Validation claims:** Sweden lockdowns (P1), nuclear HWE+"proving safe" (P1),
Canada coastline (P2), fluoride/cancer (P3), sitting president convicted (P3),
Great Wall (stability check), Hubble (stability check).

### Round 2 — Judge calibration refinements (P4 + P5 + P6)

**Changes:**
- P4: Judge dose-response vs overall-rate distinction (judge prompt)
- P5: Judge contested-category calibration (judge prompt)
- P6: Judge viral statistics source-criticality (judge prompt)

**Rationale:** All subtle judge calibration tweaks targeting edge cases. Lower
blast radius — they add guidance for specific scenarios without changing core
reasoning logic. Each targets a different claim type (nuclear/statistical,
political/legal, viral myths).

**Validation claims:** Nuclear HWE+replication (P4), Israel apartheid (P5),
US military spending (P5), marijuana/traffic (P5), coconut deaths (P6),
organic food (stability check).

### Round 3 — Structural (P7 alone)

**Changes:**
- P7: Decompose non-determinism mitigation (structural/prompt change)

**Rationale:** Changes behavior across ALL claims, so it must run solo with the
full 60-claim suite to catch regressions. This is the highest-risk change and
needs clean baseline comparison.

**Validation:** Full 60-claim suite.

---

## Appendix: Evidence & Citation Statistics

### Evidence per claim (from DB)

| Metric | Value |
|--------|-------|
| Total evidence items stored | 1,042 |
| Average per claim | 17.4 |
| Average per subclaim | 16.8 |
| Max per subclaim | 20 (judge cap) |
| Min per subclaim | 6 |

### Citation extraction (synthesize)

Multi-fact claims (those that went through synthesize) had evidence digests
built from judge-cited sources. The [N] citation format was used consistently
in synthesis reasoning, with citations extracted and mapped to evidence URLs
in the verdict JSONB column.

### Research agent timeouts

All research agents hit the 120s soft timeout, falling back to partial evidence.
Evidence counts were still substantial (6-37 items per subclaim before dedup).
This is the expected behavior — the timeout captures whatever the agent gathered.
