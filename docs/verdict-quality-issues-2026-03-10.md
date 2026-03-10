# Verdict Quality Issues — 2026-03-10 Test Run

5 claims tested after SearXNG removal + prefetch enablement. 3 have verdict
quality issues to address. 1 good result. 1 technically-correct-but-funny result.

---

## 1. SpaceX/DOGE — Decomposition Misinterprets Juxtaposition

**Claim**: "Elon Musk's SpaceX has received over $15 billion in government contracts
while he simultaneously runs DOGE, which aims to cut $2 trillion in government spending"

**Verdict**: mostly_false (should be **mostly_true**)

### Problem

The claim is a juxtaposition/irony claim: Musk receives government money AND works
to cut government spending. The point is the coexistence of these two facts, not
whether DOGE actually succeeded at cutting $2T.

The decomposer splits it into sub-claims including one evaluating whether DOGE has
actually cut $2T in spending — which it hasn't (and the claim only says it "aims to").
This drags the overall verdict to mostly_false.

### Root Cause

Decomposition instability — the same claim produces different sub-claims across runs.
In some runs the decomposer correctly treats "aims to cut" as a stated goal (true —
DOGE does aim for this). In other runs it treats it as an accomplished fact (false —
DOGE hasn't achieved $2T in cuts).

### Proposed Fix

1. **Normalize "aims to" / "seeks to" / "plans to"**: The normalizer should preserve
   intentional language. "Aims to cut $2T" should decompose as "DOGE's stated goal is
   to cut $2T in spending" not "DOGE has cut $2T in spending."

2. **Juxtaposition detection in decompose**: When a claim presents two facts joined
   by "while" / "simultaneously" / "even as", the decomposer should recognize this
   as a contrast/coexistence claim. The sub-claims should verify each fact
   independently, not evaluate whether the juxtaposition implies a conclusion.

3. **Decompose prompt guidance**: Add explicit rules for:
   - Intentional verbs (aims, seeks, plans, proposes) → verify the INTENT, not the outcome
   - Contrast connectors (while, simultaneously, even as, despite) → verify each side
     independently

### Priority: HIGH
This pattern affects any claim about stated goals, aspirations, or contrasting facts.

---

## 2. Pelosi — Absolutist Language Over-Penalization

**Claim**: "Nancy Pelosi has outperformed virtually every major hedge fund over the
past two decades while serving as a member of Congress with access to insider
information"

**Verdict**: mostly_false (should be **mostly_true**)

### Problem

The judge treats the gap between "virtually every" and "almost every" as a
meaningful factual distinction warranting a mostly_false verdict. The evidence
shows Pelosi's stock returns significantly outperformed most major hedge funds
(~65% annualized during some periods vs S&P average), which is the core claim.

The judge also applies strict timeframe attribution — Pelosi's career-spanning
trading record ($130M+ in reported trades) is questioned because some trades
occurred outside her tenure as Speaker, even though the claim says "member of
Congress" (which she's been since 1987).

### Root Cause

1. **Absolutist language sensitivity too high**: "Virtually every" ≈ "almost every"
   ≈ "the vast majority of". The judge treats minor quantifier distinctions as
   factual failures rather than rhetorical imprecision.

2. **Timeframe over-strictness**: The claim says "member of Congress" but the judge
   seems to anchor on her Speaker tenure for some trading data.

### Proposed Fix

1. **Quantifier equivalence in judge prompt**: Add guidance that rhetorical
   quantifiers like "virtually every", "almost every", "nearly all" are
   equivalent for fact-checking purposes. The key question is whether the
   DIRECTIONAL claim is true (did she outperform most hedge funds?), not
   whether "virtually" vs "almost" is precisely correct.

2. **Normalizer quantifier softening**: During normalization, "virtually every" →
   "almost every" or similar. Strip rhetorical amplification before decomposition
   so the judge evaluates the substance, not the rhetoric.

3. **Direction-first evaluation**: The judge prompt already has partial-data
   guidance for quantitative claims. Extend this to comparative claims: if the
   direction is clearly supported (outperformed most), that's mostly_true even
   if the exact scope ("every" vs "most") is slightly off.

### Priority: MEDIUM
Affects claims with rhetorical quantifiers (common in political/economic claims).

---

## 3. Google/Meta — Research Scope Too Narrow (Ireland-Focused)

**Claim**: "Google and Meta have collectively avoided paying over $50 billion in
taxes worldwide through legal loopholes and offshore profit shifting"

**Verdict**: mostly_true (0.72 confidence — weak)

### Problem

The research phase found heavily Ireland-focused sources:
- `finfacts-blog.com` — Irish financial analysis
- Irish Data Protection Commission fine articles
- Multiple articles about Ireland's corporate tax regime

While Ireland IS part of the tax avoidance story, the claim is about
**worldwide** avoidance by **two specific companies**. The research should have
found:
- SEC filings / annual reports showing effective tax rates
- Congressional/Senate hearing transcripts on tech company tax practices
- EU Commission state aid decisions (Apple €13B, etc.)
- OECD BEPS (Base Erosion and Profit Shifting) reports
- US-specific data: TCJA impact, profit repatriation figures

### Root Cause

1. **Single combined sub-claim**: The claim was treated as one atomic sub-claim
   rather than splitting into per-company facts. A per-company split would have
   produced more targeted research:
   - "Google has avoided paying over $25B in taxes through legal loopholes"
   - "Meta has avoided paying over $25B in taxes through legal loopholes"
   - "Combined avoidance exceeds $50B"

2. **Seed query quality**: The LLM-generated seed queries may have been too
   generic ("tech company tax avoidance") rather than targeted ("Google
   effective tax rate SEC filing", "Meta offshore profit shifting EU ruling").

3. **Source diversity not enforced geographically**: The ranker ensures domain
   diversity but not geographic/topic diversity. Multiple Ireland-focused
   articles from different domains all pass the diversity check.

### Proposed Fix

1. **Decompose multi-entity comparative claims**: When a claim names multiple
   entities + "collectively" / "combined" / "together", decompose into
   per-entity sub-claims + an aggregation fact.

2. **Seed query specificity**: Add guidance to the decompose prompt that seed
   queries for financial claims should target primary sources (SEC, EU
   Commission, OECD) not just news coverage.

3. **Geographic diversity in ranking** (future): Track geographic focus of
   evidence and penalize over-concentration in one jurisdiction for claims
   about "worldwide" phenomena.

### Priority: MEDIUM
Affects multi-entity claims and global-scope claims.

---

## 4. China Renewables — Good Result

**Claim**: "China has invested more in renewable energy than the United States
and European Union combined over the past five years"

**Verdict**: mostly_true (good confidence)

### Notes

This run correctly found the Energy Institute source (`seasia.co` article
citing the data) after SearXNG removal. Previous run with SearXNG returned
garbage results (travel guides, generic encyclopedia pages) that crowded out
the key source.

The prefetch feature successfully pre-loaded high-quality articles, saving
agent tool budget for follow-up research.

**No action needed** — this is the desired behavior.

---

## 5. Musk/DOGE Subsidies — Technically Correct (Amusing)

**Claim**: "Elon Musk's companies have received over $30 billion in government
subsidies while he advocates for reducing government spending through DOGE"

**Verdict**: mostly_false

### Notes

The judge determined that while Musk's companies have received substantial
government support ($15.4B in contracts, plus subsidies), the evidence doesn't
clearly support the "$30 billion" figure. The reasoning is technically sound —
the specific dollar amount is likely inflated.

However, the DIRECTIONAL claim (Musk receives government money while advocating
spending cuts) is clearly true. This is the same juxtaposition/irony pattern as
issue #1 above.

**Verdict should be**: mostly_true (the irony/juxtaposition is factual, the
exact figure is debatable but the order of magnitude is correct).

This shares the same root cause as issue #1 and would be fixed by the same
decomposition improvements.

---

## Summary of Fixes Needed

| Issue | Component | Priority | Effort |
|-------|-----------|----------|--------|
| Juxtaposition/irony decomposition | decompose prompt | HIGH | Medium |
| Intentional verb handling (aims/seeks) | normalizer + decompose | HIGH | Small |
| Quantifier equivalence (virtually≈almost) | judge prompt + normalizer | MEDIUM | Small |
| Multi-entity claim splitting | decompose prompt | MEDIUM | Medium |
| Seed query targeting for financial claims | decompose prompt | MEDIUM | Small |
| Geographic diversity in ranking | evidence_ranker.py | LOW | Large |

## Related Documentation

- `docs/seed-relevance-filtering.md` — proposed keyword gate + embedding
  similarity for topical relevance filtering (addresses garbage search results)
- `.gov` scoring adjustment also proposed there (GOV_TLD_SCORE 15→5,
  FACTUAL_UNRATED_GOV 20→12)
