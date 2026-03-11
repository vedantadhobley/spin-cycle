# Rubric-Based Prompt Restructure Plan

## Status

| Phase | Target | Status | Date |
|-------|--------|--------|------|
| Phase 1: Judge + Synthesize | Rubric prompts + schemas + validators | **COMPLETE** | 2026-03-10 |
| Phase 2: Decompose | Lighter-touch restructure | PLANNED | — |
| Phase 3: Normalize + Research | Polish only | PLANNED | — |

## Principle

Move from "long list of rules the model may ignore" to "structured output
steps that enforce the rules inherently." The model can't skip a step when
it must fill in a structured field for that step.

**Key constraint**: Do NOT remove information that was carefully tested and
proven important. Restructure and compress — don't delete. Every piece of
guidance must either (a) survive in the rubric prompt, (b) be embedded in
the output structure, or (c) be explicitly justified as redundant.

---

## Phase 1: JUDGE Prompt (26.4K → target ~12-14K) — COMPLETE

### Content Audit — Every Section Categorized

| Lines | Section | Chars | Disposition |
|-------|---------|-------|-------------|
| 1088-1098 | Role + conciseness | ~400 | **KEEP** — essential framing |
| 1100-1112 | Original claim context | ~500 | **KEEP** — critical for subclaim interpretation |
| 1114-1139 | Colloquial language (attribution, quantifiers, understatement) | ~1100 | **EMBED IN RUBRIC** — Step 1 (interpret claim) forces charitable reading. Specific rules (attribution, quantifiers, understatement) move to Step 4 as precision-check guidance |
| 1141-1158 | Quantitative claims | ~800 | **EMBED IN RUBRIC** — Step 4 (precision check). "Show your work" becomes a structured field |
| 1160-1198 | Evidence hierarchy + timeline facts | ~1600 | **KEEP compressed** — hierarchy moves to Step 2 intro. Timeline warning stays as-is (proven critical) |
| 1200-1211 | Source rating tags | ~500 | **KEEP compressed** — reference guide for reading tags. Can be 60% shorter |
| 1213-1218 | Government sources | ~300 | **EMBED** — Step 2's `is_independent` field handles this |
| 1220-1277 | Self-serving statements + trace origin + circular evidence | ~3000 | **EMBED IN STRUCTURE** — Step 2's `is_independent: bool` field on each evidence assessment FORCES this classification. The 3K of prose reduces to ~200 chars of Step 2 instruction + the field itself. This is the biggest win. |
| 1279-1317 | Automated self-serving detection (warning tags) | ~1800 | **KEEP compressed** — model needs to know what the tags mean, but examples can be shorter. ~800 chars |
| 1319-1372 | Legal/regulatory claims + regulatory anomaly detection | ~2500 | **MOVE TO APPENDIX** — only relevant for legal claims (~10% of total). Include as conditional guidance: "If this is a legal/regulatory claim, also consider..." Saves ~2K for non-legal claims |
| 1374-1467 | Rhetorical traps (9 patterns) | ~4200 | **COMPRESS to reference list** — each pattern is currently 4-8 lines of explanation. The model knows what cherry-picking, correlation≠causation, etc. mean. Compress to 1-2 lines each with just the detection instruction. ~1500 chars |
| 1472-1500 | Verdict scale + boundary rule | ~1400 | **KEEP** — proven important, recently updated |
| 1503-1506 | Output quality | ~200 | **KEEP** — short and important |
| 1508-1516 | Return format | ~300 | **REPLACE** — new schema with rubric fields |

### New Judge Prompt Structure

```
ROLE + DATE (keep)                                    ~400 chars
ORIGINAL CLAIM CONTEXT (keep)                         ~500 chars
EVIDENCE TAGS REFERENCE (compressed)                  ~500 chars
WARNING TAGS REFERENCE (compressed)                   ~800 chars

STEP 1 — INTERPRET THE CLAIM                          ~600 chars
  Restate the sub-claim charitably. Consider original
  claim context. If language is colloquial, state what
  a reasonable person would understand.
  → Output: claim_interpretation (str)

STEP 2 — TRIAGE KEY EVIDENCE                          ~800 chars
  Identify the 3-5 most relevant evidence items.
  For each: supports/contradicts/neutral + is_independent.
  Evidence hierarchy: primary docs > reporting > statements.
  Official denials don't counter primary evidence.
  TIMELINE RULE: Don't assume roles without date evidence.
  → Output: key_evidence (list[EvidenceAssessment])

STEP 3 — ASSESS DIRECTION                            ~400 chars
  Based on independent evidence only, what direction
  does the evidence point? Ignore non-independent sources
  for direction assessment.
  → Output: evidence_direction (Literal[...])
           direction_reasoning (str)

STEP 4 — ASSESS PRECISION                            ~800 chars
  How precise is the claim vs the evidence?
  - Attribution: did they say it? (not: did they originate it?)
  - Quantifiers: verify direction, not exact scope
  - Understatement: lower than reality = supports the claim
  - Quantitative: show arithmetic if comparing numbers
  - Superlatives: failing a superlative with direction correct
    = mostly_false, not false
  → Output: precision_assessment (str)

STEP 5 — RENDER VERDICT                              ~1400 chars
  Verdict scale (keep current definitions + boundary rule)
  Confidence calibration (compressed)
  Derive verdict from Steps 3+4.
  → Output: verdict, confidence, reasoning

RHETORICAL TRAPS REFERENCE (compressed to 1-2 lines each) ~1000 chars
  1. Cherry-picking: note if data point is unrepresentative
  2. Correlation≠causation: require mechanism evidence
  3. Definition games: note if truth depends on definition
  4. Time-sensitivity: note if true-then but not-now
  5. Survivorship bias: check if sources share common origin
  6. Statistical framing: note relative vs absolute
  7. Anecdotal vs systematic: one case ≠ pattern
  8. False balance: don't equate 1 dissenter with 10 corroborating
  9. Retroactive status: current title ≠ held role at event time

LEGAL CLAIMS ADDENDUM (only if claim is legal/regulatory) ~500 chars
  Legality ≠ legitimacy. Check selective enforcement,
  regulatory capture, letter vs spirit.

OUTPUT QUALITY (keep)                                  ~200 chars
RETURN FORMAT (new schema)                            ~400 chars
```

**Estimated total: ~8,400 chars** (vs 26,400 current = 68% reduction)

**Actual result: 8,533 chars** — within 2% of estimate.

### What Gets Preserved (nothing lost)

| Original Content | Where It Goes |
|-----------------|---------------|
| Original claim context rules | Step 1 intro |
| Colloquial language (attribution, quantifiers, understatement) | Step 4 precision guidance |
| Quantitative claims (show arithmetic, partial data) | Step 4 + structured field |
| Evidence hierarchy (primary > reporting > statements) | Step 2 intro |
| Timeline facts (role date verification) | Step 2 warning |
| Official denials don't counter primary evidence | Step 2 + Step 3 |
| Source rating tags | Reference section (compressed) |
| Self-serving detection (3K of prose!) | `is_independent` field on EvidenceAssessment |
| Warning tags (automated detection) | Reference section (compressed) |
| Legal/regulatory claims | Conditional addendum |
| Regulatory anomaly detection | Conditional addendum |
| 9 rhetorical traps | Compressed reference list |
| Verdict scale + boundary rule | Step 5 (kept verbatim) |
| Confidence calibration | Step 5 (compressed) |

### New Schema: JudgeOutput

```python
class EvidenceAssessment(BaseModel):
    """Assessment of a key evidence item."""
    source_index: int = Field(
        ..., description="Evidence item number from the provided list"
    )
    assessment: Literal["supports", "contradicts", "neutral"] = Field(
        ..., description="Does this evidence support or contradict the claim?"
    )
    is_independent: bool = Field(
        ..., description="Is this source independent from the claim subject? "
        "False if source IS the claim subject, quotes the claim subject, "
        "or has ownership ties to the claim subject."
    )
    key_point: str = Field(
        ..., description="1-2 sentences: what does this evidence say?"
    )


class JudgeOutput(BaseModel):
    """Rubric-based judge output with explicit reasoning chain."""

    # Step 1
    claim_interpretation: str = Field(
        ..., description="Charitable restatement of what the claim is asking"
    )

    # Step 2
    key_evidence: list[EvidenceAssessment] = Field(
        ..., description="Assessment of the 3-5 most relevant evidence items"
    )

    # Step 3
    evidence_direction: Literal[
        "clearly_supports", "leans_supports",
        "genuinely_mixed",
        "leans_contradicts", "clearly_contradicts",
        "insufficient"
    ] = Field(
        ..., description="Overall direction of independent evidence"
    )
    direction_reasoning: str = Field(
        ..., description="2-3 sentences explaining the direction assessment"
    )

    # Step 4
    precision_assessment: str = Field(
        ..., description="How precise is the claim? Where do specifics "
        "match or diverge from evidence?"
    )

    # Step 5
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ..., description="Public-facing explanation of the verdict"
    )
```

### Programmatic Validators

```python
def validate_judge_consistency(output: JudgeOutput) -> list[str]:
    """Check for contradictions between rubric steps."""
    warnings = []

    # Direction-verdict consistency
    supports = {"clearly_supports", "leans_supports"}
    contradicts = {"leans_contradicts", "clearly_contradicts"}

    if output.evidence_direction in supports and output.verdict in ("false", "mostly_false"):
        warnings.append(
            f"Direction '{output.evidence_direction}' but verdict '{output.verdict}'. "
            "If evidence supports the claim's direction, verdict should not be "
            "false/mostly_false."
        )

    if output.evidence_direction in contradicts and output.verdict in ("true", "mostly_true"):
        warnings.append(
            f"Direction '{output.evidence_direction}' but verdict '{output.verdict}'. "
            "If evidence contradicts the claim, verdict should not be true/mostly_true."
        )

    # Independence check
    independent_evidence = [e for e in output.key_evidence if e.is_independent]
    if not independent_evidence and output.verdict in ("true", "false"):
        warnings.append(
            "No independent evidence identified but strong verdict given. "
            "Consider 'unverifiable' if all evidence is from interested parties."
        )

    return warnings
```

**Deploy permissively first**: log warnings, don't reject. Monitor against
40-claim regression suite. Tighten to enforcing after verification.

### Implementation Notes (Phase 1 Judge)

- Implemented in `src/schemas/llm_outputs.py`, `src/prompts/verification.py`,
  `src/agent/judge.py`, `src/llm/validators.py`
- Added `normalize_assessment` field validator to handle LLM quirk: model outputs
  `"supports (historical)"` instead of `"supports"` ~10% of the time. Strips
  parenthetical qualifiers before Literal validation.
- Added `normalize_direction` and `normalize_verdict` field validators for
  common LLM variations (e.g., "supports" → "clearly_supports", "mixed" → "genuinely_mixed")
- Consistency validator is permissive (log-only): direction-verdict alignment +
  independence check. Logged at WARNING level with `rubric_inconsistency` event.
- Rubric summary logged at INFO level with structured fields: direction,
  evidence_assessed, independent count, non_independent count, verdict, confidence.
- Tested against 9 problematic claims: 5 improved, 0 regressions.

---

## Phase 1: SYNTHESIZE Prompt (8.3K → target ~4-5K) — COMPLETE

### Content Audit

| Section | Chars | Disposition |
|---------|-------|-------------|
| Role + audience instructions | ~800 | **KEEP compressed** |
| "Weigh by importance, not count" + examples | ~1200 | **EMBED IN STRUCTURE** — Step 2 `subclaim_weights` forces this |
| Trust subclaim verdicts | ~600 | **KEEP** — important |
| Enumerated claims guidance | ~700 | **EMBED IN STRUCTURE** — `role` field on SubclaimWeight |
| Thesis usage | ~500 | **EMBED** — Step 1 `thesis_restatement` |
| Correlated evidence | ~300 | **KEEP compressed** |
| Conflicting findings | ~300 | **KEEP compressed** |
| Unverifiable elements | ~400 | **KEEP compressed** |
| Verdict scale + boundary rule | ~800 | **KEEP** — recently updated |
| Confidence scoring | ~400 | **KEEP compressed** |
| Contextual reasoning | ~300 | **KEEP compressed** |
| Output quality | ~200 | **KEEP** |

### New Synthesize Prompt Structure

```
ROLE + AUDIENCE (compressed)                          ~500 chars
TRUST SUBCLAIM VERDICTS (keep)                        ~400 chars

STEP 1 — IDENTIFY THE THESIS                          ~300 chars
  What is the speaker fundamentally arguing?
  → Output: thesis_restatement (str)

STEP 2 — CLASSIFY EACH SUBCLAIM                       ~500 chars
  For each sub-verdict, classify its role:
  - core_assertion: this IS the thesis
  - supporting_detail: example, attribution, secondary fact
  - background_context: widely-known fact for context
  → Output: subclaim_weights (list[SubclaimWeight])

STEP 3 — DOES THE THESIS SURVIVE?                     ~300 chars
  Based on CORE ASSERTION verdicts only.
  Wrong details don't flip true core assertions.
  → Output: thesis_survives (bool)

STEP 4 — RENDER VERDICT                               ~800 chars
  Verdict scale + boundary rule (keep)
  Confidence guidelines (compressed)
  Derive from Steps 2+3.
  → Output: verdict, confidence, reasoning

CORRELATED EVIDENCE + CONFLICTING FINDINGS (compressed) ~400 chars
UNVERIFIABLE HANDLING (compressed)                     ~300 chars
CONTEXTUAL REASONING (compressed)                      ~200 chars
OUTPUT QUALITY (keep)                                  ~200 chars
RETURN FORMAT (new schema)                            ~300 chars
```

**Estimated total: ~4,200 chars** (vs 8,300 = 49% reduction)

**Actual result: 4,441 chars** — within 6% of estimate.

### New Schema: SynthesizeOutput

```python
class SubclaimWeight(BaseModel):
    """Classification of a subclaim's importance."""
    subclaim_index: int = Field(
        ..., description="Which sub-verdict this refers to (1-indexed)"
    )
    role: Literal["core_assertion", "supporting_detail", "background_context"]
    brief_reason: str = Field(
        ..., description="Why this classification (1 sentence)"
    )


class SynthesizeOutput(BaseModel):
    """Rubric-based synthesis with explicit thesis evaluation."""

    # Step 1
    thesis_restatement: str = Field(
        ..., description="One sentence: what is the speaker arguing?"
    )

    # Step 2
    subclaim_weights: list[SubclaimWeight] = Field(
        ..., description="Role classification for each subclaim"
    )

    # Step 3
    thesis_survives: bool = Field(
        ..., description="Does the thesis hold given core assertion verdicts?"
    )

    # Step 4
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ..., description="Public-facing explanation (never reference "
        "sub-claim numbers or internal process)"
    )
```

### Programmatic Validators

```python
def validate_synthesize_consistency(output: SynthesizeOutput) -> list[str]:
    warnings = []

    if output.thesis_survives and output.verdict in ("mostly_false", "false"):
        warnings.append(
            "thesis_survives=True but verdict is negative. "
            "If the thesis holds, verdict should be true/mostly_true."
        )

    if not output.thesis_survives and output.verdict in ("true", "mostly_true"):
        warnings.append(
            "thesis_survives=False but verdict is positive. "
            "If the thesis fails, verdict should be mostly_false/false."
        )

    return warnings
```

---

## Phase 2: DECOMPOSE Prompt (23.4K + 7K patterns → target ~18-20K)

### Changes (lighter touch)

1. **Add `claim_relationship` to AtomicFact**:
   ```python
   claim_relationship: Literal[
       "core", "presupposition", "supporting_detail",
       "causal_link", "falsifying_condition"
   ] = Field(default="core")
   ```
   This feeds the synthesizer's classification as a hint.

2. **Compress worked examples**: The 8 full JSON examples (lines 717-831)
   can be condensed to ~60% of current size by removing redundant field
   explanations in surrounding prose. Keep all examples — just trim padding.

3. **Keep all 12 extraction rules** verbatim — these are from the
   literature (Pryzant et al., VeriScore, SAFE, AmbiFC, FActScore) and
   were carefully tested. DO NOT MODIFY.

4. **Keep linguistic patterns file** — but compress "How to handle"
   sections that duplicate what extraction rules already say.

5. **Keep seed query rules** — these were tested and proven important
   (especially rule 8 about not injecting training knowledge).

### What NOT to Change in Decompose

- The 12 extraction rules are sacred — from academic literature
- The seed query rules (especially rule 8)
- The evidence-need categories
- The interested parties analysis structure
- The worked examples (compress, don't remove)
- The linguistic patterns file (compress, don't remove)

---

## Phase 3: NORMALIZE + RESEARCH (polish only)

### Normalize (~5.6K → ~4.5K)
- Compress verbose transformation descriptions by ~20%
- No structural changes

### Research (~8.7K → ~7K)
- Trim source blocklist (handled programmatically by source_filter.py)
- Compress source tier descriptions
- No structural changes

---

### Implementation Notes (Phase 1 Synthesize)

- Implemented in `src/schemas/llm_outputs.py`, `src/prompts/verification.py`,
  `src/agent/synthesize.py`, `src/llm/validators.py`
- Added `normalize_role` field validator for common LLM variations
  (e.g., "core" → "core_assertion", "background" → "background_context")
- Consistency validator is permissive (log-only): thesis_survives vs verdict alignment.
- Rubric summary logged at INFO level: thesis (truncated), core/supporting/background
  indices, thesis_survives, verdict, confidence.
- Key win: Murdoch claim correctly weighted supporting details as non-core,
  preventing false detail from overriding true thesis.

### Remaining Issues After Phase 1

- **Parallel "and" assertions**: Claims like "Canada did X and Y" where X and Y
  are genuinely co-equal get classified as co-equal core assertions. When one is
  true and one is false, synthesizer must give "mixed" but currently may lean
  toward one side. Not a rubric issue — needs structural handling (Phase 2+).
- **Boundary cases**: Some claims sit on true/mostly_true or false/mostly_false
  boundaries where either verdict is defensible.

---

## Implementation Sequence

1. ~~Update `src/schemas/llm_outputs.py` with new schemas~~ ✅
2. ~~Rewrite `JUDGE_SYSTEM` in `src/prompts/verification.py`~~ ✅
3. ~~Update `src/agent/judge.py` to handle new JudgeOutput fields~~ ✅
4. ~~Add judge validators (permissive — log only)~~ ✅
5. ~~Rewrite `SYNTHESIZE_SYSTEM` in `src/prompts/verification.py`~~ ✅
6. ~~Update `src/agent/synthesize.py` to handle new SynthesizeOutput fields~~ ✅
7. ~~Add synthesize validators (permissive — log only)~~ ✅
8. ~~Test against 9 affected claims~~ ✅ (5/9 improved, 0 regressions)
9. If good: run full 40-claim regression
10. If good: tighten validators to enforcing

## Risk Mitigation

- **No content deletion without justification** — every piece of current
  guidance has a mapped destination in the new structure
- **Backward-compatible schemas** — new fields have defaults during migration
- **Permissive validators first** — log warnings, don't reject
- **Incremental testing** — 9 affected claims first, then full 40-claim suite
- **Rollback plan** — git revert if regression detected
