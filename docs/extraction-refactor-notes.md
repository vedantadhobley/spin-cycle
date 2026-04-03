# Extraction Refactor Notes (2026-04-03)

## Architecture: Two-Pass Extraction

Split extraction into two focused LLM passes per chunk:

**Pass 1 (Grouping + Disposition):** Group sentences by rhetorical paragraph, label
each group claim/not_claim. No thesis writing. Model only decides structure.

**Pass 2 (Context Injection):** Take each claim group's raw sentences + surrounding
context, resolve pronouns/references to make them standalone. Editing, not writing.

**Why two passes:** Single-pass extraction asked the LLM to simultaneously classify,
group, AND write decontextualized theses. The writing burden caused fabrication on
borderline sentences (S3 "It was quite something" → Artemis thesis) and flattened
hypotheticals into factual assertions. Separating grouping from writing lets each
pass focus on one thing.

### Key files
- Schemas: `src/schemas/llm_outputs.py` — `GroupingOutput`, `ContextInjectionOutput`
- Prompts: `src/prompts/extraction.py` — `GROUPING_SYSTEM/USER`, `CONTEXT_INJECT_SYSTEM/USER`
- Logic: `src/transcript/thesis_extractor.py` — `extract_dispositions()`, `inject_context()`, `_build_theses()`
- Validators: `src/llm/validators.py` — `validate_grouping()`, `validate_context_injection()`
- Activities: `src/activities/transcript_activities.py` — `extract_dispositions_activity`, `inject_context_activity`
- Workflow: `src/workflows/extract_claims.py` — two-phase orchestration
- Config: `src/config.py` — `TARGET_SENTENCES_PER_CHUNK=50`, `TIMEOUT_INJECT_CONTEXT=600`

### Pipeline phases
1. **Pass 1** — group + classify (parallel pairs, sem=2)
2. **Pass 2** — context inject (parallel pairs, skip chunks with 0 claims)
3. **Classify** — checkable/not_checkable + factual_anchor + topic
4. **Dedup** — per-speaker embedding clustering (cosine > 0.85)
5. **Synthesize** — merge multi-member dedup groups into overarching claims

## Run History (singjupost Iran address, 186 sentences)

### Runs 1-4: Single-pass extraction (old architecture)
- Run 1: 89 theses, grouped format, failures
- Run 2: 103 theses, one-per-sentence format, over-aggressive not_claim
- Run 3: 123 theses, clean prompt, word-based chunks
- Run 4: 160 theses, sentence-based chunks, topic moved to classification, 142 final

### Run 5: Two-pass extraction (first run)
- **137 claims stored, 0 retries across both passes (all 8 LLM calls first attempt)**
- Pass 1: ~6.5 min, Pass 2: ~5.5 min, total ~12 min
- 17 not_claim sentences, 0 cross-chunk duplicates

**Not_claims (17 total):**
- 10 correct (greetings, closings, blessings, contentless filler)
- 4 borderline (S17 vague, S27 policy, S29 personal vow, S36 characterization)
- 3 genuine misses: S38 "forty-seven years" (number), S46 "terminated Iran deal" (action), S48 "took it out of banks" (action)

**Decontextualization quality:** Excellent. Pronouns resolved correctly ("I" → "President Trump", "they" → "Iran"), dates injected ("April 1, 2026"), references grounded. No fabrication in Pass 2 output.

**Issues found:**
1. **Over-atomization in chunks 0-1:** Zero multi-sentence grouping. Every sentence got its own group. Chunk 2 grouped well (14 multi-sentence groups), chunk 3 decent (4). Root cause: prompt said "new assertion = new group" which splits enumeration lists and emphasis chains.
2. **Filler-as-claim:** S5 "It was quite something" → "The launch of Artemis Two was a significant event." Not fabricated (good), but not a real claim either. Several similar: "Everyone is talking about it", "Every one of them", "Nobody's ever seen anything like it." These are rhetorical emphasis that got pulled in as standalone claims.
3. **Both issues are one problem:** Over-atomization prevents filler from being absorbed into the adjacent claim group. If S5 is in the same group as the Artemis sentences, context injection produces one clean Artemis claim and the filler disappears. The grouping granularity is the root cause.

**Fix applied:** Reframed grouping from per-assertion to per-rhetorical-paragraph. Key changes:
- "A group is a rhetorical paragraph — one point or one argument"
- "New group only when speaker shifts to fundamentally different subject"
- "Short sentences (~<10 words) almost never introduce new subjects — default to current group"
- Not_claim: "entire group contains zero verifiable information"
- Removed per-sentence sequential procedure (was reinforcing atomization)

### Run 6: Two-pass extraction (rhetorical paragraph grouping)
- **58 claims stored, 0 retries across both passes (all 8 LLM calls first attempt)**
- Pass 1: ~4.5 min, Pass 2: ~3.5 min, total ~8 min
- 4 not_claim sentences, 0 cross-chunk duplicates

**Grouping quality by chunk:**
| Chunk | Sentences | Groups | Sent/Group |
|-------|-----------|--------|-----------|
| 0 | 50 | 12 | 4.2 |
| 1 | 50 | 21 | 2.4 |
| 2 | 50 | 10 | 5.0 |
| 3 | 36 | 18 | 2.0 |

**Not_claims (4 total):** All correct — S0 greeting, S1 greeting, S184 blessing, S185 closing. The 3 genuine misses from Run 5 (S38, S46, S48) are now correctly captured as claims.

**Decontextualization quality:** Excellent. No fabrication. Pronouns resolved correctly throughout. Filler sentences like S3 "It was quite something" and S5 "It's amazing" absorbed into the Artemis group — context injection produces one clean Artemis claim and the filler disappears naturally.

**Multi-sentence grouping highlights:**
- Claim 52: 7 sentences (war duration comparisons) → single coherent claim
- Claim 34: 8 sentences (economy stats) → single claim preserving all figures
- Claim 35: 9 sentences (oil production) → single claim
- Claim 7: 6 sentences (Iran's 47-year history) → single claim with all cited events

**Remaining edge cases (expected, not bugs):**
- Claims 36, 39, 53 all say "Iran is decimated" — repeated at different points in the speech, correctly grouped as separate adjacent claims. Semantic dedup downstream handles this.
- Claims 55 vs 57 ("world is watching") — cross-chunk boundary (chunk 2 vs chunk 3). Grouping is intentionally local/adjacent within chunks; dedup merges semantic duplicates across chunks.
- Claims 23, 49, 54 — pure rhetoric ("extraordinary", "unstoppable", "investment in your children"). Correctly extracted as claims; downstream classifier marks as not_checkable.

**Run 5 → Run 6 comparison:**
| Metric | Run 5 | Run 6 |
|--------|-------|-------|
| Claims | 137 | 58 |
| Not_claims | 17 | 4 |
| Not_claim accuracy | 10/17 | 4/4 |
| Avg sent/group | ~1.0 | ~3.0 |
| Filler-as-claim | Yes | No |
| Missed claims | 3 | 0 |
| Fabrication | None | None |
| Retries | 0 | 0 |

## Design Decisions

### Grouping is local/adjacent, dedup is semantic

Pass 1 grouping only merges adjacent sentences within a chunk. When a speaker repeats the same assertion at different points in the speech (e.g., "Iran is decimated" appears 3 times), each occurrence is a separate claim group. This is intentional:

- **Grouping** answers: "which consecutive sentences form one rhetorical paragraph?"
- **Dedup** answers: "which claims across the entire transcript say the same thing?"

These are different questions. Grouping is structural (adjacency). Dedup is semantic (embedding similarity). Keeping them separate means each claim preserves its original context and quote location, and dedup can later decide which to keep and which research can be reused across duplicates.

Cross-chunk duplicates (like "world is watching" split across chunks 2 and 3) are also handled by dedup, not grouping — the model can't see across chunk boundaries.

## Key Learnings

1. **LLMs are bad at doing multiple things simultaneously.** Single-pass (classify + group + write thesis) caused the model to fabricate content for borderline sentences. Two-pass (group only → write only) eliminated fabrication.
2. **Per-assertion grouping doesn't work for political speeches.** Trump's style is rapid-fire short assertions. "Their navy is gone. Their air force is gone." = two assertions but one argument. Need paragraph-level grouping.
3. **Short sentences are the failure mode.** 4-7 word sentences get atomized into standalone groups even when they're clearly continuation/emphasis. The short-sentence heuristic addresses this directly.
4. **Null handling matters.** Model returns `null` for optional fields (reason, speakers) on groups where they're not applicable. Pydantic validators with `mode="before"` coerce nulls to defaults.
5. **Grouping and dedup solve different problems.** Grouping is local adjacency within chunks. Dedup is global semantic similarity. Trying to make grouping do dedup's job would require the model to remember the entire transcript, which it can't across chunks.

## TODO
- [ ] Fix classification index matching bug (matched=0, pre-existing)
- [ ] Evaluate whether chunk size (50 sentences) is optimal for grouping quality
- [ ] Run full pipeline (classify + dedup + synthesize) to verify dedup catches repeated assertions
