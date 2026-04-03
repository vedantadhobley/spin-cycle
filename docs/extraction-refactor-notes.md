# Extraction Refactor Notes (2026-04-03)

## What Was Done

### 1. One-Per-Sentence Output Format (COMPLETE)
Replaced grouped extraction (`claims: [{sentence_indices: [107,108,109]}]`) with
one-per-sentence dispositions. Each sentence gets exactly one entry. Multi-sentence
claims share a `claim_group` integer. Theses written once per group.

**Files changed:**
- `src/schemas/llm_outputs.py` — `SentenceDisposition`, `ClaimGroupThesis`, updated `SentenceExtractionOutput`
- `src/llm/validators.py` — Simplified validator: index coverage, group integrity, thesis quality downgrade
- `src/prompts/extraction.py` — Rewritten prompt (see below)
- `src/transcript/thesis_extractor.py` — `_convert_to_theses` groups by claim_group, updated logging

### 2. Cross-Chunk Index Mismatch Bug (FIXED, TESTED)
`extract_claims.py` returned `all_theses` (pre-dedup, 124 items) alongside `tc_ids`
(post-dedup, 123 items). Downstream `synthesize_claims.py` line 157 crashed with
`IndexError: list index out of range`.

**Fix:** In `extract_claims.py`, after cross-chunk dedup, filter `all_theses` to match
the deduped set before returning. **Verified working in Run 4** — full pipeline
through synthesis completed with no IndexError.

### 3. Prompt Rewrite (COMPLETE)
Single decision rule: "Does the sentence assert anything about the world?" → claim.
"Zero propositional content?" → not_claim. No examples from test transcripts. No
garbage-bin categories.

### 4. Sentence-Based Chunking (COMPLETE)
Switched from word-based (`TARGET_WORDS_PER_CHUNK=1500`) to sentence-based chunking.

**Config (`src/config.py`):**
- `TARGET_SENTENCES_PER_CHUNK = 45` — ~45 sentences per chunk
- `OVERLAP_SENTENCES = 15` — context sentences before/after target range
- `SPEAKER_CUTOFF_WINDOW = 5` — max sentences to extend past target for speaker boundary

**For 186-sentence transcript:** 4 chunks [45, 45, 45, 51], 2 rounds with --parallel 2.
Each chunk gets ~15 sentences of overlap context before and after (except first/last).

### 5. Topic Moved from Extraction to Classification (COMPLETE)
Topic assignment ("military", "economic", etc.) was removed from the extraction prompt
and `ClaimGroupThesis` schema. Topic is now assigned during the classification step
(`ClassifyAndDedupWorkflow` → `classify_claims_batch`), which already ran on every
claim for checkability. This eliminated the most repetitive field from extraction output.

**Why:** With `presence_penalty=0` (needed for exhaustive enumeration), repeating
"military" 40+ times in the groups array caused degenerate repetition on chunk 3
(10K+ tokens, 20+ minutes, never finished). Removing topic from extraction allowed
restoring the default `presence_penalty=1.5` from the Qwen3.5 model card.

**Files changed:**
- `src/schemas/llm_outputs.py` — removed `topic` from `ClaimGroupThesis`, added to `ClaimClassification`
- `src/prompts/extraction.py` — removed topic from output rules and JSON example
- `src/prompts/classification.py` — added Step 4 for topic, updated JSON example
- `src/transcript/thesis_extractor.py` — removed topic from `_convert_to_theses`, removed `presence_penalty=0` override
- `src/transcript/claim_classifier.py` — added topic to classification result passthrough
- `src/workflows/extract_claims.py` — removed topic from `_tag_theses_for_storage`
- `src/workflows/classify_and_dedup.py` — added topic merge from classification, added to DB update
- `src/activities/transcript_activities.py` — added topic persistence in classification update

## Test Results (4 runs on singjupost Iran address, 186 sentences)

### Run 1: Old prompt (grouped format)
- 89 theses, chunk 1 failed 2/3 attempts

### Run 2: New format, first prompt iteration
- 103 theses, 75 not_claims, 0 retries
- 20+ missed claims, model dumped short sentences into "filler" category

### Run 3: New format, clean prompt, word-based chunks (2 chunks)
- 123 theses, 34 not_claims, 0 retries, 115 unique checkable after dedup
- Captured all 20 previously-missed claims
- ~5 remaining edge-case misses
- Cross-chunk dedup bug caused IndexError in synthesize (fix untested)

### Run 4: Sentence-based chunks (4 chunks), topic removed, default presence_penalty
- **160 theses, 15 not_claims, 0 retries on all 4 chunks**
- 0 cross-chunk exact duplicates
- 15 semantic duplicates caught by embedding dedup
- 142 final claims after synthesis
- Cross-chunk index fix VERIFIED — full pipeline completed

**Not_claims (15 total, audit):**
- 9 correct (greetings, closings, fillers: S0, S1, S5, S44, S54, S55, S103, S184, S185)
- 4 defensible borderline (S8 blessing, S28 procedural framing, S37 pledge, S89 fragment)
- 2 genuine misses (S25 "We don't have to be there", S26 "We don't need their oil")
  - S24 captures the gist ("totally independent of Middle East")

**Known issues from Run 4:**
- S3 "It was quite something" classified as claim, got thesis from S4's content (over-decontextualization)
- S45 "We'd still be winning" hypothetical misread as factual assertion about previous admin
- Multi-sentence grouping occasionally causes model to inject content from adjacent sentences
  into vague/contentless sentences rather than marking them not_claim
- **Classification bug (pre-existing):** `matched=0` on all batches — model returns indices
  that don't match 0-indexed input. All claims fall back to defaults (verifiable_fact,
  checkable=True, topic=empty). Topic assignment not working until this is fixed.

## Current Architecture

### Pipeline phases
1. **Extract** — claim/not_claim + grouping + decontextualized thesis + speakers
2. **Classify** — checkable/not_checkable + factual_anchor + topic (per claim)
3. **Dedup** — per-speaker embedding clustering (cosine > 0.85)
4. **Synthesize** — merge multi-member dedup groups into overarching claims

### What extraction does NOT do
- Topic assignment (moved to classification)
- Checkability assessment (classification step)
- Semantic dedup (embedding dedup step)

### Cross-chunk dedup (extraction)
Exact text match only — `_tag_theses_for_storage()` deduplicates by identical
`thesis_statement` string. Catches cases where overlapping context causes same
thesis from adjacent chunks. NOT semantic.

## TODO
- [ ] Fix classification index matching bug (matched=0)
- [ ] Investigate multi-sentence grouping quality (S3/S45 issues)
- [ ] Consider whether contentless sentences adjacent to claims should be forced not_claim
