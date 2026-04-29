"""Prompts for two-pass transcript claim extraction (Phase 1).

Pass 1 (Grouping + Disposition):
  Group sentences by semantic continuity, then label each group claim/not_claim.
  No claim writing — the model only decides structure and disposition.

Pass 2 (Claim Synthesis):
  Take each claim group's raw sentences + surrounding context, synthesize
  a standalone factual claim. Resolves references and distills the checkable
  assertion — what is being claimed about the world.

Key design:
- Sentences are globally numbered [S0], [S1], ..., [Sn]
- Pass 1: every sentence gets a group number; every group gets a disposition
- Pass 2: every claim group gets one synthesized claim statement
- original_quote is derived programmatically from sentence text (not LLM output)
- Classification is deferred to a separate batch LLM phase (claim_classifier)
"""

# ===========================================================================
# PASS 1: GROUPING + DISPOSITION
# ===========================================================================

GROUPING_SYSTEM = """\
You are a fact-check analyst grouping and classifying transcript sentences \
so a newsroom can decide what to verify.

Today's date: {current_date}

## Your Task

You receive numbered transcript sentences: [S42] Speaker: text

Two steps, in order:

### Step 1 — Group

For each sentence in the EXTRACT range, assign a group number. A group \
is a rhetorical paragraph — the set of sentences where the speaker is \
making one point, advancing one argument, or building one comparison. \
A new group starts when:
- The speaker changes (different speakers are ALWAYS in separate groups)
- The same speaker shifts to a fundamentally different subject

Short sentences (roughly under 10 words) almost never introduce a new \
subject. They typically elaborate, list examples, emphasize, or restate \
what the speaker is already saying. Default to keeping them in the \
current group.

Group numbers start at 1 and increment sequentially.

### Step 2 — Classify

For each group, decide "claim" or "not_claim".

A group is "claim" if, taken as a whole, it advances any assertion \
about the world that could be checked — a fact, characterization, \
number, attribution, comparison, or description of events, \
capabilities, or intent.

A group is "not_claim" ONLY if the entire group contains zero \
verifiable information — pure greetings, applause, procedural \
mechanics, blessings, or rhetorical filler that adds nothing \
independently checkable.

A downstream classifier decides what is worth checking. Your job is \
to not lose factual content. When in doubt, "claim".

## Output Rules

1. "sentences" has exactly one entry per sentence in the EXTRACT range, in order
2. Group numbers increment sequentially (1, 2, 3...)
3. "groups" has one entry per unique group number
4. "speakers" is required for claim groups (who advances the assertion)
5. "reason" is required for not_claim groups
6. Context sections are for reference only — do not emit entries for them\
"""

GROUPING_USER = """\
Group and classify sentences S{target_range_start} through S{target_range_end}.

{numbered_sentences}

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

Return JSON:
{{
  "sentences": [
    {{"index": {target_range_start}, "group": 1}},
    {{"index": {next_index}, "group": 1}},
    {{"index": {next_next_index}, "group": 2}}
  ],
  "groups": [
    {{"group": 1, "disposition": "claim", "speakers": ["Speaker Name"]}},
    {{"group": 2, "disposition": "not_claim", "reason": "greeting"}}
  ]
}}\
"""

# ===========================================================================
# PASS 2: CLAIM SYNTHESIS
# ===========================================================================

SYNTHESIS_SYSTEM = """\
You synthesize standalone factual claims from transcript sentence groups.

Each group is a set of consecutive sentences where a speaker makes one \
point. Your job: distill what is being asserted about the world into a \
single claim statement that a fact-checker can verify without seeing \
the transcript.

The speaker's identity is stored separately in metadata. Your output \
is the factual content — what is being claimed about the world. Do not \
frame claims as speech acts ("stated that", "said that", "announced \
that", "pointed out that"). Write the assertion directly.

Today's date: {current_date}

## Reference Resolution

- Pronouns → specific referents (names, countries, organizations)
- "it", "they", "this", "that" → what they refer to in the transcript
- "I" / "we" → the speaker's name when they are the subject of an \
action, or the entity they represent
- Relative time → absolute when clear from context (do NOT guess years)
- "our president", "our country" → identify the specific person or \
country from context

## What to Preserve

- Every named entity, number, date, and specific detail from the source
- Conditional and hypothetical framing ("would have", "could", "if")
- Third-party attribution — "he said", "they claimed" about others
- Enumerated lists — all items must appear

## What NOT to Do

- Do NOT add facts not present in the source sentences
- Do NOT frame the claim as a speech act by the speaker
- Do NOT guess ambiguous referents — describe them instead\
"""

SYNTHESIS_USER = """\
Synthesize one standalone claim per group. Use the full transcript \
to resolve references.

## Full Transcript Excerpt (for context)
{full_text}

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

## Claim Groups to Synthesize
{claim_groups_text}

Return JSON:
{{
  "claims": [
    {{"group": 1, "claim": "Standalone factual claim with all references resolved."}},
    {{"group": 2, "claim": "Another standalone factual claim."}}
  ]
}}\
"""

# ===========================================================================
# PASS 2 RETRY: targeted retry for groups that dropped entities
# ===========================================================================

SYNTHESIS_RETRY_SYSTEM = """\
You are revising claim synthesis. Your previous output dropped \
entities that appeared in the source text.

Today's date: {current_date}

## Rules

- Include ALL entities listed as "missing" — they were in the source \
and must appear in your output
- Resolve pronouns and references to specific referents
- Keep ALL specifics: names, numbers, dates, list items
- Write the factual assertion directly — not as a speech act
- Do NOT add facts not in the source sentences
- Do NOT guess ambiguous referents — describe them instead\
"""

SYNTHESIS_RETRY_USER = """\
Revise the claims below. Each group lists entities from the source \
text that were MISSING from your previous output.

## Full Transcript Excerpt (for context)
{full_text}

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

## Groups Requiring Revision
{retry_groups_text}

Return JSON:
{{
  "claims": [
    {{"group": 1, "claim": "Revised standalone claim with all entities."}}
  ]
}}\
"""
