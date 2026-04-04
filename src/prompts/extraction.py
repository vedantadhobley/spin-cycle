"""Prompts for two-pass transcript claim extraction (Phase 1).

Pass 1 (Grouping + Disposition):
  Group sentences by semantic continuity, then label each group claim/not_claim.
  No thesis writing — the model only decides structure and disposition.

Pass 2 (Context Injection):
  Take each claim group's raw sentences + surrounding context, resolve
  pronouns/references to make them standalone. Editing, not writing.

Key design:
- Sentences are globally numbered [S0], [S1], ..., [Sn]
- Pass 1: every sentence gets a group number; every group gets a disposition
- Pass 2: every claim group gets one decontextualized statement
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
# PASS 2: CONTEXT INJECTION
# ===========================================================================

CONTEXT_INJECT_SYSTEM = """\
You receive raw transcript sentences grouped into claims. Make each \
group's sentences understandable to someone who hasn't read the transcript.

Today's date: {current_date}

## Rules

- Replace pronouns with specific referents (names, countries, organizations)
- Add dates and named entities when the referent is clear from context
- Specify what "it", "they", "this", "that" refers to
- If a group has multiple sentences, produce ONE statement that captures \
the combined assertion
- Do NOT add assertions not present in the original sentences — only \
resolve references
- Write in neutral, factual language — extract the checkable kernel, \
not the speaker's subjective framing
- The result must be understandable without any transcript context\
"""

CONTEXT_INJECT_USER = """\
Decontextualize each claim group below. Use the full transcript excerpt \
for reference resolution only.

## Full Transcript Excerpt (for context)
{full_text}

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

## Claim Groups to Decontextualize
{claim_groups_text}

Return JSON:
{{
  "claims": [
    {{"group": 1, "decontextualized_statement": "Fully standalone claim statement"}},
    {{"group": 2, "decontextualized_statement": "Another standalone claim statement"}}
  ]
}}\
"""
