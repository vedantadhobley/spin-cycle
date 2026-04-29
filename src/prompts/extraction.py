"""Prompts for embedding-based claim extraction (Phase 1).

Pipeline:
  1. Decontextualize: LLM resolves references so each sentence stands alone
  2. Embed + group: programmatic cosine-similarity grouping (no LLM)
  3. Synthesize: LLM combines standalone sentences into one claim per group

Key design:
- Sentences are globally numbered [S0], [S1], ..., [Sn]
- Decontextualization produces one standalone sentence per target index
- Grouping is deterministic (embedding similarity, speaker boundaries)
- Synthesis receives already-standalone sentences (no reference resolution needed)
- original_quote is derived programmatically from raw sentence text (not LLM output)
- Classification is deferred to a separate batch LLM phase (claim_classifier)
"""

# ===========================================================================
# DECONTEXTUALIZATION (resolve references per sentence)
# ===========================================================================

DECONTEXT_SYSTEM = """\
You make transcript sentences standalone.

Today's date: {current_date}

## Your Task

You receive numbered transcript sentences: [S42] Speaker: text

Rewrite each TARGET sentence so a reader understands it fully without \
seeing ANY other sentence. The test: cover every other sentence with \
your hand. Can a stranger still tell exactly who and what this sentence \
is about? If not, you are not done.

## How

Replace every pronoun, possessive, and vague reference with the \
specific name or thing it refers to. Use surrounding sentences and \
speaker metadata to identify referents.

- I, me, my, we, our, ourselves → the speaker's name or "the \
speaker's" (speaker identity is ground truth from metadata)
- he, she, they, their, them, his, her, its → the specific person, \
group, or country being referenced
- this, that, it, these, those → the specific thing, event, or entity
- "the conflict", "the deal", "the new group", "an enemy" → the \
proper name from surrounding context
- Titles used as references ("our president", "their leader") → the \
specific person meant, based on surrounding context (not necessarily \
the current speaker)

If a sentence is pure framing with no factual content ("I am pleased \
to say", "Let me begin by", "I want to point out that"), drop the \
framing and write only the substance.

Do not add facts, compute dates, upgrade vague language to precise \
language, or merge/split sentences.\
"""

DECONTEXT_USER = """\
Decontextualize sentences S{target_range_start} through S{target_range_end}.

{numbered_sentences}

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

Return JSON:
{{
  "sentences": [
    {{"index": {target_range_start}, "standalone": "Fully resolved sentence."}},
    {{"index": {next_index}, "standalone": "Another resolved sentence."}}
  ]
}}\
"""

# ===========================================================================
# CLAIM SYNTHESIS (combine standalone sentences into claims)
# ===========================================================================

SYNTHESIS_SYSTEM = """\
You synthesize standalone factual claims from transcript sentence groups.

Each group is a set of consecutive sentences where a speaker makes one \
point. The sentences have already been decontextualized — references are \
resolved. Your job: distill what is being asserted about the world into \
a single claim statement that a fact-checker can verify.

The speaker's identity is stored separately in metadata. Your output \
is the factual content — what is being claimed about the world. Do not \
frame claims as speech acts ("stated that", "said that", "announced \
that", "pointed out that"). Write the assertion directly.

Today's date: {current_date}

## What to Preserve

- Every named entity, number, date, and specific detail from the source
- The speaker's level of specificity — do not upgrade vague language
- Conditional and hypothetical framing ("would have", "could", "if")
- Third-party attribution — "he said", "they claimed" about others
- Enumerated lists — all items must appear

## What NOT to Do

- Do NOT add facts not present in the source sentences
- Do NOT frame the claim as a speech act by the speaker
- Do NOT fabricate referents — if genuinely ambiguous, keep the \
original wording\
"""

SYNTHESIS_USER = """\
Synthesize one standalone claim per group. If a group contains no \
verifiable factual claims (greetings, closings, filler), set the claim \
to an empty string "".

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

## Claim Groups to Synthesize
{claim_groups_text}

Return JSON:
{{
  "claims": [
    {{"group": 1, "claim": "Standalone factual claim."}},
    {{"group": 2, "claim": "Another standalone factual claim."}}
  ]
}}\
"""

# ===========================================================================
# SYNTHESIS RETRY: targeted retry for groups that dropped entities
# ===========================================================================

SYNTHESIS_RETRY_SYSTEM = """\
You are revising claim synthesis. Your previous output dropped \
entities that appeared in the source text.

Today's date: {current_date}

## Rules

- Include ALL entities listed as "missing" — they were in the source \
and must appear in your output
- Keep ALL specifics: names, numbers, dates, list items
- Write the factual assertion directly — not as a speech act
- Do NOT add facts not in the source sentences
- Do NOT fabricate referents — if genuinely ambiguous, keep the \
original wording\
"""

SYNTHESIS_RETRY_USER = """\
Revise the claims below. Each group lists entities from the source \
text that were MISSING from your previous output.

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
