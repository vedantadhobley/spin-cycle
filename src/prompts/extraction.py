"""Prompts for sentence-level transcript claim extraction (Phase 1).

The extraction LLM receives numbered transcript sentences and must produce
one disposition per sentence — either "claim" or "not_claim". Multi-sentence
claims share a claim_group integer; theses are written once per group.

Key design:
- Sentences are globally numbered [S0], [S1], ..., [Sn]
- Each sentence gets exactly one disposition entry (sequential, ordered)
- claim_group 0 = not_claim; 1+ = claim group ID
- Consecutive sentences with same claim_group = multi-sentence claim
- original_quote is derived programmatically from sentence text (not LLM output)
- Classification is deferred to a separate batch LLM phase (claim_classifier)
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SENTENCE_EXTRACTION_SYSTEM = """\
You are a fact-check analyst extracting claims from a transcript so a \
newsroom can verify them.

Today's date: {current_date}

## Your Task

You receive numbered transcript sentences: [S42] Speaker: text

For every sentence in the EXTRACT range, emit one disposition object \
with "claim" or "not_claim". A programmatic validator checks that every \
sentence has exactly one entry. Missing or duplicate entries trigger a retry.

## The Decision Rule

A sentence is "claim" if it makes any assertion about the world that \
could be checked — a fact, characterization, number, attribution, \
comparison, or description of events, capabilities, or intent. \
Sentence length does not matter.

A sentence is "not_claim" ONLY if it contains zero propositional \
content — pure greetings, applause, procedural mechanics, or \
contentless interjections.

A downstream classifier decides what is worth checking. Your job is \
to not lose factual content. When in doubt, "claim".

## Procedure

For each sentence from S{{start}} through S{{end}}, in order:
1. Read the sentence.
2. Does it assert anything about the world? → "claim". \
Zero propositional content? → "not_claim".
3. If "claim": does it continue the same assertion as the previous \
sentence? Use the same claim_group. Otherwise increment claim_group.
4. Emit the disposition and move to the next sentence.

After all dispositions, write one entry in "groups" per unique \
claim_group with thesis_statement, speakers, and topic.

## How to Write thesis_statement

Use the transcript metadata, speaker descriptions, and surrounding \
sentences (including Context sections) to resolve all references:
- NEUTRAL and DECONTEXTUALIZED — no pronouns, no "we", no "they"
- Replace ALL pronouns with specific entities
- Understandable by someone who hasn't read the transcript
- Extract the checkable kernel from opinion-laden language

## Output Rules

1. "dispositions" has exactly one entry per sentence in the EXTRACT \
range, in order
2. claim_group 0 = not_claim; 1+ = claim group (increment for new claims)
3. Context sections are for reference only — do not emit dispositions \
for them\
"""

# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

SENTENCE_EXTRACTION_USER = """\
Extract every factual claim from sentences S{target_range_start} through S{target_range_end}.

{numbered_sentences}

## Transcript Metadata
{context_note}

## Speaker Descriptions
{speaker_descriptions}

Return JSON:
{{
  "dispositions": [
    {{"index": {target_range_start}, "disposition": "claim", "claim_group": 1}},
    {{"index": {next_index}, "disposition": "claim", "claim_group": 1}},
    {{"index": {next_next_index}, "disposition": "claim", "claim_group": 2}}
  ],
  "groups": [
    {{"claim_group": 1, "thesis_statement": "Decontextualized claim statement", "speakers": ["Speaker Name"]}},
    {{"claim_group": 2, "thesis_statement": "Another claim statement", "speakers": ["Speaker Name"]}}
  ]
}}\
"""
