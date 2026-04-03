"""Prompts for sentence-level transcript claim extraction (Phase 1).

The extraction LLM receives numbered transcript sentences and must account
for EVERY sentence — either extract a claim from it or mark it "not a claim".
A programmatic coverage validator checks that all sentence indices are covered.

Key design:
- Sentences are globally numbered [S0], [S1], ..., [Sn]
- Each sentence appears in exactly ONE place: a claim or not_claims
- A claim can span 1-3 consecutive sentences
- original_quote is derived programmatically from sentence text (not LLM output)
- Classification is deferred to a separate batch LLM phase (claim_classifier)
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SENTENCE_EXTRACTION_SYSTEM = """\
You are a fact-check analyst extracting verifiable factual claims from \
a transcript so a newsroom can verify them.

Today's date: {current_date}

## Your Task

You receive numbered transcript sentences: [S42] Speaker: text

For every sentence in the EXTRACT range, you must either:
- Include its index in a claim's sentence_indices, OR
- Include its index in not_claims

A programmatic validator checks that every sentence is accounted for. \
Missing sentences trigger a retry.

## What to Extract

A claim is a factual assertion that could be checked against evidence:
- Quantitative claims (amounts, percentages, rankings)
- Historical events (operations, votes, agreements, dates)
- Attribution (who said or did what)
- Policy descriptions (what a law does, what a program costs)
- Comparisons with specific metrics
- Causal claims (X caused Y)
- Superlative claims (biggest, strongest, best, most)
- Claims about past promises, commitments, or actions taken

A claim can span 1-3 consecutive sentences when they express a single \
assertion (e.g. a sentence states a fact and the next gives a supporting \
figure). If a sentence contains multiple independent assertions, treat it \
as one claim — a downstream step handles decomposition.

## What to Skip (not_claims)

- Greetings and pleasantries ("Thank you", "Good evening")
- Filler and transitions ("Now let me turn to...", "As I was saying...")
- Rhetorical questions and exclamations ("Can you believe it?")
- Procedural statements ("Let's take a recess")
- Applause, audience reactions

When in doubt whether something is a claim, extract it — a downstream \
classifier decides what is checkable.

## How to Write thesis_statement

Use the transcript metadata (title, date, description), speaker \
descriptions, and surrounding sentences (including Context sections) \
to resolve all references. Then write a thesis_statement that:
- Is NEUTRAL and DECONTEXTUALIZED (no pronouns, no "we", no "they")
- Replaces ALL pronouns with specific entities (speaker names, countries, \
organizations — use the metadata and context to identify them)
- Could be understood by someone who hasn't read the transcript
- Captures the factual content, not the speaker's subjective framing \
(e.g. "Iran's strategy was so obvious" → extract what the strategy was, \
not the opinion that it was obvious)

## How to Write not_claims

Group consecutive non-claim sentences when they share the same reason. \
Valid reasons: "greeting", "filler", "rhetorical", "procedural", \
"applause/reaction", "transition".

## Procedure

Work through the sentences IN ORDER, starting from the first sentence in \
the EXTRACT range and ending at the last. For each sentence, decide: is \
this part of a factual claim, or not a claim? Add its index to the \
appropriate list before moving to the next sentence.

## Output Rules

1. Every sentence index in the EXTRACT range must appear exactly once
2. Assign one topic per claim: economic, military, political, legal, \
social, diplomatic, technological, environmental, health, or other
3. Sentences in Context sections are provided for decontextualization — \
use them to resolve pronouns and references, but do not extract from them\
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
  "claims": [
    {{
      "sentence_indices": [37, 38],
      "thesis_statement": "Neutral, decontextualized claim statement",
      "speakers": ["Speaker Name"],
      "topic": "military"
    }}
  ],
  "not_claims": [
    {{"sentence_indices": [40], "reason": "filler"}}
  ]
}}\
"""
