"""Prompts for batch claim classification.

Classification operates on decontextualized claim text only — no transcript
context needed. Output per claim is tiny (~25 tokens), so 50+ claims per
call is safe.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CLASSIFY_CLAIMS_SYSTEM = """\
You classify factual claims for a newsroom fact-checking pipeline. \
For each claim, follow the steps below.

IMPORTANT: Classify based on the claim's structure, not your knowledge of \
whether the events actually occurred. A claim about a past event is \
verifiable regardless of whether you believe it happened — verifying it \
is the next step in the pipeline, not yours.

## Step 1 — Factual Anchor

Ask: what evidence would a fact-checker look for to investigate this claim? \
This could be statistics, records, events, observable actions, or measurable \
outcomes. If there is genuinely nothing to investigate, write "none".

## Step 2 — Classify

Categories:
- verifiable_fact: A fact-checker could investigate this against evidence
- not_checkable: No investigable claim — procedural statements, pure value \
judgments with no observable indicators, or filler with no factual content

Default to verifiable_fact. Only mark not_checkable if Step 1 produced "none".

## Step 3 — Checkable

checkable=true if Step 1 produced an evidence path. false otherwise.

check_rationale: One sentence naming what evidence exists or why none does.\
"""

# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

CLASSIFY_CLAIMS_USER = """\
Classify each claim below.

## Claims
{claims_list}

Return JSON:
{{
  "classifications": [
    {{
      "index": 0,
      "factual_anchor": "Budget figures for FY2026",
      "classification": "verifiable_fact",
      "checkable": true,
      "check_rationale": "Budget figures can be verified against public records."
    }}
  ]
}}\
"""
