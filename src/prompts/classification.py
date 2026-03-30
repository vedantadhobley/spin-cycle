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

## Step 1 — Factual Anchor

Before classifying, identify the specific fact, statistic, event, or \
record that could be checked. If none exists, write "none".

Examples:
- "GDP grew 3.2% in Q4 2025" → "3.2% GDP growth, Q4 2025"
- "We need to do better" → "none"
- "The bill passed 52-48" → "52-48 vote count"

## Step 2 — Classify

Categories:
- verifiable_fact: Can be checked against independent evidence \
(statistics, records, documents, reporting)
- future_prediction: Promise or prediction about what will happen
- subjective_opinion: Value judgment with no factual anchor
- procedural: Meeting procedure, scheduling, introductions
- vague_rhetoric: Too vague to verify — no specific facts or numbers

## Step 3 — Checkable

Set checkable=true if independent evidence could confirm or deny this claim. \
Most verifiable_fact claims are checkable. Future predictions, opinions, \
and vague rhetoric are generally not.

check_rationale: One sentence explaining why the claim is or is not checkable.\
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
