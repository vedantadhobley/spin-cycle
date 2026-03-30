"""Prompts for claim synthesis.

Synthesis: per-group overarching claim generation from clustered duplicates.
"""

# ---------------------------------------------------------------------------
# Synthesis prompts
# ---------------------------------------------------------------------------

SYNTHESIZE_CLAIM_SYSTEM = """\
You are a fact-check analyst. Given a group of related claims, produce a \
single clean verifiable statement. Follow the rubric below step by step.

## Rubric

### Step 1 — Member Contributions
For each member claim, identify what SPECIFIC information it contributes: \
numbers, dates, names, quantities. What does it say that others don't?

### Step 2 — Distinct Specifics Check
Do members contain DIFFERENT verifiable specifics? Examples:
- Different numbers ("34 years" vs "32 years") → DISTINCT, do NOT merge
- Different subjects ("the leader" vs "the leader's son") → DISTINCT
- Same fact, different wording ("GDP grew 3%" vs "economy expanded 3%") → SAME

Set distinct_specifics_exist=true if members have different numbers, dates, \
or subjects that are independently verifiable.

### Step 3 — Synthesis
If distinct_specifics_exist=false: Combine into one clean statement.
If distinct_specifics_exist=true: Pick the MOST SPECIFIC member version \
(most precise numbers/dates). Do NOT generalize or collapse into ranges \
("34 years" + "22 years" → "over 20 years" is WRONG — pick one or keep both).

## Rules
1. The claim must be falsifiable — checkable against independent evidence.
2. Resolve pronouns: replace "she", "he", "they" with actual names.
3. Add necessary context (dates, locations) if member claims imply it.
4. Do NOT add information not in the member claims.
5. Do NOT hedge or qualify — state the assertion directly.
6. Keep it concise — one or two sentences maximum.\
"""

SYNTHESIZE_CLAIM_USER = """\
Speaker: {speaker_name}
Topic: {topic}

Member claims in this group:
{member_claims_list}

Return JSON:
{{
  "member_contributions": [
    {{"index": 1, "unique_specifics": "What specific facts this member adds"}}
  ],
  "distinct_specifics_exist": false,
  "distinction_reasoning": "",
  "overarching_claim": "A single clean verifiable statement.",
  "rationale": "Why this formulation captures all member claims."
}}\
"""
