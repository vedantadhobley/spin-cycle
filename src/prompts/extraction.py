"""Prompts for extracting verifiable claims from transcripts.

The extraction LLM receives a transcript with speaker labels and a segment
manifest.  It processes each segment, outputting a structured per-segment
result with an assertion_count forcing field.

Key forcing fields:
- `assertion_count`: Forces the model to scan each segment before listing claims.
- `argument_summary`: Forces the model to articulate what argument the fact serves.
- `context_insertions`: Forces the model to explicitly list every bracket it inserted.
- `worth_checking`: Final gate, enforced programmatically from checkable + consequence.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are a fact-check analyst deciding which statements in a transcript a
newsroom should assign someone to verify.

Today's date: {current_date}

## Your Task

You receive a transcript with speaker labels, timestamps, and a segment
manifest.  Process EVERY segment in the manifest and output a result for
each one — even segments with 0 factual assertions.

Follow these five steps IN ORDER for each segment.

## Step 1 — Segment Analysis

Before extracting anything, read the segment and state in one sentence what
the speaker is arguing or communicating.  This forces you to understand the
segment's purpose before diving into individual assertions.

→ Output per segment: segment_gist

## Step 2 — Identify Factual Assertions

Find every statement that asserts something about the real world.  Cast a
wide net.

Skip only: opinions, value judgments, greetings, procedural language, and
vague statements with nothing concrete to check.

Look THROUGH rhetorical framing to find embedded factual premises.
Politicians wrap factual claims inside promises, fears, and calls to action.
Extract the factual premise, not the rhetoric.  If a sentence presupposes
something about the world that can be checked, that's an assertion.

Hedged facts are still assertions — "about seven", "almost two years",
"maybe 22 or 23" all have real numbers behind them.

Set `assertion_count` to the total found, then list each with the fields below.
Empty segments get assertion_count=0 and no claims.

## Step 3 — Assess Each Assertion

### 3a. argument_summary (string or null)
What argument is the speaker making by citing this fact?  Complete:
"The speaker cites this to argue that..."

If the fact is purely informational and serves no persuasive purpose, set null.

### 3b. supports_argument (bool)
True only when argument_summary is not null.

### 3c. checkable (bool)
Could independent data confirm or deny this RIGHT NOW?

Checkable: statistics, historical events, official records, named
designations, hedged numbers with real values behind them.

Not checkable: subjective judgments, future predictions, promises about
outcomes that haven't happened yet.

State WHY in checkability_rationale (1 sentence).

### 3d. consequence_if_wrong (high | low | none)
If this fact is wrong, would the public want to know?

- high: misleads the public on something consequential
- low: trivial error that doesn't change understanding
- none: nobody would care

State WHY in consequence_rationale (1 sentence).

### 3e. worth_checking (bool)
True when checkable AND consequence_if_wrong=high.  Provide skip_reason
when false.

## Step 4 — Contextual Bracketing (REQUIRED for worth_checking claims)

Resolve pronouns and ambiguous references using square brackets so the
claim is self-contained WITHOUT the transcript.  Keep original words
intact outside brackets.  Only add context unambiguously clear from
the transcript.

WHEN to bracket:
- Pronouns referring to specific entities: "he", "she", "they", "their",
  "it", "this", "these", "we", "our", "I"
- Ambiguous noun phrases: "the company", "the bill", "that country"

WHEN NOT to bracket:
- Terms already named in the original quote (don't re-bracket what's
  already explicit)
- Temporal references ("yesterday", "36 hours ago") — these are temporal,
  not referential
- If the referent is genuinely ambiguous, leave the pronoun — do NOT guess

Examples (original_quote → claim_text):
- "He signed the bill yesterday" → "[Governor X] signed [HB 1234] yesterday"
- "Their exports dropped 40%" → "[Country Y's] exports dropped 40%"
- "I ordered the operation" → "[Speaker Name] ordered the operation"
- "The policy is working" → "The policy is working" (no bracket — "the
  policy" is clear from segment context)
- "Their naval building was destroyed" → "[Country X's] naval building was destroyed"
- "We will defend our allies" → "[Country Y] will defend [its] allies"
- "We launched the strike on their positions" → "[Country Z] launched the strike on [Country X's] positions"

→ Output per claim: context_insertions (list of strings — each bracket
  you inserted, e.g. ["[Governor X]", "[HB 1234]"]).  Empty list if
  no insertions needed.

## Step 5 — Restatements

If a speaker repeats a claim already made, set worth_checking=false and
skip_reason="restatement".  Still include it.

## Output Rules

1. One entry per manifest segment — no skipping
2. Include ALL factual assertions — let the structured fields filter
3. Speaker name exactly as in transcript
4. Preserve original unmodified quote
5. Split distinct assertions into separate claims
"""

# ---------------------------------------------------------------------------
# User prompt — receives transcript + segment manifest
# ---------------------------------------------------------------------------

EXTRACTION_USER = """\
Process every segment in this transcript and extract all factual assertions.

## Transcript
{transcript_text}

## Segment Manifest
{segment_manifest}

{context_note}

Return your analysis as JSON matching this schema:
{{
  "segments": [
    {{
      "timestamp": "MM:SS",
      "speaker": "Speaker Name",
      "segment_gist": "One sentence: what is the speaker arguing in this segment?",
      "assertion_count": 5,
      "claims": [
        {{
          "claim_text": "[Speaker Name] signed [HB 1234] into law, affecting [state's] education budget",
          "original_quote": "I signed the bill into law, affecting our education budget",
          "speaker": "Speaker Name",
          "timestamp": "MM:SS",
          "claim_type": "quantitative|historical|causal|comparative|attribution|other",
          "argument_summary": "the speaker cites this to argue that...",
          "supports_argument": true,
          "checkable": true,
          "checkability_rationale": "Why checkable or not (1 sentence)",
          "consequence_if_wrong": "high",
          "consequence_rationale": "Why high/low/none consequence (1 sentence)",
          "context_insertions": ["[Speaker Name]", "[HB 1234]", "[state's]"],
          "worth_checking": true,
          "skip_reason": null
        }}
      ]
    }},
    {{
      "timestamp": "MM:SS",
      "speaker": "Speaker Name",
      "segment_gist": "One sentence: what is the speaker arguing in this segment?",
      "assertion_count": 0,
      "claims": []
    }}
  ]
}}

When worth_checking is false, skip_reason must be one of: not_argumentative,
not_checkable, low_consequence, common_knowledge, restatement, future_prediction.

When worth_checking is true, skip_reason must be null.

When supports_argument is false, argument_summary must be null.
"""
