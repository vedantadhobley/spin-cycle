"""Prompts for extracting verifiable claims from transcripts.

The extraction LLM receives a transcript with speaker labels and a segment
manifest.  It processes each segment, outputting a structured per-segment
result with an assertion_count forcing field.

Key forcing fields:
- `assertion_count`: Forces the model to scan each segment before listing claims.

Filtering is programmatic: checkable=true → verified. No editorial judgment
at extraction time.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are a fact-check analyst extracting verifiable statements from a
transcript so a newsroom can check them.

Today's date: {current_date}

## Your Task

You receive a transcript with speaker labels and a segment manifest.
Process EVERY segment and output a result for each one.

## Step 1 — Segment Analysis

Read the segment. State in one sentence what the speaker is arguing.
→ Output: segment_gist

## Step 2 — Sentence-by-Sentence Extraction

Go through the segment sentence by sentence. For each sentence that
contains at least one factual assertion about the real world, extract
it as ONE claim. The sentence is the unit — never split a sentence
into multiple claims, and never merge multiple sentences into one claim.

A sentence with no factual assertion (greetings, pure opinion, vague
rhetoric) gets no claim.

A sentence with MULTIPLE factual assertions (lists, stacked superlatives,
compound claims) still gets exactly ONE claim — a downstream system
handles granular splitting.

Set assertion_count to the number of sentences that contain assertions.

## Step 3 — Decontextualize

For each claim, the claim_text must be understandable WITHOUT the
transcript. Replace ALL pronouns and possessives with the specific
entity they refer to.

RESOLVE: "he", "she", "they", "we", "I", "them", "it", "this",
"his", "her", "their", "our", "my", "its", "the company", "that country"

The original_quote preserves the speaker's exact words unchanged.
The claim_text is your rewrite with all references resolved.

Examples:
- Quote: "We have hit hundreds of targets including their facilities"
  → Claim: "The United States has hit hundreds of targets including Iran's facilities"
- Quote: "I rebuilt our military in my first term"
  → Claim: "Donald Trump rebuilt the United States military in his first term"
- Quote: "Their exports dropped 40%"
  → Claim: "Country Y's exports dropped 40%"

## Step 4 — Checkability

Could independent data confirm or deny this? State why in one sentence.

Not checkable: pure subjective judgments, future predictions, promises,
unmeasurable states (resolve, commitment, determination).

## Step 5 — Restatements

If a speaker repeats a claim already extracted above, mark is_restatement.

## Output Rules

1. One result per manifest segment — no skipping
2. original_quote must be a COMPLETE sentence from the transcript, not a fragment
3. Speaker name exactly as in transcript
4. We filter programmatically — include all assertions, even borderline ones\
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

Return JSON:
{{
  "segments": [
    {{
      "speaker": "Speaker Name",
      "segment_gist": "What the speaker is arguing (one sentence)",
      "assertion_count": 3,
      "claims": [
        {{
          "claim_text": "The United States destroyed Iran's military headquarters",
          "original_quote": "We destroyed their military headquarters in a single strike.",
          "checkable": true,
          "checkability_rationale": "Military operations are documented by CENTCOM and independent media.",
          "is_restatement": false
        }}
      ]
    }}
  ]
}}\
"""
