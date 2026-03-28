"""Prompts for thesis-level transcript extraction.

The extraction LLM receives a full transcript with numbered segments and
identifies 15-30 major ARGUMENTS (theses), each with supporting segment
references.  This replaces per-segment atomic claim extraction.

Key design:
- One LLM pass over the full transcript (no batching)
- Merge repetitions: same argument in segments 5, 23, 41 → one thesis
- Segment numbers must match [N] labels in transcript
- Excerpts must be actual words from the segment, not paraphrased
- Checkable = could independent data confirm or deny?
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

THESIS_EXTRACTION_SYSTEM = """\
You are a fact-check analyst identifying the major ARGUMENTS in a transcript \
so a newsroom can verify them.

Today's date: {current_date}

## Your Task

You receive a full transcript with numbered segments: [0], [1], [2], etc. \
Identify the 15-30 major arguments (theses) that speakers make.

## What Is a Thesis?

A thesis is a COMPLETE ARGUMENT, not an individual sentence. If a speaker \
makes the same point across segments [5], [23], and [41], that is ONE thesis \
with THREE supporting references.

Examples of theses vs. not-theses:
- THESIS: "The United States military spending exceeded $1 trillion per year \
during the first Trump administration" (verifiable aggregate claim)
- NOT A THESIS: "Thank you for being here" (greeting)
- NOT A THESIS: "This is going to be amazing" (subjective prediction)
- THESIS: "Operation X destroyed Country Y's nuclear facilities" (specific \
military claim)
- NOT A THESIS: "We have the best people" (vague rhetoric)

## Step 1 — Scan the Full Transcript

Read the entire transcript. Identify the distinct arguments being made. \
Many speakers repeat and elaborate the same argument across multiple segments \
— these are ONE thesis, not many.

## Step 2 — Write Thesis Statements

For each argument, write a thesis_statement that:
- Is NEUTRAL and DECONTEXTUALIZED (no pronouns, no "we", no "they")
- Replaces ALL pronouns with specific entities
- Could be understood by someone who hasn't read the transcript
- Captures the FULL argument, not just one sentence of it

## Step 3 — Attach Supporting References

For each thesis, list 2-6 supporting_references. Each reference has:
- segment_index: the [N] number from the transcript
- excerpt: the ACTUAL first 15-20 words from that passage (copy directly, \
do not paraphrase)

CRITICAL: segment_index must be an actual [N] label from the transcript. \
Excerpts must be real words from that segment — the system verifies them.

## Step 4 — Assess Checkability

Could independent data (statistics, records, official documents, reporting) \
confirm or deny this argument?

NOT checkable:
- Pure subjective opinions ("this is the greatest")
- Future predictions ("we will achieve")
- Promises or intentions ("I'm going to do")
- Unmeasurable states (resolve, commitment, determination)
- Vague rhetoric without specific claims

Checkable:
- Quantitative claims (amounts, percentages, rankings)
- Historical events (operations, votes, agreements)
- Attribution (who said or did what)
- Policy descriptions (what a law does, what a program costs)
- Comparisons with specific metrics

## Step 5 — Classify Topic

Assign one topic label: economic, military, political, legal, social, \
diplomatic, technological, environmental, health, or other.

## Output Rules

1. [Section: ...] headers in the transcript are editorial context, NOT spoken words
2. Aim for 15-30 theses for a typical transcript
3. Merge repetitions — if the same point appears in 5 segments, ONE thesis
4. Every thesis needs at least 2 supporting references
5. Excerpts must be copied from the transcript, not invented\
"""

# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

THESIS_EXTRACTION_USER = """\
Identify the major arguments in this transcript.

## Transcript
{numbered_transcript}

## Context
{context_note}

## Speaker Descriptions
{speaker_descriptions}

Return JSON:
{{
  "theses": [
    {{
      "thesis_statement": "Neutral, decontextualized argument statement",
      "speakers": ["Speaker Name"],
      "supporting_references": [
        {{"segment_index": 0, "excerpt": "First 15-20 words from segment..."}},
        {{"segment_index": 5, "excerpt": "First 15-20 words from segment..."}}
      ],
      "topic": "military",
      "checkable": true,
      "checkability_rationale": "Military operations are documented by defense agencies and independent media."
    }}
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Legacy re-exports (old batch extraction prompt, still importable)
# ---------------------------------------------------------------------------

from src.prompts._archive_extraction import (  # noqa: F401, E402
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
)
