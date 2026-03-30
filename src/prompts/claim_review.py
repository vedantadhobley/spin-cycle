"""Prompts for Phase 2 claim review: classify, dedup, group, assess.

One LLM call per speaker. Input is structured extracted claims (not raw
transcript). The LLM classifies each claim, flags cross-chunk duplicates,
groups related verifiable claims into arguments, and assesses checkability
per group.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CLAIM_REVIEW_SYSTEM = """\
You are a fact-check analyst reviewing extracted claims from a transcript. \
Your job is to classify, deduplicate, group, and assess claims for a \
single speaker.

Today's date: {current_date}

## Your Task

You receive a numbered list of claims already extracted from a transcript. \
Perform ALL 4 steps below.

## Step 1 — Classify Each Claim

For each claim, assign exactly one classification:
- verifiable_fact: A factual assertion that could be checked against \
independent evidence (statistics, records, documents, reporting)
- future_prediction: A prediction or promise about what will happen \
("We will achieve", "This is going to lead to")
- subjective_opinion: A value judgment with no factual anchor \
("This is the greatest", "They don't care about")
- procedural: Parliamentary/meeting procedure, scheduling, introductions \
("I yield my time", "We'll take a short recess")
- vague_rhetoric: A claim too vague to verify — no specific facts, \
numbers, or events ("We have tremendous support", "Everyone knows")

Provide a rationale for each classification.

## Step 2 — Flag Duplicates

Claims extracted from overlapping transcript chunks may appear twice. \
Identify claims that make the SAME factual assertion (even if worded \
slightly differently). Mark duplicates with the index of the FIRST \
occurrence they duplicate. Do NOT flag claims that merely share a topic.

## Step 3 — Group Related Claims

Among the verifiable, non-duplicate claims, identify groups that are part \
of the SAME argument. Grouping criteria:
- Same underlying factual assertion or thesis
- Would be verified together with the same evidence
- Different facets of the same core claim

Standalone claims that don't relate to others = group of 1. \
NEVER group claims about different factual questions just because they \
share a topic.

## Step 4 — Assess Checkability

For each group: could independent evidence (statistics, records, official \
documents, reporting) confirm or deny the core assertion? Mark as not \
checkable ONLY if verification is genuinely impossible.

## Output Format

Return JSON with ALL 3 top-level fields: classifications, groups.\
"""

# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

CLAIM_REVIEW_USER = """\
Review these {claim_count} claims by {speaker_name}.

{claims_list}

Return JSON:
{{
  "classifications": [
    {{
      "claim_index": 0,
      "classification": "verifiable_fact",
      "is_duplicate": false,
      "duplicate_of": null,
      "rationale": "Claims a specific statistic that can be checked."
    }}
  ],
  "groups": [
    {{
      "member_indices": [0, 3],
      "group_rationale": "Both claims address the same military operation budget.",
      "checkable": true,
      "checkability_rationale": "Budget figures are available in public DOD records.",
      "topic": "military"
    }}
  ]
}}\
"""


def format_claims_for_review(theses: list, speaker: str) -> str:
    """Format extracted theses as a numbered list for the review prompt.

    Args:
        theses: List of ExtractedThesis (or dicts with thesis_statement/topic/supporting_references).
        speaker: Speaker name for the header.

    Returns:
        Formatted string like:
        [0] "thesis statement" (topic: military, refs: [1], [5])
        [1] "thesis statement" (topic: economic, refs: [2])
    """
    lines = []
    for i, t in enumerate(theses):
        if hasattr(t, "thesis_statement"):
            statement = t.thesis_statement
            topic = t.topic
            refs = [str(r.segment_index) for r in t.supporting_references]
        else:
            statement = t.get("thesis_statement", t.get("claim_text", ""))
            topic = t.get("topic", "")
            refs = [str(r["segment_index"]) for r in t.get("supporting_references", [])]

        ref_str = ", ".join(f"[{r}]" for r in refs) if refs else "none"
        lines.append(f'[{i}] "{statement}" (topic: {topic}, refs: {ref_str})')

    return "\n".join(lines)
