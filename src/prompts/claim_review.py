"""Prompts for Phase 2 claim review and synthesis.

Review: sequential batches of ~10 claims with accumulating group context.
Synthesis: per-group overarching claim generation.
"""

# ---------------------------------------------------------------------------
# Review batch prompts
# ---------------------------------------------------------------------------

REVIEW_BATCH_SYSTEM = """\
You are a fact-check analyst reviewing extracted claims from a transcript. \
Your job is to classify each claim and assign it to a group.

Today's date: {current_date}

## Your Task

You receive a batch of claims (numbered 0 to N-1) plus any groups already \
created from earlier batches. For each claim:

### Step 1 — Classify

Assign exactly one classification:
- verifiable_fact: A factual assertion checkable against independent evidence \
(statistics, records, documents, reporting)
- future_prediction: A prediction or promise about what will happen
- subjective_opinion: A value judgment with no factual anchor
- procedural: Parliamentary/meeting procedure, scheduling, introductions
- vague_rhetoric: Too vague to verify — no specific facts, numbers, or events

### Step 2 — Decide Action

- new_group: This claim starts a new group (verifiable_fact only). \
You must also provide the group definition in new_groups.
- add_to_group: This claim belongs to an existing group — same factual \
assertion, different wording. Provide the group_id.
- duplicate: Exact duplicate of an already-grouped claim. Provide the \
group_id it duplicates. The claim will be dropped.
- drop: Not verifiable, not worth checking.

## CRITICAL: Grouping Criteria

Group = SAME factual assertion, possibly worded differently from overlapping \
extraction chunks. Ask yourself: "Would these claims be verified with the \
exact same evidence and produce the exact same verdict?"

Do NOT group claims that merely share a topic or mention the same person \
but make DIFFERENT factual assertions.

Example — SAME group:
- "Omar married her brother to commit immigration fraud"
- "Omar's marriage to Elmi was a sham for immigration purposes"
(Same assertion, same evidence needed)

Example — SEPARATE groups:
- "Omar married her brother"
- "Omar was here illegally"
(Different assertions, different evidence needed)

When in doubt, keep claims as separate standalone groups.

## Output Format

Return JSON with two fields: dispositions, new_groups.\
"""

REVIEW_BATCH_USER = """\
Review these {claim_count} claims by {speaker_name}.

{existing_groups_section}\
{claims_list}

Return JSON:
{{
  "dispositions": [
    {{
      "claim_index": 0,
      "classification": "verifiable_fact",
      "action": "new_group",
      "group_id": "G7",
      "rationale": "Specific statistic that can be checked against records."
    }},
    {{
      "claim_index": 1,
      "classification": "subjective_opinion",
      "action": "drop",
      "group_id": null,
      "rationale": "Value judgment with no factual anchor."
    }}
  ],
  "new_groups": [
    {{
      "group_id": "G7",
      "topic": "military",
      "checkable": true,
      "checkability_rationale": "Budget figures available in public DOD records."
    }}
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Synthesis prompts
# ---------------------------------------------------------------------------

SYNTHESIZE_CLAIM_SYSTEM = """\
You are a fact-check analyst. Given a group of claims that all make the same \
factual assertion (possibly worded differently), produce a single clean \
verifiable statement that captures what they all assert.

## Rules

1. The overarching claim must be falsifiable — it must be possible to check \
it against independent evidence.
2. Resolve pronouns: replace "she", "he", "they" with the actual names.
3. Add necessary context (dates, locations) if the member claims imply it.
4. Do NOT add information not present in the member claims.
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
  "overarching_claim": "A single clean verifiable statement.",
  "rationale": "Why this formulation captures all member claims."
}}\
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

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


def format_existing_groups(groups: dict) -> str:
    """Format accumulated groups as context for the next review batch.

    Args:
        groups: dict of group_id → group state dict with keys:
            topic, member_count, representative_statement

    Returns:
        Formatted section, or empty string if no groups yet.
    """
    if not groups:
        return ""

    lines = ["Existing groups from earlier batches:"]
    for gid, g in groups.items():
        count = g.get("member_count", len(g.get("member_indices", [])))
        rep = g.get("representative_statement", "")
        topic = g.get("topic", "")
        lines.append(f'[{gid}] Topic: {topic} | Members: {count} | "{rep}"')
    lines.append("")
    lines.append("You may add_to_group or duplicate against these existing groups.")
    lines.append("")

    return "\n".join(lines) + "\n"
