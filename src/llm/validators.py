"""Semantic validators for LLM outputs.

These validators check domain-specific constraints BEYOND schema validation.
Schema validation ensures the JSON has the right shape.
Semantic validation ensures the content makes sense.

Examples:
- Decompose: At least one fact must exist
- Judge: Verdict must be one of the valid values
- Judge: Confidence and verdict should be consistent
- Judge: Minimum 3 [N] citations in reasoning
- Synthesize: Minimum 5 [N] citations in reasoning
"""

import re
from typing import Callable, Tuple

from src.schemas.llm_outputs import (
    NormalizeOutput,
    DecomposeOutput,
    JudgeOutput,
    SynthesizeOutput,
    ThesisExtractionOutput,
    ReviewBatchOutput,
    SynthesizedClaim,
)
from src.utils.logging import log, get_logger

MODULE = "llm.validators"
logger = get_logger()

# Type alias for validator functions
ValidatorFunc = Callable[[any], Tuple[bool, str]]


def validate_normalize(output: NormalizeOutput) -> tuple[bool, str]:
    """Validate normalize output semantically.

    Checks:
    1. Normalized claim is not empty
    2. Normalized claim is not suspiciously long (prevents elaboration)

    Args:
        output: The normalize output to validate

    Returns:
        (is_valid, error_message) tuple
    """
    if not output.normalized_claim or not output.normalized_claim.strip():
        return False, "Normalized claim is empty"

    # Prevent the LLM from elaborating instead of normalizing.
    # 5000 chars is generous — even complex claims shouldn't triple in length.
    if len(output.normalized_claim) > 5000:
        return False, "Normalized claim is suspiciously long (>5000 chars)"

    return True, ""


def validate_decompose(output: DecomposeOutput) -> tuple[bool, str]:
    """Validate decompose output semantically.

    Checks:
    1. Has at least one fact (not empty)
    2. Facts are non-trivial (not just whitespace)
    3. No near-duplicate facts (independence check)
    4. Facts meet minimum length (decontextualization quality signal)

    Args:
        output: The decompose output to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Not empty
    if not output.facts:
        return False, "Decomposition produced no facts"

    # Check 2: Facts are non-trivial
    # Facts are AtomicFact objects with .text attribute
    fact_texts = [f.text for f in output.facts if f and f.text and f.text.strip()]
    if not fact_texts:
        return False, "All facts are empty or whitespace"

    # Check 3: Near-duplicate facts (independence)
    # Only flag when the shorter string covers >80% of the longer one.
    # This catches true duplicates ("X happened" vs "X happened in 2024")
    # but allows legitimate splits that share a subject prefix
    # ("AIPAC's FARA exemption" vs "AIPAC's FARA exemption differs from...").
    normalized = [f.strip().lower() for f in fact_texts]
    for i, a in enumerate(normalized):
        for j, b in enumerate(normalized):
            if i < j:
                shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
                if shorter in longer and len(shorter) / len(longer) > 0.8:
                    return False, f"Facts {i+1} and {j+1} appear redundant: '{fact_texts[i]}' vs '{fact_texts[j]}'"

    # Check 4: Minimum fact length (decontextualization quality signal)
    for i, f in enumerate(fact_texts):
        if len(f.strip()) < 15:
            return False, f"Fact {i+1} is too short to be self-contained: '{f}'"

    # Check 5: Rubric field — claim_analysis present must be substantive
    if output.claim_analysis and len(output.claim_analysis.strip()) < 20:
        return False, "claim_analysis is present but too short (<20 chars)"

    # Check 6: Non-simple structure needs justification
    if output.structure != "simple" and output.structure_justification:
        if len(output.structure_justification.strip()) < 10:
            return False, "structure_justification is too short for non-simple structure"

    # Check 7: Thesis and key_test must not be identical
    if (output.thesis and output.key_test
            and output.thesis.strip().lower() == output.key_test.strip().lower()):
        return False, "thesis and key_test are identical — they must be different"

    # Check 8: Named interested parties need reasoning
    ip = output.interested_parties
    has_named = bool(ip.direct or ip.institutional or ip.affiliated_media)
    if has_named and ip.reasoning and len(ip.reasoning.strip()) < 15:
        return False, "interested_parties.reasoning is too short for named parties"

    return True, ""


def _count_citations(text: str) -> int:
    """Count unique [N] citation indices in text."""
    return len(set(int(m) for m in re.findall(r'\[(\d+)\]', text)))


# Minimum citation counts — enforced via validator retry
MIN_JUDGE_CITATIONS = 3
MIN_SYNTHESIZE_CITATIONS = 5


def validate_judge(output: JudgeOutput) -> tuple[bool, str]:
    """Validate judge output semantically.

    Checks rubric completeness + confidence/verdict consistency + citation density.
    """
    # Check 1: Rubric fields are populated
    if not output.claim_interpretation or len(output.claim_interpretation.strip()) < 5:
        return False, "claim_interpretation is empty or too short"

    if not output.key_evidence:
        return False, "key_evidence is empty — must assess at least one evidence item"

    if not output.direction_reasoning or len(output.direction_reasoning.strip()) < 10:
        return False, "direction_reasoning is empty or too short"

    if not output.precision_assessment or len(output.precision_assessment.strip()) < 10:
        return False, "precision_assessment is empty or too short"

    # Check 2: Reasoning exists
    if not output.reasoning or len(output.reasoning.strip()) < 10:
        return False, "Reasoning is empty or too short"

    # Check 3: Minimum citation density in reasoning
    # Every subclaim judgment must cite at least 3 sources (or all evidence
    # items if fewer than 3). Unverifiable verdicts are exempt — thin evidence
    # is the problem, not lazy citing.
    if output.verdict != "unverifiable":
        min_required = min(MIN_JUDGE_CITATIONS, len(output.key_evidence))
        citation_count = _count_citations(output.reasoning)
        if citation_count < min_required:
            return False, (
                f"Reasoning cites only {citation_count} sources (minimum {min_required}). "
                f"Every factual assertion must cite at least one source using [N] notation."
            )

    # Check 4: Confidence/verdict consistency (warnings, not hard failures)
    if output.verdict in ("true", "false") and output.confidence < 0.3:
        log.warning(logger, MODULE, "low_confidence_strong_verdict",
                   f"Strong verdict '{output.verdict}' with low confidence {output.confidence}",
                   verdict=output.verdict, confidence=output.confidence)

    if output.verdict == "unverifiable" and output.confidence > 0.8:
        log.warning(logger, MODULE, "high_confidence_unverifiable",
                   f"Unverifiable verdict with high confidence {output.confidence}",
                   verdict=output.verdict, confidence=output.confidence)

    return True, ""


def validate_synthesize(output: SynthesizeOutput) -> tuple[bool, str]:
    """Validate synthesize output semantically.

    Checks rubric completeness + confidence/verdict consistency + citation density.
    """
    # Check 1: Rubric fields are populated
    if not output.thesis_restatement or len(output.thesis_restatement.strip()) < 5:
        return False, "thesis_restatement is empty or too short"

    if not output.subclaim_weights:
        return False, "subclaim_weights is empty — must classify at least one subclaim"

    # Check 2: Reasoning exists
    if not output.reasoning or len(output.reasoning.strip()) < 10:
        return False, "Reasoning is empty or too short"

    # Check 3: Minimum citation density in reasoning
    # Final synthesis must cite at least 5 sources from the evidence digest.
    # Unverifiable verdicts are exempt.
    if output.verdict != "unverifiable":
        citation_count = _count_citations(output.reasoning)
        if citation_count < MIN_SYNTHESIZE_CITATIONS:
            return False, (
                f"Reasoning cites only {citation_count} sources (minimum {MIN_SYNTHESIZE_CITATIONS}). "
                f"Cite evidence using [N] notation from the evidence digest."
            )

    # Check 4: Confidence/verdict consistency
    if output.verdict in ("true", "false") and output.confidence < 0.3:
        log.warning(logger, MODULE, "low_confidence_strong_verdict",
                   f"Strong verdict '{output.verdict}' with low confidence {output.confidence}")

    return True, ""


def validate_extraction(output) -> tuple[bool, str]:
    """Validate extraction output semantically.

    Drops malformed claims in-place rather than failing the entire batch.
    A 10-minute LLM call shouldn't be retried because one claim has an
    empty quote.
    """
    if not output.segments:
        return False, "Extraction produced no segments"

    dropped = 0
    for seg in output.segments:
        good_claims = []
        for claim in seg.claims:
            if not claim.claim_text or len(claim.claim_text.strip()) < 10:
                dropped += 1
                log.warning(logger, MODULE, "extraction_claim_dropped",
                            f"Dropped claim with short/empty claim_text",
                            speaker=seg.speaker, gist=seg.segment_gist)
                continue
            if not claim.original_quote or len(claim.original_quote.strip()) < 5:
                dropped += 1
                log.warning(logger, MODULE, "extraction_claim_dropped",
                            f"Dropped claim with short/empty original_quote",
                            speaker=seg.speaker, claim_text=claim.claim_text[:80])
                continue
            good_claims.append(claim)
        seg.claims = good_claims

    if dropped:
        log.info(logger, MODULE, "extraction_claims_filtered",
                 f"Filtered {dropped} malformed claims from extraction output",
                 dropped=dropped)

    return True, ""


def validate_thesis_extraction(output: ThesisExtractionOutput) -> tuple[bool, str]:
    """Validate thesis extraction output semantically.

    Drops malformed theses in-place (same pattern as validate_extraction).
    Never rejects the entire output — the LLM call is expensive.
    """
    if not output.theses:
        return False, "Thesis extraction produced no theses"

    dropped = 0
    good_theses = []
    for thesis in output.theses:
        # Must have a substantive thesis statement
        if not thesis.thesis_statement or len(thesis.thesis_statement.strip()) < 15:
            dropped += 1
            log.warning(logger, MODULE, "thesis_dropped",
                        "Dropped thesis with short/empty statement",
                        statement=thesis.thesis_statement[:60] if thesis.thesis_statement else "")
            continue
        # Must have a non-empty original_quote
        if not thesis.original_quote or len(thesis.original_quote.strip()) < 10:
            dropped += 1
            log.warning(logger, MODULE, "thesis_dropped",
                        "Dropped thesis with short/empty original_quote",
                        statement=thesis.thesis_statement[:60])
            continue
        # Must have at least one speaker
        if not thesis.speakers:
            dropped += 1
            log.warning(logger, MODULE, "thesis_dropped",
                        "Dropped thesis with no speakers",
                        statement=thesis.thesis_statement[:60])
            continue
        good_theses.append(thesis)

    output.theses = good_theses

    if dropped:
        log.info(logger, MODULE, "thesis_extraction_filtered",
                 f"Filtered {dropped} malformed theses",
                 dropped=dropped)

    return True, ""


def validate_review_batch(
    output: ReviewBatchOutput,
    claim_count: int,
    existing_group_ids: list[str],
) -> tuple[bool, str]:
    """Validate review batch output semantically.

    Checks:
    1. Every input index (0 to claim_count-1) has exactly one disposition
    2. new_group/add_to_group → must be verifiable_fact
    3. add_to_group/duplicate → group_id must exist in existing_group_ids
    4. drop → classification must NOT be verifiable_fact
    5. New group IDs match those referenced by new_group dispositions
    6. No duplicate new group IDs, no collision with existing
    7. Rationale min length (10 chars)
    """
    # Check 1: Every claim index has a disposition
    disp_indices = {d.claim_index for d in output.dispositions}
    expected = set(range(claim_count))
    missing = expected - disp_indices
    if missing:
        return False, f"Missing dispositions for claim indices: {sorted(missing)}"

    extra = disp_indices - expected
    if extra:
        return False, f"Dispositions reference invalid indices: {sorted(extra)}"

    # Collect new group IDs defined in this batch
    new_group_ids_defined = {ng.group_id for ng in output.new_groups}
    all_known = set(existing_group_ids) | new_group_ids_defined

    for d in output.dispositions:
        # Check 2: grouped actions require verifiable_fact
        if d.action in ("new_group", "add_to_group"):
            if d.classification != "verifiable_fact":
                return False, (
                    f"Claim {d.claim_index}: action '{d.action}' requires "
                    f"verifiable_fact, got '{d.classification}'"
                )

        # Check 3: new_group/add_to_group/duplicate must have group_id
        if d.action in ("new_group", "add_to_group", "duplicate"):
            if not d.group_id:
                return False, f"Claim {d.claim_index}: action '{d.action}' requires group_id"

        if d.action in ("add_to_group", "duplicate"):
            if d.group_id not in all_known:
                return False, (
                    f"Claim {d.claim_index}: group_id '{d.group_id}' not in "
                    f"existing or newly created groups"
                )

        # Check 4: drop must NOT be verifiable_fact
        if d.action == "drop" and d.classification == "verifiable_fact":
            return False, (
                f"Claim {d.claim_index}: cannot drop a verifiable_fact — "
                f"use new_group or add_to_group instead"
            )

        # Check 7: Rationale min length
        if len(d.rationale.strip()) < 10:
            return False, f"Claim {d.claim_index}: rationale too short"

    # Check 5: new_group dispositions must have matching new_groups entry
    new_group_refs = {
        d.group_id for d in output.dispositions if d.action == "new_group"
    }
    if new_group_refs != new_group_ids_defined:
        missing_defs = new_group_refs - new_group_ids_defined
        extra_defs = new_group_ids_defined - new_group_refs
        parts = []
        if missing_defs:
            parts.append(f"referenced but not defined: {missing_defs}")
        if extra_defs:
            parts.append(f"defined but not referenced: {extra_defs}")
        return False, f"new_groups mismatch: {'; '.join(parts)}"

    # Check 6: No collision with existing group IDs
    collisions = new_group_ids_defined & set(existing_group_ids)
    if collisions:
        return False, f"New group IDs collide with existing: {collisions}"

    # Validate new group fields
    for ng in output.new_groups:
        if len(ng.checkability_rationale.strip()) < 10:
            return False, f"Group {ng.group_id}: checkability_rationale too short"

    return True, ""


def validate_synthesized_claim(output: SynthesizedClaim) -> tuple[bool, str]:
    """Validate synthesized claim output.

    Checks:
    1. overarching_claim min length (20 chars)
    2. rationale min length (10 chars)
    """
    if len(output.overarching_claim.strip()) < 20:
        return False, "overarching_claim too short (<20 chars)"
    if len(output.rationale.strip()) < 10:
        return False, "rationale too short (<10 chars)"
    return True, ""
