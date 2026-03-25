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

    Checks:
    1. At least one segment exists
    2. Claims have non-empty claim_text (≥10 chars) and original_quote (≥5 chars)

    Intentionally minimal — rationale fields are forcing fields, not gates.
    """
    if not output.segments:
        return False, "Extraction produced no segments"

    for seg in output.segments:
        for i, claim in enumerate(seg.claims):
            if not claim.claim_text or len(claim.claim_text.strip()) < 10:
                return False, (
                    f"Segment {seg.speaker} ({seg.segment_gist}), "
                    f"claim {i+1}: claim_text is empty or too short (<10 chars)"
                )
            if not claim.original_quote or len(claim.original_quote.strip()) < 5:
                return False, (
                    f"Segment {seg.speaker} ({seg.segment_gist}), "
                    f"claim {i+1}: original_quote is empty or too short (<5 chars)"
                )

    return True, ""
