"""Semantic validators for LLM outputs.

These validators check domain-specific constraints BEYOND schema validation.
Schema validation ensures the JSON has the right shape.
Semantic validation ensures the content makes sense.

Examples:
- Decompose: At least one fact must exist
- Judge: Verdict must be one of the valid values
- Judge: Confidence and verdict should be consistent
"""

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

    return True, ""


def validate_judge(output: JudgeOutput) -> tuple[bool, str]:
    """Validate judge output semantically.
    
    Checks:
    1. Confidence and verdict are consistent
       - "true" with confidence < 0.3 is suspicious
       - "unverifiable" with confidence > 0.8 is suspicious
    2. Reasoning is not empty/trivial
    
    Args:
        output: The judge output to validate
        
    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Confidence/verdict consistency
    # These are warnings, not hard failures — LLM might have good reasons
    if output.verdict in ("true", "false") and output.confidence < 0.3:
        log.warning(logger, MODULE, "low_confidence_strong_verdict",
                   f"Strong verdict '{output.verdict}' with low confidence {output.confidence}",
                   verdict=output.verdict, confidence=output.confidence)
    
    if output.verdict == "unverifiable" and output.confidence > 0.8:
        log.warning(logger, MODULE, "high_confidence_unverifiable",
                   f"Unverifiable verdict with high confidence {output.confidence}",
                   verdict=output.verdict, confidence=output.confidence)
    
    # Check 2: Reasoning exists
    if not output.reasoning or len(output.reasoning.strip()) < 10:
        return False, "Reasoning is empty or too short"
    
    return True, ""


def validate_synthesize(output: SynthesizeOutput) -> tuple[bool, str]:
    """Validate synthesize output semantically.
    
    Checks:
    1. Reasoning is not empty/trivial
    2. Confidence/verdict consistency
    
    Args:
        output: The synthesize output to validate
        
    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Reasoning exists
    if not output.reasoning or len(output.reasoning.strip()) < 10:
        return False, "Reasoning is empty or too short"
    
    # Check 2: Same confidence/verdict checks as judge
    if output.verdict in ("true", "false") and output.confidence < 0.3:
        log.warning(logger, MODULE, "low_confidence_strong_verdict",
                   f"Strong verdict '{output.verdict}' with low confidence {output.confidence}")
    
    return True, ""


# Export validator functions for use with invoke_llm
normalize_validator = validate_normalize
decompose_validator = validate_decompose
judge_validator = validate_judge
synthesize_validator = validate_synthesize
