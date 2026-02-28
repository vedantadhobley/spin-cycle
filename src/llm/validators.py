"""Semantic validators for LLM outputs.

These validators check domain-specific constraints BEYOND schema validation.
Schema validation ensures the JSON has the right shape.
Semantic validation ensures the content makes sense.

Examples:
- Decompose: At least one predicate or comparison must exist
- Decompose: No predicate should have multiple {entity} placeholders
- Judge: Verdict must be one of the valid values
- Judge: Confidence and verdict should be consistent
"""

from typing import Callable, Tuple

from src.llm.placeholders import (
    count_entity_placeholders,
    has_entity_placeholder,
    find_non_compliant,
)
from src.schemas.llm_outputs import (
    DecomposeOutput,
    JudgeOutput,
    SynthesizeOutput,
)
from src.utils.logging import log, get_logger

MODULE = "llm.validators"
logger = get_logger()

# Type alias for validator functions
ValidatorFunc = Callable[[any], Tuple[bool, str]]


def validate_decompose(output: DecomposeOutput) -> tuple[bool, str]:
    """Validate decompose output semantically.
    
    Checks:
    1. Has at least one predicate or comparison (not empty)
    2. No predicate uses {entity} more than once
    3. All predicates have at least one entity in applies_to
    
    Args:
        output: The decompose output to validate
        
    Returns:
        (is_valid, error_message) tuple
    """
    # Check 1: Not empty
    if not output.predicates and not output.comparisons:
        return False, "Decomposition produced no predicates or comparisons"
    
    # Check 2: No double {entity} in predicates (uses centralized placeholder logic)
    for i, pred in enumerate(output.predicates):
        entity_count = count_entity_placeholders(pred.claim)
        if entity_count > 1:
            log.warning(logger, MODULE, "double_entity",
                       f"Predicate {i} has multiple {{entity}} placeholders",
                       predicate=pred.claim)
            # Don't fail — the code-level safeguard handles this
        
        # Check for non-compliant placeholders (target_entity, etc.)
        non_compliant = find_non_compliant(pred.claim)
        if non_compliant:
            log.warning(logger, MODULE, "non_compliant_placeholder",
                       f"Predicate {i} uses non-compliant placeholder '{non_compliant}'",
                       predicate=pred.claim)
            # Don't fail — the expansion code handles this
    
    # Check 3: Predicates have applies_to
    for i, pred in enumerate(output.predicates):
        if has_entity_placeholder(pred.claim):
            if not pred.applies_to:
                return False, f"Predicate {i} uses {{entity}} but has empty applies_to"
    
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
decompose_validator = validate_decompose
judge_validator = validate_judge
synthesize_validator = validate_synthesize
