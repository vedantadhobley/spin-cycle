"""Centralized placeholder handling for LLM predicate templates.

This module provides the single source of truth for:
1. What the canonical placeholder is ({entity})
2. What non-compliant patterns LLMs sometimes produce
3. How to detect, normalize, and expand placeholders

The decomposition prompt asks the LLM to produce predicate templates like:
  {"claim": "{entity} is cutting aid", "applies_to": ["US", "China"]}

But LLMs sometimes produce non-compliant variations:
  - target_entity, other_entity, source_entity
  - entity1, entity2
  - the_entity
  - {subject}, {actor}

This module catches these mistakes and normalizes them.
"""

import re
from typing import Optional, List, Tuple

from src.utils.logging import log, get_logger

MODULE = "llm.placeholders"
logger = get_logger()


# =============================================================================
# CANONICAL PLACEHOLDERS
# =============================================================================

# The placeholder the prompt asks for
ENTITY_PLACEHOLDER = "{entity}"
VALUE_PLACEHOLDER = "{value}"

# Regex patterns to match placeholder variations
# Handles {entity}, {{entity}}, and case variations
ENTITY_PATTERN = re.compile(r'\{+entity\}+', re.IGNORECASE)
VALUE_PATTERN = re.compile(r'\{+value\}+', re.IGNORECASE)

# Generic cleanup - any remaining {something} after expansion
LEFTOVER_BRACE_PATTERN = re.compile(r'\{+([^{}]+)\}+')


# =============================================================================
# NON-COMPLIANT PATTERNS
# =============================================================================

# Patterns LLMs sometimes use instead of {entity}
# Each tuple: (compiled_pattern, human_readable_name)
NON_COMPLIANT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\btarget_entity\b', re.IGNORECASE), "target_entity"),
    (re.compile(r'\bother_entity\b', re.IGNORECASE), "other_entity"),
    (re.compile(r'\bsource_entity\b', re.IGNORECASE), "source_entity"),
    (re.compile(r'\bentity_?[12]\b', re.IGNORECASE), "entity1/entity2"),
    (re.compile(r'\bthe_entity\b', re.IGNORECASE), "the_entity"),
    (re.compile(r'\bsubject_entity\b', re.IGNORECASE), "subject_entity"),
    (re.compile(r'\bactor_entity\b', re.IGNORECASE), "actor_entity"),
]

# List for prompt documentation
NON_COMPLIANT_NAMES = [name for _, name in NON_COMPLIANT_PATTERNS]


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def count_entity_placeholders(template: str) -> int:
    """Count {entity} placeholders in a template.
    
    Handles both {entity} and {{entity}} as single occurrences.
    
    Args:
        template: The predicate template string
        
    Returns:
        Number of distinct placeholder positions
    """
    return len(ENTITY_PATTERN.findall(template))


def has_entity_placeholder(template: str) -> bool:
    """Check if template has any {entity} placeholder."""
    return bool(ENTITY_PATTERN.search(template))


def find_non_compliant(template: str) -> Optional[str]:
    """Check if template has non-compliant placeholders.
    
    Args:
        template: The predicate template string
        
    Returns:
        Name of first non-compliant pattern found, or None
    """
    for pattern, name in NON_COMPLIANT_PATTERNS:
        if pattern.search(template):
            return name
    return None


def has_non_compliant(template: str) -> bool:
    """Check if template has any non-compliant placeholder."""
    return find_non_compliant(template) is not None


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_non_compliant(
    template: str,
    replacement: str = ENTITY_PLACEHOLDER,
) -> Tuple[str, bool]:
    """Replace non-compliant placeholders with a standard form.
    
    Args:
        template: The predicate template string
        replacement: What to replace non-compliant patterns with.
                    Use ENTITY_PLACEHOLDER to normalize, or an actual
                    entity name to expand directly.
        
    Returns:
        Tuple of (normalized_template, was_modified)
    """
    original = template
    for pattern, _ in NON_COMPLIANT_PATTERNS:
        template = pattern.sub(replacement, template)
    return template, template != original


def cleanup_leftover_braces(text: str) -> str:
    """Remove any remaining {placeholder} patterns after expansion.
    
    Handles nested braces like {{value}} → value
    
    Args:
        text: Text that may have leftover placeholders
        
    Returns:
        Cleaned text with braces removed
    """
    # Keep replacing until no more braces
    while '{' in text and '}' in text:
        new_text = LEFTOVER_BRACE_PATTERN.sub(r'\1', text)
        if new_text == text:
            break  # No more matches
        text = new_text
    return text


# =============================================================================
# EXPANSION FUNCTIONS
# =============================================================================

def expand_template(
    template: str,
    entity: str,
    value: str = "",
) -> str:
    """Expand {entity} and {value} placeholders with actual values.
    
    Args:
        template: Predicate template with placeholders
        entity: Value to substitute for {entity}
        value: Value to substitute for {value} (optional)
        
    Returns:
        Expanded string with placeholders replaced
    """
    result = ENTITY_PATTERN.sub(entity, template)
    if value:
        result = VALUE_PATTERN.sub(value, result)
    result = cleanup_leftover_braces(result)
    return result.strip()


def expand_predicate(
    claim_template: str,
    applies_to: list,
) -> List[str]:
    """Expand a predicate template into concrete facts.
    
    This is the main expansion function. It handles:
    1. Non-compliant placeholders (target_entity, etc.)
    2. Multiple {entity} placeholders (shouldn't happen)
    3. Normal single {entity} expansion
    4. Value substitution for detailed applies_to entries
    
    Args:
        claim_template: The predicate template string
        applies_to: List of entities or {"entity": "...", "value": "..."} dicts
        
    Returns:
        List of expanded fact strings
    """
    facts = []
    
    # Check for non-compliant placeholders
    non_compliant = find_non_compliant(claim_template)
    has_proper = has_entity_placeholder(claim_template)
    
    if non_compliant and not has_proper:
        # LLM used non-standard placeholder without proper {entity}
        if applies_to:
            # Replace with first entity and return as single fact
            first_entity = _get_entity_from_item(applies_to[0])
            fixed, _ = normalize_non_compliant(claim_template, first_entity)
            fixed = cleanup_leftover_braces(fixed).strip()
            log.warning(logger, MODULE, "fixed_non_compliant",
                       f"Fixed non-compliant placeholder '{non_compliant}'",
                       original=claim_template, fixed=fixed)
            if fixed:
                facts.append(fixed)
        else:
            # No applies_to - use placeholder literal as fallback
            fixed, _ = normalize_non_compliant(claim_template, "[entity]")
            fixed = cleanup_leftover_braces(fixed).strip()
            log.warning(logger, MODULE, "orphan_non_compliant",
                       f"Non-compliant placeholder '{non_compliant}' with no applies_to",
                       original=claim_template, fixed=fixed)
            if fixed:
                facts.append(fixed)
        return facts
    
    # Check for multiple {entity} placeholders (LLM mistake)
    entity_count = count_entity_placeholders(claim_template)
    if entity_count > 1:
        # Multiple placeholders = LLM mistake
        # Strip braces and use as-is
        fact = cleanup_leftover_braces(claim_template).strip()
        log.warning(logger, MODULE, "multiple_placeholders",
                   f"Found {entity_count} {{entity}} placeholders",
                   template=claim_template, fixed=fact)
        if fact:
            facts.append(fact)
        return facts
    
    # Normal expansion: one {entity} placeholder
    if entity_count == 1:
        for item in applies_to:
            entity = _get_entity_from_item(item)
            value = _get_value_from_item(item)
            if entity:
                fact = expand_template(claim_template, entity, value)
                if fact:
                    facts.append(fact)
    else:
        # No {entity} placeholder - use template as-is
        fact = cleanup_leftover_braces(claim_template).strip()
        if fact:
            facts.append(fact)
    
    return facts


def _get_entity_from_item(item) -> str:
    """Extract entity name from applies_to item."""
    if isinstance(item, str):
        return item.strip()
    elif isinstance(item, dict):
        return item.get("entity", "").strip()
    return ""


def _get_value_from_item(item) -> str:
    """Extract value from applies_to item (for detailed format)."""
    if isinstance(item, dict):
        return item.get("value", "").strip()
    return ""
