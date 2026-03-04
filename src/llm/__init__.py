"""LLM invocation package.

This package provides a unified interface for all LLM calls:

  from src.llm import invoke_llm, get_llm

  # Validated invocation (preferred)
  result = await invoke_llm(
      system_prompt=DECOMPOSE_SYSTEM,
      user_prompt=DECOMPOSE_USER.format(...),
      schema=DecomposeOutput,
      semantic_validator=validate_decompose,
  )

  # Raw client access (for custom use cases like agents)
  llm = get_llm()
  response = await llm.ainvoke([...])

Architecture:
  client.py    → LLM client configuration (ChatOpenAI instances)
  parser.py    → JSON extraction from raw LLM output
  validators.py → Semantic validation beyond schema checks
  invoker.py   → Unified invoke-parse-validate-retry logic

The invoker implements defense-in-depth:
  1. PROMPT: Tell LLM what format to produce
  2. PARSE: Extract JSON, handling markdown wrappers
  3. SCHEMA: Validate against Pydantic model
  4. SEMANTIC: Domain-specific validation
  5. RETRY: On failure, retry with higher temperature
"""

# Client access
from src.llm.client import get_llm

# Unified invocation
from src.llm.invoker import (
    invoke_llm,
    invoke_llm_raw,
    LLMInvocationError,
    InvocationResult,
)

# Parsing utilities
from src.llm.parser import (
    extract_json,
    JSONExtractionError,
)

# Validators
from src.llm.validators import (
    validate_normalize,
    validate_decompose,
    validate_judge,
    validate_synthesize,
)

__all__ = [
    # Client
    "get_llm",
    # Invoker
    "invoke_llm",
    "invoke_llm_raw",
    "LLMInvocationError",
    "InvocationResult",
    # Parser
    "extract_json",
    "JSONExtractionError",
    # Validators
    "validate_normalize",
    "validate_decompose",
    "validate_judge",
    "validate_synthesize",
]
