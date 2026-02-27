"""Unified LLM invocation with parsing, validation, and retry.

This module provides a single entry point for all LLM calls in the pipeline.
It handles the full lifecycle:

  1. INVOKE: Call the LLM with system/user messages
  2. PARSE: Extract JSON from raw response
  3. VALIDATE: Check against Pydantic schema
  4. RETRY: On failure, retry with adjusted parameters

This eliminates scattered try/except blocks and ensures consistent
error handling across all activities.
"""

import asyncio
import time
from typing import Any, Callable, Optional, Type, TypeVar

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from src.llm.client import get_llm
from src.llm.parser import extract_json, JSONExtractionError
from src.utils.logging import log, get_logger

MODULE = "llm.invoker"
logger = get_logger()

T = TypeVar("T", bound=BaseModel)


class LLMInvocationError(Exception):
    """Raised when LLM invocation fails after all retries."""
    
    def __init__(
        self,
        message: str,
        raw_output: Optional[str] = None,
        parse_error: Optional[str] = None,
        validation_error: Optional[str] = None,
        attempts: int = 0,
    ):
        super().__init__(message)
        self.raw_output = raw_output
        self.parse_error = parse_error
        self.validation_error = validation_error
        self.attempts = attempts


class InvocationResult(BaseModel):
    """Result of an LLM invocation."""
    
    class Config:
        arbitrary_types_allowed = True
    
    success: bool
    data: Optional[Any] = None
    raw_output: str = ""
    latency_ms: int = 0
    attempts: int = 1
    error: Optional[str] = None


async def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    *,
    max_retries: int = 2,
    temperature: float = 0.1,
    temperature_on_retry: float = 0.3,
    semantic_validator: Optional[Callable[[T], tuple[bool, str]]] = None,
    activity_name: str = "invoke",
) -> T:
    """Invoke LLM and return validated, typed output.
    
    This is the main entry point for LLM calls. It:
    1. Calls the LLM with the given prompts
    2. Extracts JSON from the response
    3. Validates against the Pydantic schema
    4. Optionally runs semantic validation
    5. Retries on failure with higher temperature
    
    Args:
        system_prompt: System message content
        user_prompt: User message content
        schema: Pydantic model class to validate against
        max_retries: Number of retry attempts (default: 2)
        temperature: Initial temperature (default: 0.1)
        temperature_on_retry: Temperature for retry attempts (default: 0.3)
        semantic_validator: Optional function (model) -> (is_valid, error_msg)
        activity_name: Name for logging context
        
    Returns:
        Validated instance of the schema type
        
    Raises:
        LLMInvocationError: If all attempts fail
    """
    last_error: Optional[str] = None
    last_raw: Optional[str] = None
    last_parse_error: Optional[str] = None
    last_validation_error: Optional[str] = None
    
    for attempt in range(max_retries + 1):
        current_temp = temperature if attempt == 0 else temperature_on_retry
        
        try:
            # Step 1: INVOKE
            llm = get_llm(temperature=current_temp)
            _t0 = time.monotonic()
            
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            
            latency_ms = int((time.monotonic() - _t0) * 1000)
            raw = response.content.strip()
            last_raw = raw
            
            log.debug(logger, MODULE, "llm_response",
                     f"LLM call complete for {activity_name}",
                     attempt=attempt + 1, latency_ms=latency_ms,
                     raw_length=len(raw))
            
            # Step 2: PARSE
            try:
                parsed = extract_json(raw)
            except JSONExtractionError as e:
                last_parse_error = str(e)
                log.warning(logger, MODULE, "parse_failed",
                           f"JSON extraction failed for {activity_name}",
                           attempt=attempt + 1, error=str(e))
                continue
            
            # Step 3: VALIDATE (schema)
            try:
                validated = schema.model_validate(parsed)
            except ValidationError as e:
                last_validation_error = str(e)
                log.warning(logger, MODULE, "validation_failed",
                           f"Schema validation failed for {activity_name}",
                           attempt=attempt + 1, error=str(e),
                           schema=schema.__name__)
                continue
            
            # Step 4: VALIDATE (semantic)
            if semantic_validator:
                is_valid, semantic_error = semantic_validator(validated)
                if not is_valid:
                    last_validation_error = f"Semantic: {semantic_error}"
                    log.warning(logger, MODULE, "semantic_failed",
                               f"Semantic validation failed for {activity_name}",
                               attempt=attempt + 1, error=semantic_error)
                    continue
            
            # Success!
            log.info(logger, MODULE, "invoke_success",
                    f"LLM invocation successful for {activity_name}",
                    attempts=attempt + 1, latency_ms=latency_ms,
                    schema=schema.__name__)
            
            return validated
            
        except Exception as e:
            last_error = str(e)
            log.error(logger, MODULE, "invoke_error",
                     f"LLM invocation error for {activity_name}",
                     attempt=attempt + 1, error=str(e))
            if attempt < max_retries:
                await asyncio.sleep(1)  # Brief pause before retry
    
    # All attempts failed
    raise LLMInvocationError(
        f"LLM invocation failed for {activity_name} after {max_retries + 1} attempts",
        raw_output=last_raw,
        parse_error=last_parse_error,
        validation_error=last_validation_error,
        attempts=max_retries + 1,
    )


async def invoke_llm_raw(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.1,
    activity_name: str = "invoke",
) -> tuple[str, int]:
    """Invoke LLM and return raw output without parsing/validation.
    
    Use this for cases where custom parsing is needed (e.g., research agent).
    
    Args:
        system_prompt: System message content
        user_prompt: User message content
        temperature: Temperature setting
        activity_name: Name for logging context
        
    Returns:
        Tuple of (raw_output, latency_ms)
    """
    llm = get_llm(temperature=temperature)
    _t0 = time.monotonic()
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    latency_ms = int((time.monotonic() - _t0) * 1000)
    raw = response.content.strip()
    
    log.debug(logger, MODULE, "llm_raw_response",
             f"Raw LLM call complete for {activity_name}",
             latency_ms=latency_ms, raw_length=len(raw))
    
    return raw, latency_ms


def create_fallback(
    schema: Type[T],
    fallback_data: dict,
    reason: str,
) -> T:
    """Create a fallback instance when LLM invocation fails.
    
    Use this to provide graceful degradation instead of crashing.
    
    Args:
        schema: Pydantic model class
        fallback_data: Data to populate the fallback instance
        reason: Reason for using fallback (for logging)
        
    Returns:
        Instance of the schema type
    """
    log.warning(logger, MODULE, "using_fallback",
               f"Using fallback for {schema.__name__}",
               reason=reason)
    return schema.model_validate(fallback_data)
