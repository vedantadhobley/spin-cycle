"""Unified LLM invocation with streaming, parsing, validation, and retry.

This module provides a single entry point for all LLM calls in the pipeline.
It handles the full lifecycle:

  1. INVOKE: Stream tokens from the LLM with idle timeout
  2. PARSE: Extract JSON from raw response
  3. VALIDATE: Check against Pydantic schema
  4. RETRY: On failure, retry

Streaming detects two failure modes early:
  - Server goes silent (no tokens for LLM_IDLE_TIMEOUT seconds)
  - Model produces excessive output without JSON structure
    (LLM_NO_JSON_TOKEN_LIMIT tokens with no '{' seen)
"""

import asyncio
import time
from typing import Any, Callable, Optional, Type, TypeVar

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from src.llm.client import get_llm, LLMProfile
from src.llm.parser import extract_json, JSONExtractionError
from src.config import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_RETRY_DELAY,
    LLM_SERVER_RETRY_DELAY,
    LLM_SERVER_MAX_RETRIES,
)
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


class LLMStreamError(Exception):
    """Raised when streaming detects a stalled or degenerate generation."""
    pass


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


async def _invoke_with_stream(
    llm,
    messages: list,
    activity_name: str,
    expect_json: bool = True,
) -> str:
    """Stream LLM response with idle timeout and sanity checks.

    Wraps llm.astream() with per-chunk idle timeout: if no token arrives
    within LLM_IDLE_TIMEOUT seconds, the generation is aborted.

    When expect_json=True (default), also aborts if LLM_NO_JSON_TOKEN_LIMIT
    tokens are produced without any '{' appearing — catches degenerate
    generations that will never produce parseable output.
    """
    from src.config import LLM_IDLE_TIMEOUT, LLM_NO_JSON_TOKEN_LIMIT

    parts: list[str] = []
    token_count = 0
    found_json = False

    stream = llm.astream(messages)

    try:
        async_iter = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    async_iter.__anext__(),
                    timeout=LLM_IDLE_TIMEOUT,
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                accumulated = "".join(parts)
                log.warning(
                    logger, MODULE, "stream_idle_timeout",
                    f"Aborting {activity_name}: no tokens for "
                    f"{LLM_IDLE_TIMEOUT}s",
                    activity_name=activity_name,
                    token_count=token_count,
                    accumulated_len=len(accumulated),
                )
                raise LLMStreamError(
                    f"Idle timeout ({LLM_IDLE_TIMEOUT}s) after "
                    f"{token_count} tokens"
                )

            token_text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token_text:
                parts.append(token_text)
                token_count += 1

                if not found_json and "{" in token_text:
                    found_json = True

                if (expect_json and not found_json
                        and token_count >= LLM_NO_JSON_TOKEN_LIMIT):
                    accumulated = "".join(parts)
                    log.warning(
                        logger, MODULE, "stream_no_json",
                        f"Aborting {activity_name}: {token_count} tokens "
                        f"with no JSON structure",
                        activity_name=activity_name,
                        token_count=token_count,
                        tail=accumulated[-200:],
                    )
                    raise LLMStreamError(
                        f"No JSON after {token_count} tokens"
                    )
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()

    return "".join(parts)


async def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    *,
    max_retries: int = LLM_MAX_RETRIES,
    profile: LLMProfile = "general",
    max_tokens: int = LLM_MAX_TOKENS,
    presence_penalty: float | None = None,
    semantic_validator: Optional[Callable[[T], tuple[bool, str]]] = None,
    activity_name: str = "invoke",
) -> T:
    """Invoke LLM and return validated, typed output.

    This is the main entry point for LLM calls. It:
    1. Streams tokens from the LLM with idle timeout
    2. Extracts JSON from the response
    3. Validates against the Pydantic schema
    4. Optionally runs semantic validation
    5. Retries on failure

    Args:
        system_prompt: System message content
        user_prompt: User message content
        schema: Pydantic model class to validate against
        max_retries: Number of retry attempts (default: 2)
        profile: "general" or "reasoning" (default: "general")
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
    server_retries = 0  # separate counter for server unavailability

    for attempt in range(max_retries + 1):
        try:
            # Stream tokens from LLM with idle timeout
            llm = get_llm(profile=profile, max_tokens=max_tokens, presence_penalty=presence_penalty)
            _t0 = time.monotonic()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            raw = await _invoke_with_stream(llm, messages, activity_name)

            latency_ms = int((time.monotonic() - _t0) * 1000)
            raw = raw.strip()
            last_raw = raw

            log.debug(logger, MODULE, "llm_response",
                     f"LLM call complete for {activity_name}",
                     attempt=attempt + 1, latency_ms=latency_ms,
                     raw_length=len(raw))

            # Extract JSON from raw response
            try:
                parsed = extract_json(raw)
            except JSONExtractionError as e:
                last_parse_error = str(e)
                log.warning(logger, MODULE, "parse_failed",
                           f"JSON extraction failed for {activity_name}",
                           attempt=attempt + 1, error=str(e))
                continue

            # Validate against Pydantic schema
            try:
                validated = schema.model_validate(parsed)
            except ValidationError as e:
                last_validation_error = str(e)
                log.warning(logger, MODULE, "validation_failed",
                           f"Schema validation failed for {activity_name}",
                           attempt=attempt + 1, error=str(e),
                           schema=schema.__name__)
                continue

            # Run domain-specific semantic validation
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

        except LLMStreamError:
            # Stream aborted early — counts as a failed attempt, retry
            if attempt < max_retries:
                await asyncio.sleep(LLM_RETRY_DELAY)
            continue

        except BaseException as e:
            # CancelledError = Temporal killed the activity (timeout, workflow cancel)
            # Re-raise BaseException subclasses that shouldn't be retried
            if not isinstance(e, Exception):
                raise

            err_str = str(e)
            err_type = type(e).__name__
            is_server_error = (
                "503" in err_str
                or "Loading model" in err_str
                or "Connection error" in err_str
                or "APIConnectionError" in err_type
            )

            if is_server_error and server_retries < LLM_SERVER_MAX_RETRIES:
                server_retries += 1
                log.warning(logger, MODULE, "server_unavailable",
                           f"LLM server unavailable for {activity_name}, "
                           f"waiting {LLM_SERVER_RETRY_DELAY}s before retry",
                           server_retry=server_retries,
                           max_server_retries=LLM_SERVER_MAX_RETRIES,
                           error=err_str, error_type=err_type)
                await asyncio.sleep(LLM_SERVER_RETRY_DELAY)
                # Don't consume an attempt — server errors are transient
                continue

            last_error = err_str
            log.error(logger, MODULE, "invoke_error",
                     f"LLM invocation error for {activity_name}",
                     attempt=attempt + 1, error=err_str,
                     error_type=err_type)
            if attempt < max_retries:
                await asyncio.sleep(LLM_RETRY_DELAY)

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
    profile: LLMProfile = "general",
    activity_name: str = "invoke",
) -> tuple[str, int]:
    """Invoke LLM and return raw output without parsing/validation.

    Use this for cases where custom parsing is needed (e.g., research agent).
    Streams with idle timeout but skips the no-JSON check (raw callers
    may not expect JSON).

    Args:
        system_prompt: System message content
        user_prompt: User message content
        profile: "general" or "reasoning" (default: "general")
        activity_name: Name for logging context

    Returns:
        Tuple of (raw_output, latency_ms)
    """
    llm = get_llm(profile=profile)
    _t0 = time.monotonic()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    raw = await _invoke_with_stream(
        llm, messages, activity_name, expect_json=False,
    )

    latency_ms = int((time.monotonic() - _t0) * 1000)
    raw = raw.strip()

    log.debug(logger, MODULE, "llm_raw_response",
             f"Raw LLM call complete for {activity_name}",
             latency_ms=latency_ms, raw_length=len(raw))

    return raw, latency_ms
