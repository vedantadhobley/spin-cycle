"""LLM client configuration.

This module provides configured LLM clients for the verification pipeline.
All clients hit the same Qwen3.5-122B-A10B instance via llama.cpp's
OpenAI-compatible API.

  get_llm() → Non-thinking mode (fast, structured output)
"""

import os

from langchain_openai import ChatOpenAI

from src.utils.logging import log, get_logger

MODULE = "llm"
logger = get_logger()

# Single model instance — thinking disabled for structured output
LLAMA_URL = os.getenv("LLAMA_URL")
if not LLAMA_URL:
    raise RuntimeError("LLAMA_URL environment variable is required")
MODEL = os.getenv("LLAMA_MODEL", "Qwen3.5-122B-A10B")


def get_llm(temperature: float = 0.1, max_tokens: int = 8192) -> ChatOpenAI:
    """Get the LLM client with thinking disabled.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            Default 0.1 for fact-checking — consistent, conservative.
        max_tokens: Maximum output tokens. Default 8192, sufficient for
            verification pipeline calls. Callers processing large inputs
            (e.g. transcript extraction) may need higher values.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    log.debug(logger, MODULE, "llm_init", "LLM client created (thinking=off)",
              base_url=LLAMA_URL, model=MODEL, temperature=temperature)
    return client


