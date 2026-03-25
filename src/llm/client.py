"""LLM client configuration.

This module provides configured LLM clients for the verification pipeline.
All clients hit the same Qwen3.5-122B-A10B instance via llama.cpp's
OpenAI-compatible API.

  get_llm()                → Non-thinking mode (fast, structured output)
  get_llm(thinking=True)   → Thinking mode (slower, deeper reasoning)
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


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 8192,
    thinking: bool = False,
) -> ChatOpenAI:
    """Get the LLM client.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            Default 0.1 for fact-checking — consistent, conservative.
            Overridden to 0.6 when thinking=True (Qwen3.5 recommended
            minimum for thinking mode).
        max_tokens: Maximum output tokens. Default 8192, sufficient for
            non-thinking verification calls. When thinking=True, forced
            to at least 32768 — thinking tokens count against the limit
            and can easily consume 10-15K before the actual output.
        thinking: Enable thinking/reasoning mode. The model gets an
            internal scratchpad before producing structured output.
            Significantly slower (5-10 min per call vs 1-3 min) but
            better at cross-referencing evidence and catching contradictions.
            Sets Qwen-recommended sampling: temp=0.6, top_p=0.95,
            presence_penalty=1.5, top_k=20.
    """
    extra_body: dict = {
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    if thinking:
        temperature = 0.6
        max_tokens = max(max_tokens, 32768)
        extra_body["top_k"] = 20

    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95 if thinking else None,
        presence_penalty=1.5 if thinking else None,
        extra_body=extra_body,
    )
    mode = "thinking" if thinking else "instruct"
    log.debug(logger, MODULE, "llm_init", f"LLM client created ({mode})",
              base_url=LLAMA_URL, model=MODEL, temperature=temperature,
              max_tokens=max_tokens, thinking=thinking)
    return client


