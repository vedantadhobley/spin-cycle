"""LLM client configuration.

This module provides configured LLM clients for the verification pipeline.
All clients hit the same Qwen3.5-35B-A3B instance on joi via llama.cpp's
OpenAI-compatible API.

  get_llm()           → Non-thinking mode (fast, structured output)
  get_reasoning_llm() → Thinking mode (chain-of-thought reasoning)

Mode is toggled per-request via chat_template_kwargs. Thinking mode is
currently unused because llama.cpp doesn't support limiting thinking tokens.
"""

import os

from langchain_openai import ChatOpenAI

from src.utils.logging import log, get_logger

MODULE = "llm"
logger = get_logger()

# Single model instance — thinking mode toggled per-request
LLAMA_URL = os.getenv("LLAMA_URL", "http://joi:3101")
MODEL = os.getenv("LLAMA_MODEL", "Qwen3.5-35B-A3B")


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Get the LLM client with thinking disabled.

    Use for tasks that need fast, structured output:
      - decompose_claim (JSON array)
      - research_subclaim (ReAct tool-routing)
      - synthesize_verdict (JSON object)

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            Default 0.1 for fact-checking — consistent, conservative.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=temperature,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    log.debug(logger, MODULE, "llm_init", "LLM client created (thinking=off)",
              base_url=LLAMA_URL, model=MODEL, temperature=temperature)
    return client


def get_reasoning_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Get the LLM client with thinking enabled.

    CURRENTLY UNUSED in the pipeline. Kept for future experiments.
    llama.cpp doesn't support limiting thinking tokens, so the model
    generates 3-4 minutes of internal monologue before responding.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=temperature,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    log.debug(logger, MODULE, "reasoning_init", "LLM client created (thinking=on)",
              base_url=LLAMA_URL, model=MODEL, temperature=temperature)
    return client
