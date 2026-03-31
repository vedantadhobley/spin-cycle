"""LLM client configuration.

This module provides configured LLM clients for the verification pipeline.
All clients hit the same Qwen3.5-122B-A10B instance via llama.cpp's
OpenAI-compatible API.

Two sampling profiles from the Qwen3.5 model card:
  - "general": extraction, classification, summarization, tool routing
  - "reasoning": decompose, judge, synthesize verdict

Thinking mode was tested and reverted — see ARCHITECTURE.md.
"""

import os
from typing import Literal

from langchain_openai import ChatOpenAI

from src.config import (
    LLM_GENERAL_TEMPERATURE,
    LLM_GENERAL_TOP_P,
    LLM_GENERAL_TOP_K,
    LLM_GENERAL_PRESENCE_PENALTY,
    LLM_REASONING_TEMPERATURE,
    LLM_REASONING_TOP_P,
    LLM_REASONING_TOP_K,
    LLM_REASONING_PRESENCE_PENALTY,
)
from src.utils.logging import log, get_logger

MODULE = "llm"
logger = get_logger()

LLAMA_URL = os.getenv("LLAMA_URL")
if not LLAMA_URL:
    raise RuntimeError("LLAMA_URL environment variable is required")
MODEL = os.getenv("LLAMA_MODEL", "Qwen3.5-122B-A10B")

# Profiles map to Qwen3.5 model card recommended sampling parameters.
_PROFILES = {
    "general": {
        "temperature": LLM_GENERAL_TEMPERATURE,
        "top_p": LLM_GENERAL_TOP_P,
        "top_k": LLM_GENERAL_TOP_K,
        "presence_penalty": LLM_GENERAL_PRESENCE_PENALTY,
    },
    "reasoning": {
        "temperature": LLM_REASONING_TEMPERATURE,
        "top_p": LLM_REASONING_TOP_P,
        "top_k": LLM_REASONING_TOP_K,
        "presence_penalty": LLM_REASONING_PRESENCE_PENALTY,
    },
}

LLMProfile = Literal["general", "reasoning"]


def get_llm(
    profile: LLMProfile = "general",
    max_tokens: int = 8192,
) -> ChatOpenAI:
    """Get the LLM client with Qwen3.5 recommended sampling.

    Args:
        profile: "general" for extraction/classification/routing,
            "reasoning" for decompose/judge/synthesize.
        max_tokens: Maximum output tokens. Default 8192.
    """
    params = _PROFILES[profile]
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=params["temperature"],
        max_tokens=max_tokens,
        top_p=params["top_p"],
        presence_penalty=params["presence_penalty"],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": params["top_k"],
        },
    )
    log.debug(logger, MODULE, "llm_init", f"LLM client created ({profile})",
              base_url=LLAMA_URL, model=MODEL,
              temperature=params["temperature"], max_tokens=max_tokens)
    return client
