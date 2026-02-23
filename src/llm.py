"""Shared LLM client configuration.

All LLM calls in the project go through this module. It handles:
- Connection to joi (our local llama.cpp server via Docker DNS)
- Model selection
- Default parameters

The LLM runs on a separate machine (joi) accessed via Tailscale.
Docker DNS resolves the hostname → systemd-resolved → Tailscale MagicDNS.

We use LangChain's ChatOpenAI because joi's llama.cpp server exposes an
OpenAI-compatible /v1/chat/completions endpoint. This means we get all
the LangChain tooling (structured output, tool calling, streaming) for free.
"""

import os

from langchain_openai import ChatOpenAI

LLAMA_URL = os.getenv("LLAMA_URL", "http://joi:3101")

# The model name must match what's loaded in llama.cpp.
# Check loaded models: curl http://joi:3101/v1/models
MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Get an LLM client pointed at joi.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            We default to 0.1 for fact-checking — we want consistent,
            conservative responses, not creative ones.
    """
    return ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",  # local model, no auth required
        model=MODEL_NAME,
        temperature=temperature,
    )
