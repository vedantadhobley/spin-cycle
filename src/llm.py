"""Shared LLM client configuration.

All LLM calls in the project go through this module.

  get_llm()           → Non-thinking mode (fast, structured output)
                        Used for ALL pipeline steps (decompose, research,
                        judge, synthesize).

  get_reasoning_llm() → Thinking mode (chain-of-thought reasoning)
                        CURRENTLY UNUSED. Kept for future experiments.
                        llama.cpp has no way to limit thinking tokens,
                        so the model generates 5000-9500 tokens of internal
                        monologue (3-4 min) per call. Not worth the cost.

Both clients hit the **same** Qwen3.5-35B-A3B instance on joi. Qwen3.5
unified thinking and non-thinking into one model — the mode is toggled
per-request via chat_template_kwargs:

  enable_thinking=False  → Direct response, no <think> blocks.
                           Fast, good at structured output and tool-routing.

  enable_thinking=True   → Produces <think>...</think> before answering.
                           Better at weighing conflicting evidence and
                           making nuanced judgments.

Port allocation on joi:
  :3101 — Qwen3.5-35B-A3B (unified thinking + non-thinking)
  :3103 — Embeddings (not yet used)

We use LangChain's ChatOpenAI because joi's llama.cpp server exposes an
OpenAI-compatible /v1/chat/completions endpoint. This means we get all
the LangChain tooling (structured output, tool calling, streaming) for free.
"""

import os

from langchain_openai import ChatOpenAI

from src.utils.logging import log, get_logger

MODULE = "llm"
logger = get_logger()

# Single model instance — thinking mode toggled per-request
LLAMA_URL = os.getenv("LLAMA_URL", "http://joi:3101")
MODEL = "Qwen3.5-35B-A3B"


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Get the LLM client with thinking disabled.

    Use for tasks that need fast, structured output:
      - decompose_claim (JSON array)
      - research_subclaim (ReAct tool-routing — picking search queries)
      - synthesize_verdict (JSON object)

    Thinking is disabled via chat_template_kwargs. This avoids wasting
    ~25-45s per call on <think> blocks that don't improve tool-routing
    or structured output quality.

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            Default 0.1 for fact-checking — consistent, conservative.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=temperature,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    log.debug(logger, MODULE, "llm_init", "LLM client created (thinking=off)",
              base_url=LLAMA_URL, model=MODEL, temperature=temperature)
    return client


def get_reasoning_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Get the LLM client with thinking enabled.

    CURRENTLY UNUSED in the pipeline. Kept for future experiments.

    Problem: llama.cpp doesn't support limiting thinking tokens.
    The model generates 5000-9500 thinking tokens (3-4 min at 40 tok/s)
    before producing output. This makes thinking mode impractical for
    production use without server-side token limits.

    When/if llama.cpp adds max_thinking_tokens support (or we switch
    to vLLM which has it), this can be re-enabled for judge_subclaim.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=MODEL,
        temperature=temperature,
        max_tokens=16384,  # Large buffer for thinking + response
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    log.debug(logger, MODULE, "reasoning_init", "LLM client created (thinking=on)",
              base_url=LLAMA_URL, model=MODEL, temperature=temperature)
    return client
