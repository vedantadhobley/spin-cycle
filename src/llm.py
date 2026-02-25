"""Shared LLM client configuration.

All LLM calls in the project go through this module. It provides two clients:

  get_llm()           → Instruct model (fast, structured output)
                        Used for decompose + synthesize steps.

  get_reasoning_llm() → Thinking model (slower, chain-of-thought reasoning)
                        Used for research + judge steps.

Both models run on joi (our local llama.cpp server via Tailscale MagicDNS).
They share the same Qwen3-VL-30B-A3B architecture (30B params, 3B active)
but are fine-tuned differently:

  - Instruct: optimised for instruction following and structured output.
    No chain-of-thought — fast, direct responses.

  - Thinking: optimised for multi-step reasoning. Produces <think>...</think>
    blocks before answering. Better at weighing conflicting evidence and
    making nuanced judgments.

Port allocation on joi:
  :3101 — Instruct (Qwen3-VL-30B-A3B-Instruct)
  :3102 — Thinking (Qwen3-VL-30B-A3B-Thinking)
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

# Instruct model — fast, structured output
LLAMA_URL = os.getenv("LLAMA_URL", "http://joi:3101")
INSTRUCT_MODEL = "Qwen3-VL-30B-A3B-Instruct"

# Thinking model — slower, better reasoning
LLAMA_REASONING_URL = os.getenv("LLAMA_REASONING_URL", "http://joi:3102")
REASONING_MODEL = "Qwen3-VL-30B-A3B-Thinking"


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Get the instruct LLM client.

    Use for tasks that need fast, structured output:
      - decompose_claim (JSON array)
      - research_subclaim (ReAct tool-routing — picking search queries)
      - synthesize_verdict (JSON object)

    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
            Default 0.1 for fact-checking — consistent, conservative.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model=INSTRUCT_MODEL,
        temperature=temperature,
        max_tokens=2048,
    )
    log.debug(logger, MODULE, "instruct_init", "Instruct LLM client created",
              base_url=LLAMA_URL, model=INSTRUCT_MODEL, temperature=temperature)
    return client


def get_reasoning_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Get the thinking/reasoning LLM client.

    Use for tasks that benefit from chain-of-thought reasoning:
      - judge_subclaim (weighing conflicting evidence, calibrating confidence)

    NOT used for research — the ReAct loop is pure tool-routing where
    <think> blocks waste ~25-45s per iteration without improving query
    quality.  Research uses the instruct model via get_llm() instead.

    The thinking model produces <think>...</think> blocks before its answer.
    Callers should strip these before parsing structured output.

    Slightly higher default temperature (0.2) to allow more exploratory
    reasoning without becoming unreliable.
    """
    client = ChatOpenAI(
        base_url=f"{LLAMA_REASONING_URL}/v1",
        api_key="not-needed",
        model=REASONING_MODEL,
        temperature=temperature,
        max_tokens=4096,
    )
    log.debug(logger, MODULE, "reasoning_init", "Reasoning LLM client created",
              base_url=LLAMA_REASONING_URL, model=REASONING_MODEL, temperature=temperature)
    return client
