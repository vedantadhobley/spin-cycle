"""LangGraph graph nodes for claim verification.

Each function is a node in the verification graph. It receives the current
VerificationState and returns updates to that state.
"""

import os
import structlog
from langchain_openai import ChatOpenAI

from src.agent.state import VerificationState

logger = structlog.get_logger()

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080")


def get_llm() -> ChatOpenAI:
    """Get the LLM client pointed at our local model."""
    return ChatOpenAI(
        base_url=f"{LLAMA_URL}/v1",
        api_key="not-needed",
        model="Qwen3-VL-30B-A3B-Instruct",
        temperature=0.1,
    )


async def decompose(state: VerificationState) -> dict:
    """Break the original claim into atomic, verifiable sub-claims."""
    llm = get_llm()

    response = await llm.ainvoke(
        f"""Decompose this claim into independently verifiable sub-claims.
Return a JSON array of strings, each a single factual assertion.

Claim: {state["claim_text"]}

Return ONLY a JSON array. /no_think"""
    )

    # TODO: Parse JSON response into sub_claims
    logger.info("decompose", claim=state["claim_text"], response=response.content)

    return {
        "sub_claims": [],  # TODO: parse
        "current_sub_claim_index": 0,
        "research_iterations": 0,
    }


async def research(state: VerificationState) -> dict:
    """Research evidence for the current sub-claim using tools."""
    # TODO: Use LangGraph tool nodes for web search, Wikipedia, news APIs
    logger.info(
        "research",
        sub_claim_index=state["current_sub_claim_index"],
        iteration=state["research_iterations"],
    )

    return {
        "research_iterations": state["research_iterations"] + 1,
        "evidence": state.get("evidence", []),
    }


async def evaluate_evidence(state: VerificationState) -> dict:
    """Evaluate whether we have enough evidence to render a verdict."""
    llm = get_llm()

    # TODO: Ask LLM if evidence is sufficient
    # If not, set needs_more_research = True to loop back

    return {
        "needs_more_research": False,  # TODO
    }


async def judge(state: VerificationState) -> dict:
    """Render verdict on the current sub-claim based on collected evidence."""
    llm = get_llm()

    # TODO: LLM evaluates evidence for/against the sub-claim
    logger.info("judge", sub_claim_index=state["current_sub_claim_index"])

    return {
        "sub_claims": state["sub_claims"],  # TODO: update with verdict
    }


async def synthesize(state: VerificationState) -> dict:
    """Combine sub-claim verdicts into an overall claim verdict."""
    llm = get_llm()

    # TODO: Aggregate sub-verdicts
    logger.info("synthesize", num_sub_claims=len(state.get("sub_claims", [])))

    return {
        "verdict": None,  # TODO
        "confidence": None,  # TODO
    }
