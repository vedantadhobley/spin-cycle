"""Domain logic for verdict synthesis.

Combines child sub-verdicts into a final overall verdict.

The Temporal activity wrapper in verify_activities.py calls synthesize() here.
"""

from datetime import date

from src.llm import invoke_llm, LLMInvocationError, validate_synthesize
from src.prompts.verification import SYNTHESIZE_SYSTEM, SYNTHESIZE_USER
from src.schemas.llm_outputs import SynthesizeOutput
from src.utils.logging import log, get_logger
from src.utils.text_cleanup import cleanup_text

MODULE = "synthesize"
logger = get_logger()


async def synthesize(
    claim_text: str,
    child_results: list[dict],
    thesis_info: dict | None = None,
) -> dict:
    """Combine child verdicts into a final overall verdict.

    Args:
        claim_text: The original claim text.
        child_results: List of sub-verdict dicts from the judge step.
        thesis_info: Thesis metadata from decompose (thesis, structure, key_test).

    Returns:
        Dict: sub_claim, verdict, confidence, reasoning, evidence,
              child_results, reasoning_chain.
    """
    log.info(logger, MODULE, "start", "Synthesizing verdict",
             claim=claim_text, num_children=len(child_results))

    # Format sub-verdicts for the LLM prompt
    sub_verdict_parts = []
    for i, sub in enumerate(child_results, 1):
        part = (
            f"[{i}] Sub-claim: {sub['sub_claim']}\n"
            f"    Verdict: {sub['verdict']}\n"
            f"    Confidence: {sub['confidence']}\n"
            f"    Reasoning: {sub['reasoning']}"
        )
        sub_verdict_parts.append(part)
    sub_verdicts_text = "\n\n".join(sub_verdict_parts)

    synthesis_context = (
        "This is the FINAL OVERALL verdict for the original claim. "
        "Your verdict is the definitive assessment."
    )
    # Build thesis context for the synthesizer
    thesis_block = ""
    if thesis_info and thesis_info.get("thesis"):
        thesis_block = (
            f"\n\nSPEAKER'S THESIS: {thesis_info['thesis']}\n"
            f"Claim structure: {thesis_info.get('structure', 'simple')}\n"
            f"Key test: {thesis_info.get('key_test', 'N/A')}\n"
            f"\nEvaluate whether THIS THESIS survives the sub-verdicts, "
            f"not just whether a majority of individual facts are true."
        )
    synthesis_framing = f"Original claim: {claim_text}{thesis_block}"

    try:
        output = await invoke_llm(
            system_prompt=SYNTHESIZE_SYSTEM.format(
                current_date=date.today().isoformat(),
                synthesis_context=synthesis_context,
            ),
            user_prompt=SYNTHESIZE_USER.format(
                synthesis_framing=synthesis_framing,
                sub_verdicts_text=sub_verdicts_text,
            ),
            schema=SynthesizeOutput,
            semantic_validator=validate_synthesize,
            max_retries=2,
            temperature=0,
            activity_name="synthesize",
        )

        verdict = output.verdict
        confidence = output.confidence
        reasoning = output.reasoning

    except LLMInvocationError as e:
        log.warning(logger, MODULE, "invocation_failed",
                    "LLM invocation failed after retries",
                    error=str(e), attempts=e.attempts)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to synthesize verdict after {e.attempts} attempts"

    # Clean up reasoning text using LanguageTool
    reasoning = cleanup_text(reasoning)

    log.info(logger, MODULE, "done", "Verdict synthesized",
             claim=claim_text, verdict=verdict, confidence=confidence)

    return {
        "sub_claim": claim_text,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": [],
        "child_results": child_results,
        "reasoning_chain": [sub.get("reasoning", "") for sub in child_results],
    }
