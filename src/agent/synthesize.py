"""Domain logic for verdict synthesis.

Combines child sub-verdicts into a final overall verdict.

The Temporal activity wrapper in verify_activities.py calls synthesize() here.
"""

import re
from datetime import date

from src.llm import invoke_llm, LLMInvocationError, validate_synthesize
from src.prompts.verification import SYNTHESIZE_SYSTEM, SYNTHESIZE_USER
from src.schemas.llm_outputs import SynthesizeOutput
from src.utils.logging import log, get_logger

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
        # Add key cited sources so synthesizer can reference them
        if sub.get("citations"):
            cited = sub["citations"][:5]
            source_lines = []
            for c in cited:
                label = c.get("title") or c.get("domain", "?")
                source_lines.append(f"      - {label} ({c.get('url', '')})")
            part += "\n    Key sources:\n" + "\n".join(source_lines)
        sub_verdict_parts.append(part)
    sub_verdicts_text = "\n\n".join(sub_verdict_parts)

    # Build unified evidence digest from judge-cited sources
    evidence_digest = _build_evidence_digest(child_results)
    evidence_digest_text = _format_evidence_digest(evidence_digest)

    log.info(logger, MODULE, "evidence_digest",
             "Built unified evidence digest from judge citations",
             claim=claim_text, digest_size=len(evidence_digest),
             child_count=len(child_results))

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

    synthesis_rubric = None
    try:
        output = await invoke_llm(
            system_prompt=SYNTHESIZE_SYSTEM.format(
                current_date=date.today().isoformat(),
                synthesis_context=synthesis_context,
            ),
            user_prompt=SYNTHESIZE_USER.format(
                synthesis_framing=synthesis_framing,
                sub_verdicts_text=sub_verdicts_text,
                evidence_digest=evidence_digest_text,
            ),
            schema=SynthesizeOutput,
            semantic_validator=validate_synthesize,
            max_retries=2,
            temperature=0,
            max_tokens=16384,
            activity_name="synthesize",
        )

        verdict = output.verdict
        confidence = output.confidence
        reasoning = output.reasoning

        # Log rubric steps — INFO level for key decisions, DEBUG for details
        core_indices = [
            w.subclaim_index for w in output.subclaim_weights
            if w.role == "core_assertion"
        ]
        supporting_indices = [
            w.subclaim_index for w in output.subclaim_weights
            if w.role == "supporting_detail"
        ]
        background_indices = [
            w.subclaim_index for w in output.subclaim_weights
            if w.role == "background_context"
        ]
        log.info(logger, MODULE, "rubric_summary",
                 "Synthesize rubric completed",
                 claim=claim_text,
                 thesis=output.thesis_restatement[:150],
                 core_assertions=core_indices,
                 supporting_details=supporting_indices,
                 background_context=background_indices,
                 thesis_survives=output.thesis_survives,
                 verdict=output.verdict,
                 confidence=output.confidence)
        log.debug(logger, MODULE, "rubric_classification_detail",
                  "Subclaim classification reasoning",
                  claim=claim_text,
                  weights=[
                      {"idx": w.subclaim_index, "role": w.role,
                       "reason": w.brief_reason}
                      for w in output.subclaim_weights
                  ])

        synthesis_rubric = {
            "thesis_restatement": output.thesis_restatement,
            "subclaim_weights": [w.model_dump() for w in output.subclaim_weights],
            "thesis_survives": output.thesis_survives,
        }

        # Programmatic consistency check (permissive — log only)
        consistency_warnings = _validate_synthesize_consistency(output)
        for warning in consistency_warnings:
            log.warning(logger, MODULE, "rubric_inconsistency",
                        warning, claim=claim_text,
                        thesis_survives=output.thesis_survives,
                        verdict=output.verdict)

    except LLMInvocationError as e:
        log.warning(logger, MODULE, "invocation_failed",
                    "LLM invocation failed after retries",
                    error=str(e), attempts=e.attempts,
                    parse_error=e.parse_error,
                    validation_error=e.validation_error,
                    raw_output_tail=e.raw_output[-500:] if e.raw_output else None)
        verdict = "unverifiable"
        confidence = 0.0
        reasoning = f"Failed to synthesize verdict after {e.attempts} attempts"

    log.info(logger, MODULE, "done", "Verdict synthesized",
             claim=claim_text, verdict=verdict, confidence=confidence)

    # Extract [N] citations from reasoning, mapped to evidence digest
    citations = _extract_synthesize_citations(reasoning, evidence_digest)

    log.info(logger, MODULE, "citations_extracted",
             "Extracted citations from synthesis reasoning",
             claim=claim_text, citation_count=len(citations),
             digest_size=len(evidence_digest),
             cited_indices=[c["index"] for c in citations])

    return {
        "sub_claim": claim_text,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": [],
        "child_results": child_results,
        "reasoning_chain": [sub.get("reasoning", "") for sub in child_results],
        "citations": citations,
        "synthesis_rubric": synthesis_rubric,
    }


def _build_evidence_digest(child_results: list[dict]) -> list[dict]:
    """Build unified evidence list from judge-cited sources across all sub-claims.

    Only includes sources the judges actually cited in [N] notation —
    typically 3-5 per sub-claim, ~10-20 total after dedup.
    """
    digest = []
    seen_urls: set[str] = set()

    for child in child_results:
        # Build URL → metadata lookup for this sub-claim's evidence
        ev_by_url = {
            ev.get("source_url"): ev
            for ev in child.get("evidence", [])
            if ev.get("source_url")
        }

        for c in child.get("citations", []):
            url = c.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            meta = ev_by_url.get(url, {})
            digest.append({
                "source_url": url,
                "title": c.get("title") or meta.get("title"),
                "domain": c.get("domain") or meta.get("domain"),
                "tier": meta.get("tier"),
                "assessment": meta.get("assessment"),
                "key_point": meta.get("key_point"),
                "bias": meta.get("bias"),
                "factual": meta.get("factual"),
            })

    return digest


def _format_evidence_digest(digest: list[dict]) -> str:
    """Format evidence digest as numbered reference list for the prompt."""
    if not digest:
        return ""

    lines = ["=== EVIDENCE SOURCES ===",
             "Cite these using [N] in your reasoning.\n"]
    for i, ev in enumerate(digest, 1):
        domain = ev.get("domain", "?")
        title = ev.get("title", "")
        tier = ev.get("tier", "")

        header = f"[{i}] {domain}"
        if title:
            header += f' — "{title}"'
        if tier:
            header += f" — {tier}"
        lines.append(header)

        # Add key finding if the judge assessed this source
        if ev.get("key_point"):
            assessment = ev.get("assessment", "")
            finding = f"    Finding: {ev['key_point']}"
            if assessment:
                finding += f" ({assessment})"
            lines.append(finding)

    return "\n".join(lines)


def _extract_synthesize_citations(
    reasoning: str,
    evidence_digest: list[dict],
) -> list[dict]:
    """Extract [N] citation indices from reasoning and map to evidence digest."""
    indices = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', reasoning)))
    citations = []
    for idx in indices:
        if 1 <= idx <= len(evidence_digest):
            ev = evidence_digest[idx - 1]
            citations.append({
                "index": idx,
                "url": ev.get("source_url"),
                "title": ev.get("title"),
                "domain": ev.get("domain"),
            })
    return citations


def _validate_synthesize_consistency(output: SynthesizeOutput) -> list[str]:
    """Check for contradictions between rubric steps (permissive — log only)."""
    warnings = []

    if output.thesis_survives and output.verdict in ("mostly_false", "false"):
        warnings.append(
            "thesis_survives=True but verdict is negative. "
            "If the thesis holds, verdict should be true/mostly_true."
        )

    if not output.thesis_survives and output.verdict in ("true", "mostly_true"):
        warnings.append(
            "thesis_survives=False but verdict is positive. "
            "If the thesis fails, verdict should be mostly_false/false."
        )

    return warnings
