"""Domain logic for verdict synthesis.

Combines child sub-verdicts into a final overall verdict.

The Temporal activity wrapper in verify_activities.py calls synthesize() here.
"""

import re
from datetime import date
from functools import partial

from src.llm import invoke_llm, LLMInvocationError, validate_synthesize
from src.prompts.verification import SYNTHESIZE_SYSTEM, SYNTHESIZE_USER, build_claim_date_line
from src.schemas.llm_outputs import SynthesizeOutput
from src.utils.logging import log, get_logger

MODULE = "synthesize"
logger = get_logger()


def _build_transcript_context(title: str | None, description: str) -> str:
    parts = []
    if title:
        parts.append(f"Source transcript: {title}")
    if description:
        parts.append(f"Description: {description}")
    return ("\n" + "\n".join(parts)) if parts else ""


async def synthesize(
    claim_text: str,
    child_results: list[dict],
    thesis_info: dict | None = None,
    claim_date: str | None = None,
    transcript_title: str | None = None,
    transcript_description: str = "",
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

    # Separate real verdicts from judge failures — only feed real ones to LLM
    usable_results = [r for r in child_results if not r.get("judge_failed", False)]
    failed_count = len(child_results) - len(usable_results)

    if failed_count:
        log.info(logger, MODULE, "judge_failures_filtered",
                 "Filtering judge-failed sub-claims from synthesis input",
                 claim=claim_text, usable=len(usable_results),
                 failed=failed_count)

    # If no usable results, skip LLM entirely
    if not usable_results:
        log.info(logger, MODULE, "all_failed",
                 "All sub-claims failed judge — returning unverifiable",
                 claim=claim_text, total=len(child_results))
        return {
            "sub_claim": claim_text,
            "verdict": "unverifiable",
            "confidence": 0.0,
            "reasoning": (
                f"All {len(child_results)} sub-claim judgments failed to produce "
                f"parseable results. Insufficient data for verdict synthesis."
            ),
            "evidence": [],
            "child_results": child_results,
            "reasoning_chain": [sub.get("reasoning", "") for sub in child_results],
            "citations": [],
            "synthesis_rubric": None,
        }

    # Format sub-verdicts for the LLM prompt (only usable results)
    sub_verdict_parts = []
    for i, sub in enumerate(usable_results, 1):
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

    if failed_count:
        sub_verdict_parts.append(
            f"Note: {failed_count} additional sub-claim(s) could not be evaluated "
            f"(judge parse failure). Base your verdict on the sub-claims above."
        )
    sub_verdicts_text = "\n\n".join(sub_verdict_parts)

    # Build unified evidence digest from judge-cited sources (usable only)
    evidence_digest = _build_evidence_digest(usable_results)
    evidence_digest_text = _format_evidence_digest(evidence_digest)

    log.info(logger, MODULE, "evidence_digest",
             "Built unified evidence digest from judge citations",
             claim=claim_text, digest_size=len(evidence_digest),
             usable_children=len(usable_results),
             failed_children=failed_count)

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
                claim_date_line=build_claim_date_line(claim_date),
                synthesis_context=synthesis_context,
            ),
            user_prompt=SYNTHESIZE_USER.format(
                synthesis_framing=synthesis_framing,
                sub_verdicts_text=sub_verdicts_text,
                evidence_digest=evidence_digest_text,
                transcript_context=_build_transcript_context(transcript_title, transcript_description),
            ),
            schema=SynthesizeOutput,
            semantic_validator=partial(
                validate_synthesize,
                evidence_digest_size=len(evidence_digest),
            ),
            max_retries=2,
            profile="reasoning",
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
                    "LLM invocation failed after retries, attempting deterministic fallback",
                    error=str(e), attempts=e.attempts,
                    parse_error=e.parse_error,
                    validation_error=e.validation_error,
                    raw_output_tail=e.raw_output[-500:] if e.raw_output else None)
        verdict, confidence, reasoning = _deterministic_fallback(
            child_results, claim_text, e.attempts,
        )

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


def _deterministic_fallback(
    child_results: list[dict],
    claim_text: str,
    llm_attempts: int,
) -> tuple[str, float, str]:
    """Compute verdict from sub-claim verdicts when LLM synthesis fails.

    Filters out judge-failed sub-claims (confidence=0 with parse failure),
    then uses the remaining verdicts to produce a weighted result.

    Returns (verdict, confidence, reasoning).
    """
    # Separate real verdicts from judge failures
    usable = [
        r for r in child_results
        if not r.get("judge_failed", False)
    ]
    failed_count = len(child_results) - len(usable)

    if not usable:
        log.info(logger, MODULE, "fallback_no_data",
                 "No usable sub-verdicts for deterministic fallback",
                 claim=claim_text, total_subs=len(child_results))
        return (
            "unverifiable", 0.0,
            f"Failed to synthesize verdict after {llm_attempts} attempts. "
            f"All {len(child_results)} sub-claim judgments also failed."
        )

    # Score verdicts numerically: true=1, mostly_true=0.75, mixed=0.5,
    # mostly_false=0.25, false=0, unverifiable=None (excluded)
    VERDICT_SCORES = {
        "true": 1.0,
        "mostly_true": 0.75,
        "mixed": 0.5,
        "mostly_false": 0.25,
        "false": 0.0,
    }
    SCORE_TO_VERDICT = [
        (0.875, "true"),
        (0.625, "mostly_true"),
        (0.375, "mixed"),
        (0.125, "mostly_false"),
        (0.0, "false"),
    ]

    scored = []
    for r in usable:
        v = r.get("verdict", "")
        if v in VERDICT_SCORES:
            scored.append((VERDICT_SCORES[v], r.get("confidence", 0.5)))

    if not scored:
        log.info(logger, MODULE, "fallback_no_scored",
                 "No scoreable sub-verdicts (all unverifiable)",
                 claim=claim_text, usable=len(usable))
        return (
            "unverifiable", 0.0,
            f"Failed to synthesize verdict after {llm_attempts} attempts. "
            f"{len(usable)} sub-claims were unverifiable."
        )

    # Confidence-weighted average
    total_weight = sum(conf for _, conf in scored)
    avg_score = sum(score * conf for score, conf in scored) / total_weight

    # Map back to verdict
    final_verdict = "false"
    for threshold, label in SCORE_TO_VERDICT:
        if avg_score >= threshold:
            final_verdict = label
            break

    # Reduce confidence: average of sub-confidences, penalized for failures
    avg_conf = total_weight / len(scored)
    penalty = 0.9 ** failed_count  # 10% penalty per failed sub-claim
    final_confidence = round(avg_conf * penalty, 2)

    # Build reasoning from sub-verdicts
    sub_lines = []
    for r in usable:
        sub_lines.append(
            f"- {r['sub_claim'][:100]}: {r['verdict']} ({r['confidence']})"
        )
    if failed_count:
        sub_lines.append(
            f"- {failed_count} sub-claim(s) had judge failures and were excluded"
        )

    reasoning = (
        f"Deterministic fallback (LLM synthesis failed after {llm_attempts} attempts). "
        f"Verdict computed from {len(scored)} sub-claim verdicts "
        f"(weighted average score: {avg_score:.2f}).\n"
        + "\n".join(sub_lines)
    )

    log.info(logger, MODULE, "fallback_computed",
             "Deterministic fallback verdict computed",
             claim=claim_text, verdict=final_verdict,
             confidence=final_confidence, avg_score=avg_score,
             usable=len(scored), failed=failed_count)

    return final_verdict, final_confidence, reasoning


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
