"""Stress test suite: 20 claims designed to break the pipeline.

Each claim targets a specific weakness — temporal anchoring, negation handling,
proving negatives, stale viral claims, false attribution, statistical sleight
of hand, jurisdiction confusion, and claims requiring mathematical reasoning.

Run with: python -m tests.stress_claims [--api-url URL]

Uses the same infrastructure as regression_claims.py.
"""

import argparse
import asyncio
import json
import sys
import time

# ---------------------------------------------------------------------------
# Stress claims — designed to expose pipeline edge cases
# ---------------------------------------------------------------------------

CLAIMS = [
    # ── Temporal ambiguity: truth depends on WHEN ────────────────────────
    {
        "text": (
            "NASA's budget has been increasing every year for the past two decades"
        ),
        "tests": [
            "decompose:temporal_anchor",
            "judge:cherry_picked_timeframe",
            "judge:quantitative",
        ],
        "notes": (
            "True for some year-windows, false for others (e.g. sequestration "
            "years). Tests whether the pipeline checks the FULL range rather "
            "than cherry-picking a favorable subset."
        ),
    },
    {
        "text": (
            "Brazil has the highest deforestation rate in the world"
        ),
        "tests": [
            "decompose:temporal_anchor",
            "decompose:superlative",
            "judge:definition_dependent",
        ],
        "notes": (
            "Was true for decades, but Indonesia and others have overtaken "
            "in some years. Tests whether pipeline anchors to present data "
            "and operationalizes 'deforestation rate' (absolute vs relative)."
        ),
    },

    # ── Negation & double negation ───────────────────────────────────────
    {
        "text": (
            "No country in the G7 other than the United States lacks "
            "a universal healthcare system"
        ),
        "tests": [
            "decompose:negation",
            "decompose:exception_clause",
            "judge:exhaustive_check",
        ],
        "notes": (
            "Double negation with exception ('no country... other than... lacks'). "
            "Semantically: all G7 countries except the US have universal healthcare. "
            "Tests whether normalizer/decomposer correctly resolves the logic."
        ),
    },
    {
        "text": (
            "It is not the case that no evidence exists linking microplastics "
            "to adverse health outcomes in humans"
        ),
        "tests": [
            "normalization:double_negation",
            "decompose:negation",
            "judge:scientific_consensus",
        ],
        "notes": (
            "Triple negation: not(no evidence exists) = evidence does exist. "
            "Tests whether normalizer simplifies this to the positive assertion. "
            "The underlying claim (evidence linking microplastics to health "
            "outcomes) is actually true — emerging research supports it."
        ),
    },

    # ── Compound claims with mixed truth values ──────────────────────────
    {
        "text": (
            "Finland has the best education system in the world and "
            "the highest suicide rate in Europe"
        ),
        "tests": [
            "decompose:parallel_comparison",
            "decompose:superlative",
            "synthesize:mixed_verdicts",
        ],
        "notes": (
            "Two independent superlatives joined by 'and'. Finland ranks "
            "high in education (but 'best' is debatable — PISA rankings "
            "fluctuate). Finland's suicide rate WAS among the highest in "
            "Europe but has dropped significantly. Tests mixed-verdict synthesis."
        ),
    },
    {
        "text": (
            "Canada has more coastline than any other country and is the "
            "second largest country by land area"
        ),
        "tests": [
            "decompose:parallel_comparison",
            "decompose:superlative",
            "judge:quantitative",
        ],
        "notes": (
            "Both parts are actually true (Canada: 202,080 km coastline, "
            "second to Russia by area). Tests whether pipeline handles a "
            "claim where BOTH halves verify cleanly — no false-positive "
            "skepticism."
        ),
    },

    # ── Claims about absence (proving a negative) ────────────────────────
    {
        "text": (
            "There is no scientific evidence that fluoride in drinking "
            "water at recommended levels causes cancer"
        ),
        "tests": [
            "decompose:absence_claim",
            "judge:proving_negative",
            "judge:scientific_consensus",
        ],
        "notes": (
            "Proving a negative. The pipeline must determine whether the "
            "ABSENCE of evidence is itself well-supported by systematic "
            "reviews, rather than just failing to find evidence. This is "
            "the scientific consensus position (WHO, CDC)."
        ),
    },
    {
        "text": (
            "No sitting US president has ever been convicted of a crime "
            "while in office"
        ),
        "tests": [
            "decompose:absence_claim",
            "decompose:negation",
            "judge:exhaustive_check",
        ],
        "notes": (
            "Universal negative requiring exhaustive historical check. "
            "Tests whether decomposer correctly treats this as a single "
            "existential claim rather than splitting per president."
        ),
    },

    # ── Stale viral claims ───────────────────────────────────────────────
    {
        "text": (
            "Sweden never implemented any lockdown measures during the "
            "COVID-19 pandemic"
        ),
        "tests": [
            "normalization:absolute_language",
            "decompose:temporal",
            "judge:oversimplification",
        ],
        "notes": (
            "Viral claim that oversimplifies. Sweden avoided mandatory "
            "lockdowns EARLY but did implement restrictions later (ban "
            "on gatherings >8, high school closures, etc). 'Never' and "
            "'any' make this falsifiable. Tests absolute language handling."
        ),
    },
    {
        "text": (
            "The Great Wall of China is the only man-made structure "
            "visible from space"
        ),
        "tests": [
            "decompose:superlative",
            "judge:common_misconception",
        ],
        "notes": (
            "Classic misconception. Not visible from low Earth orbit "
            "with the naked eye (confirmed by astronauts including Yang "
            "Liwei). Many other structures ARE visible. Tests whether "
            "pipeline correctly debunks widely-believed false claims."
        ),
    },

    # ── Attribution to wrong source ──────────────────────────────────────
    {
        "text": (
            "According to a recent Stanford study, remote workers are "
            "70% more productive than office workers"
        ),
        "tests": [
            "decompose:attribution",
            "decompose:evidentiality_marker",
            "judge:source_verification",
        ],
        "notes": (
            "Fabricated attribution. Stanford's Bloom study found ~13% "
            "productivity increase, not 70%. Tests whether the pipeline "
            "verifies the SPECIFIC claim against the ACTUAL source rather "
            "than just confirming Stanford studied remote work."
        ),
    },
    {
        "text": (
            "Winston Churchill said that democracy is the worst form of "
            "government except for all the others"
        ),
        "tests": [
            "decompose:attribution",
            "judge:source_verification",
        ],
        "notes": (
            "Commonly attributed quote that Churchill DID say (House of "
            "Commons, 1947), though he noted it was already a saying. "
            "Tests whether pipeline can verify historical attribution "
            "accurately rather than being overly skeptical of all quotes."
        ),
    },

    # ── Statistical manipulation / misleading framing ────────────────────
    {
        "text": (
            "You are more likely to be killed by a coconut falling on "
            "your head than by a shark attack"
        ),
        "tests": [
            "decompose:comparative",
            "judge:statistical_framing",
            "judge:quantitative",
        ],
        "notes": (
            "Viral statistical claim. The '150 coconut deaths/year' figure "
            "traces to a single unverified source and is likely false. "
            "Shark deaths are ~5-10/year. Tests whether pipeline traces "
            "statistics to primary sources rather than repeating viral data."
        ),
    },
    {
        "text": (
            "The United States spends more on its military than the next "
            "ten countries combined"
        ),
        "tests": [
            "decompose:comparative",
            "judge:quantitative",
            "judge:cherry_picked_timeframe",
        ],
        "notes": (
            "Often cited but the exact number of countries varies by year. "
            "As of recent data it's closer to 'next 7-9 combined' depending "
            "on the source. Tests precise quantitative verification vs "
            "accepting a commonly-repeated approximate claim."
        ),
    },

    # ── Jurisdiction / scope confusion ───────────────────────────────────
    {
        "text": (
            "It is illegal to collect rainwater on your own property"
        ),
        "tests": [
            "normalization:missing_jurisdiction",
            "decompose:scope_ambiguity",
            "judge:jurisdiction_dependent",
        ],
        "notes": (
            "No jurisdiction specified. True in some places (e.g. Colorado "
            "had restrictions until 2016), false in most US states and most "
            "countries. Tests whether the pipeline flags scope ambiguity "
            "rather than giving a blanket true/false."
        ),
    },
    {
        "text": (
            "The legal drinking age is 21 and has been since the 1980s"
        ),
        "tests": [
            "normalization:missing_jurisdiction",
            "decompose:scope_ambiguity",
            "decompose:temporal",
        ],
        "notes": (
            "True for the US (National Minimum Drinking Age Act, 1984) but "
            "not globally. Tests whether pipeline identifies the implicit "
            "US jurisdiction and evaluates accordingly, or flags the "
            "missing scope."
        ),
    },

    # ── Claims requiring math / logic ────────────────────────────────────
    {
        "text": (
            "If Jeff Bezos gave every person on Earth $1 million, he would "
            "still have money left over"
        ),
        "tests": [
            "decompose:conditional_counterfactual",
            "judge:mathematical_reasoning",
        ],
        "notes": (
            "Viral claim that sounds plausible but fails basic math. "
            "8 billion × $1M = $8 quadrillion. Bezos has ~$200B. "
            "He'd need 40,000× his wealth. Tests whether pipeline does "
            "the arithmetic rather than pattern-matching to 'billionaires "
            "have lots of money'."
        ),
    },
    {
        "text": (
            "The global population has doubled since 1970"
        ),
        "tests": [
            "judge:quantitative",
            "judge:mathematical_reasoning",
        ],
        "notes": (
            "1970 population: ~3.7B. Current: ~8.1B. That's 2.19×, so "
            "'doubled' is approximately true (slightly more than doubled). "
            "Tests whether pipeline handles 'approximately true' numeric "
            "claims without being pedantically false."
        ),
    },

    # ── Very recent / rapidly evolving events ────────────────────────────
    {
        "text": (
            "The European Central Bank has raised interest rates at every "
            "meeting in 2025"
        ),
        "tests": [
            "decompose:universal_quantifier",
            "decompose:temporal",
            "judge:evidence_freshness",
        ],
        "notes": (
            "Requires very recent data (2025 ECB decisions). The ECB "
            "actually cut rates in 2025. Tests whether the pipeline's "
            "search tools can find current-year monetary policy data "
            "and whether 'every meeting' is checked exhaustively."
        ),
    },
    {
        "text": (
            "Australia's Great Barrier Reef has lost more than half its "
            "coral cover since 1995"
        ),
        "tests": [
            "decompose:temporal",
            "judge:quantitative",
            "judge:scientific_consensus",
        ],
        "notes": (
            "Based on a real 2022 ARC study finding ~50% decline between "
            "1995-2022. But 2023-2024 surveys showed some recovery. Tests "
            "whether pipeline uses the most current data and contextualizes "
            "the trend rather than relying on a single study."
        ),
    },
]


# ---------------------------------------------------------------------------
# Reuse infrastructure from regression_claims
# ---------------------------------------------------------------------------

from tests.regression_claims import (
    _verdict_color,
    _print_results,
    _http_post,
    _http_get,
    submit_batch,
    poll_until_done,
)


async def run_suite(api_url: str, count: int | None = None, timeout: int | None = None):
    """Run the stress test suite."""
    claims = CLAIMS[:count] if count else CLAIMS
    n = len(claims)

    if timeout is None:
        timeout = max(2400, n * 35 * 60)

    print(f"\nSpin Cycle STRESS Test Suite — {n} of {len(CLAIMS)} claims")
    print(f"API: {api_url}")
    print(f"Timeout: {timeout}s ({timeout // 60} min)")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    submitted = await submit_batch(api_url, claims)
    claim_ids = [s["id"] for s in submitted]

    id_to_claim = {}
    for s, c in zip(submitted, claims):
        id_to_claim[s["id"]] = c

    print(f"\nPolling for results (up to {timeout // 60} min)...\n")
    raw_results = await poll_until_done(api_url, claim_ids, timeout=timeout)

    results = []
    for cid in claim_ids:
        claim_meta = id_to_claim[cid]
        if cid in raw_results:
            data = raw_results[cid]
            verdict = data.get("verdict", "???")
            confidence = data.get("confidence", 0)
        else:
            verdict = "???"
            confidence = 0

        results.append({
            "claim": claim_meta,
            "id": cid,
            "verdict": verdict,
            "confidence": confidence,
            "raw": raw_results.get(cid),
        })

    _print_results(results)

    from pathlib import Path
    results_dir = Path("tests/regression_results")
    results_dir.mkdir(exist_ok=True)
    output_file = str(results_dir / f"stress_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    serializable = []
    for r in results:
        serializable.append({
            "text": r["claim"]["text"],
            "tests": r["claim"]["tests"],
            "notes": r["claim"]["notes"],
            "verdict": r["verdict"],
            "confidence": r["confidence"],
        })
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Full results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Spin Cycle stress test suite"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:4500",
        help="API base URL (default: http://localhost:4500)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Just list claims and what they test, don't submit",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Run only the first N claims (default: all)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Poll timeout in seconds (default: count × 35 min)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\nStress Suite: {len(CLAIMS)} claims\n")
        for i, c in enumerate(CLAIMS, 1):
            text = c["text"]
            if len(text) > 90:
                text = text[:87] + "..."
            print(f"{i:2d}. {text}")
            print(f"    Tests: {', '.join(c['tests'])}")
            print(f"    Notes: {c['notes'][:100]}...")
            print()
        return

    asyncio.run(run_suite(args.api_url, count=args.count, timeout=args.timeout))


if __name__ == "__main__":
    main()
