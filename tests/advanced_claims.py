"""Advanced test suite: 20 claims testing reasoning depth and nuance.

Focuses on areas NOT covered by regression_claims.py or stress_claims.py:
methodology disputes, correlation/causation, misleading aggregation,
historical revisionism, cross-domain reasoning, and claims where the
"obvious" answer is wrong.

Run with: python -m tests.advanced_claims [--api-url URL]

Uses the same infrastructure as regression_claims.py.
"""

import argparse
import asyncio
import json
import time

# ---------------------------------------------------------------------------
# Advanced claims — reasoning depth, nuance, and counterintuitive truths
# ---------------------------------------------------------------------------

CLAIMS = [
    # ── Misleading aggregation / Simpson's paradox ────────────────────────
    {
        "text": (
            "UC Berkeley's graduate admissions were biased against women, "
            "with men admitted at a significantly higher rate overall"
        ),
        "tests": [
            "judge:simpsons_paradox",
            "judge:statistical_framing",
            "research:academic_source",
        ],
        "notes": (
            "Classic Simpson's paradox case. Overall admission rates favored "
            "men, but department-by-department analysis showed slight bias "
            "TOWARD women. Women applied more to competitive departments. "
            "Tests whether pipeline finds the Bickel 1975 study and reasons "
            "about aggregation fallacies."
        ),
    },
    {
        "text": (
            "Exposed workers at a nuclear weapons facility had lower cancer "
            "rates than the general population"
        ),
        "tests": [
            "judge:healthy_worker_effect",
            "judge:statistical_framing",
            "judge:counterintuitive",
        ],
        "notes": (
            "Healthy worker effect — employed people are healthier than the "
            "general population (which includes elderly, disabled, etc). "
            "The statistic is TRUE but the implied conclusion (radiation is "
            "safe) is fallacious. Tests whether model identifies the bias."
        ),
    },

    # ── Correlation ≠ causation ───────────────────────────────────────────
    {
        "text": (
            "Countries with higher chocolate consumption per capita have "
            "won more Nobel Prizes"
        ),
        "tests": [
            "judge:correlation_vs_causation",
            "judge:statistical_framing",
            "research:academic_source",
        ],
        "notes": (
            "Based on a real 2012 NEJM paper by Messerli. The correlation "
            "IS real (r=0.79), but the causal implication is absurd — both "
            "correlate with national wealth and research funding. Tests "
            "whether pipeline confirms the correlation while flagging "
            "the causal fallacy."
        ),
    },
    {
        "text": (
            "States that legalized recreational marijuana saw an increase "
            "in traffic fatalities"
        ),
        "tests": [
            "judge:correlation_vs_causation",
            "judge:cherry_picked_timeframe",
            "judge:confounding_variables",
        ],
        "notes": (
            "Mixed evidence — some studies show increases, others don't "
            "after controlling for pre-existing trends and other factors "
            "(rideshare adoption, speed limit changes). Tests whether "
            "pipeline handles genuinely contested empirical claims."
        ),
    },

    # ── Historical claims with modern reinterpretation ────────────────────
    {
        "text": (
            "The Allied bombing of Dresden in World War II killed over "
            "200,000 civilians"
        ),
        "tests": [
            "judge:historical_revisionism",
            "judge:quantitative",
            "judge:source_verification",
        ],
        "notes": (
            "Wildly inflated number originating from Nazi propaganda and "
            "amplified by David Irving. Modern historical consensus (2010 "
            "Dresden commission) puts the figure at 22,700-25,000. Tests "
            "whether pipeline finds authoritative sources over viral numbers."
        ),
    },
    {
        "text": (
            "The Tuskegee syphilis study deliberately infected Black men "
            "with syphilis"
        ),
        "tests": [
            "judge:common_misconception",
            "judge:historical_accuracy",
        ],
        "notes": (
            "Common misconception. The study did NOT infect participants — "
            "it tracked men who ALREADY had syphilis and withheld treatment. "
            "Both are horrific, but the distinction matters for accuracy. "
            "Tests whether pipeline corrects widely-held false details about "
            "real atrocities."
        ),
    },

    # ── Scientific claims requiring methodology reasoning ─────────────────
    {
        "text": (
            "Organic food has been proven to be more nutritious than "
            "conventionally grown food"
        ),
        "tests": [
            "judge:scientific_consensus",
            "judge:methodology",
            "decompose:operationalize_vague",
        ],
        "notes": (
            "Major meta-analyses (Stanford 2012, Baranski 2014) reach "
            "different conclusions depending on methodology and which "
            "nutrients are measured. No clear consensus on 'more nutritious' "
            "as a blanket claim. Tests nuanced scientific reasoning."
        ),
    },
    {
        "text": (
            "The average human swallows eight spiders per year while sleeping"
        ),
        "tests": [
            "judge:urban_legend",
            "judge:source_verification",
        ],
        "notes": (
            "Fabricated statistic, often attributed to a 1993 magazine "
            "column by Lisa Holst — but that column itself may be "
            "fabricated. Spider behavior makes this implausible. Tests "
            "whether pipeline traces viral claims to their (non)source."
        ),
    },

    # ── Economic claims with hidden complexity ────────────────────────────
    {
        "text": (
            "The US national debt has increased under every president "
            "since Jimmy Carter"
        ),
        "tests": [
            "judge:exhaustive_check",
            "judge:quantitative",
            "decompose:universal_quantifier",
        ],
        "notes": (
            "Actually true in nominal terms (Reagan through Biden). Clinton "
            "ran surpluses but the DEBT still increased due to Social Security "
            "trust fund accounting. Tests whether pipeline distinguishes "
            "deficit vs debt and checks each presidency."
        ),
    },
    {
        "text": (
            "Exposed workers at a nuclear weapons facility had lower cancer "
            "rates than the general population, proving that low-level "
            "radiation exposure is safe"
        ),
        "tests": [
            "decompose:causal",
            "judge:logical_fallacy",
            "judge:healthy_worker_effect",
        ],
        "notes": (
            "Variant of the healthy worker claim but with an explicit "
            "causal conclusion. The statistical claim may be true but "
            "the causal conclusion is a non-sequitur. Tests whether "
            "decomposer separates the statistic from the conclusion."
        ),
    },

    # ── Claims where the "obvious" answer is wrong ────────────────────────
    {
        "text": (
            "More people are alive today than have ever died in all of "
            "human history"
        ),
        "tests": [
            "judge:mathematical_reasoning",
            "judge:counterintuitive",
        ],
        "notes": (
            "Sounds plausible given exponential growth, but is FALSE. "
            "PRB estimates ~109 billion humans have ever lived; ~8 billion "
            "are alive today (~7%). Tests whether pipeline does the math "
            "rather than accepting the intuitive-sounding claim."
        ),
    },
    {
        "text": (
            "Lightning never strikes the same place twice"
        ),
        "tests": [
            "judge:common_misconception",
            "judge:scientific_consensus",
        ],
        "notes": (
            "Obviously false — tall structures get struck repeatedly "
            "(Empire State Building: ~20-25 times/year). But tests whether "
            "the pipeline handles folk wisdom debunking cleanly and finds "
            "concrete counterexamples."
        ),
    },

    # ── Geopolitical claims requiring nuance ──────────────────────────────
    {
        "text": (
            "Switzerland has been neutral in every military conflict "
            "since 1815"
        ),
        "tests": [
            "decompose:universal_quantifier",
            "judge:exhaustive_check",
            "judge:definition_dependent",
        ],
        "notes": (
            "Mostly true but with asterisks — Switzerland's sanctions "
            "against Russia in 2022 challenged its neutrality, and its "
            "WWII-era dealings with Nazi Germany are debated. 'Neutral' "
            "and 'military conflict' need operationalization. Tests how "
            "pipeline handles a claim that's 'true with caveats.'"
        ),
    },
    {
        "text": (
            "India will surpass China as the world's most populous "
            "country by 2030"
        ),
        "tests": [
            "judge:temporal_prediction",
            "judge:quantitative",
        ],
        "notes": (
            "India already surpassed China in 2023 according to UN "
            "estimates. The claim's prediction is already fulfilled. "
            "Tests whether pipeline recognizes a future prediction "
            "that has already come true and adjusts accordingly."
        ),
    },

    # ── Claims mixing fact and value judgment ──────────────────────────────
    {
        "text": (
            "The Nordic countries consistently rank as the happiest in "
            "the world despite having some of the highest tax rates"
        ),
        "tests": [
            "decompose:concession_structure",
            "judge:quantitative",
            "judge:definition_dependent",
        ],
        "notes": (
            "Both parts are well-supported (World Happiness Report + OECD "
            "tax data). The 'despite' implies tension but both are true. "
            "Tests whether pipeline handles a compound claim where both "
            "halves verify AND the connecting word implies contrast."
        ),
    },
    {
        "text": (
            "Cuba has a higher life expectancy than the United States "
            "despite spending a fraction of the amount on healthcare"
        ),
        "tests": [
            "judge:quantitative",
            "judge:data_quality",
            "judge:concession_structure",
        ],
        "notes": (
            "Cuba's reported life expectancy (~78-79 years) is very close "
            "to the US (~77-78 years). Whether Cuba is actually higher is "
            "debatable due to data reliability concerns. The spending "
            "difference is real. Tests data quality skepticism."
        ),
    },

    # ── Multi-step reasoning chains ───────────────────────────────────────
    {
        "text": (
            "Exposed workers at nuclear facilities are healthier than "
            "average, which explains why France's nuclear-heavy energy "
            "mix results in lower cancer rates than Germany"
        ),
        "tests": [
            "decompose:multi_step_reasoning",
            "judge:logical_fallacy",
            "judge:confounding_variables",
        ],
        "notes": (
            "Multi-step fallacy: premise 1 (healthy worker effect) may be "
            "true, premise 2 (France lower cancer than Germany) may be true, "
            "but the causal chain connecting them is invalid. Tests whether "
            "pipeline identifies fallacious reasoning chains even when "
            "individual premises are true."
        ),
    },
    {
        "text": (
            "The gender pay gap in the United States is 84 cents on the "
            "dollar, meaning women are paid less than men for the exact "
            "same work"
        ),
        "tests": [
            "decompose:multi_part",
            "judge:statistical_framing",
            "judge:definition_dependent",
        ],
        "notes": (
            "The 84-cent figure (BLS median earnings ratio) is real, but "
            "it compares ALL full-time workers, not same job/experience. "
            "The 'exact same work' interpretation is a common conflation. "
            "Adjusted gap is ~95-98 cents. Tests whether pipeline separates "
            "the statistic from its misinterpretation."
        ),
    },

    # ── Very specific verifiable claims ───────────────────────────────────
    {
        "text": (
            "There have been more mass shootings in the United States "
            "in 2025 than days in the year"
        ),
        "tests": [
            "judge:quantitative",
            "judge:definition_dependent",
            "judge:evidence_freshness",
        ],
        "notes": (
            "Depends entirely on the definition of 'mass shooting' — Gun "
            "Violence Archive (4+ shot) says yes most years; FBI definition "
            "(4+ killed in public) says far fewer. Tests whether pipeline "
            "surfaces the definitional ambiguity."
        ),
    },
    {
        "text": (
            "Exposed workers at a nuclear weapons facility had lower cancer "
            "rates than the general population, and this finding has been "
            "replicated across multiple studies in different countries"
        ),
        "tests": [
            "judge:healthy_worker_effect",
            "judge:scientific_consensus",
            "judge:replication",
        ],
        "notes": (
            "The healthy worker effect IS well-documented across multiple "
            "studies internationally. Both the statistical finding and the "
            "replication claim are true. But the IMPLIED conclusion (that "
            "radiation is safe) remains fallacious. Tests whether pipeline "
            "can confirm facts while noting the interpretive gap."
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
    """Run the advanced test suite."""
    claims = CLAIMS[:count] if count else CLAIMS
    n = len(claims)

    if timeout is None:
        timeout = max(2400, n * 35 * 60)

    print(f"\nSpin Cycle ADVANCED Test Suite — {n} of {len(CLAIMS)} claims")
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
    output_file = str(results_dir / f"advanced_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
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
        description="Run Spin Cycle advanced test suite"
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
        print(f"\nAdvanced Suite: {len(CLAIMS)} claims\n")
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
