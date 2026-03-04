"""Regression test suite: 20 categorized claims for end-to-end verification.

Each claim is annotated with what pipeline stages/scenarios it exercises.
Run with: python -m tests.regression_claims [--api-url URL]

Submits all claims via the batch API, polls until all are verified,
then prints a results table with verdicts + what each claim tests.
"""

import argparse
import asyncio
import json
import sys
import time

# ---------------------------------------------------------------------------
# Test claims — each tagged with the pipeline scenarios it exercises
# ---------------------------------------------------------------------------

CLAIMS = [
    # ── Normalization: bias & loaded language ──────────────────────────
    {
        "text": (
            "Biden's radical open border policy let in millions of illegal "
            "aliens who are destroying American cities"
        ),
        "tests": [
            "normalization:right_wing_bias",
            "normalization:loaded_language",
            "normalization:operationalize_vague",
            "decompose:causal",
        ],
        "notes": (
            "Heavy right-wing framing ('radical', 'illegal aliens', "
            "'destroying'). Normalizer should neutralize to verifiable "
            "claims about border policy changes and immigration numbers."
        ),
    },
    {
        "text": (
            "Greedy pharmaceutical companies are price-gouging life-saving "
            "medications while raking in record profits"
        ),
        "tests": [
            "normalization:left_wing_bias",
            "normalization:loaded_language",
            "normalization:opinion_separation",
        ],
        "notes": (
            "Left-wing framing ('greedy', 'price-gouging', 'raking in'). "
            "Normalizer should strip opinion, decomposer should separate "
            "the profit claim from the pricing claim."
        ),
    },
    {
        "text": (
            "America's crumbling infrastructure is the worst among developed "
            "nations and has been criminally neglected by both parties for decades"
        ),
        "tests": [
            "normalization:loaded_language",
            "normalization:opinion_separation",
            "normalization:operationalize_vague",
            "decompose:comparative",
            "decompose:temporal",
            "judge:definition_dependent",
        ],
        "notes": (
            "'Crumbling', 'criminally neglected' are loaded. 'Worst among "
            "developed nations' needs operationalization (what metric? which "
            "nations count?). 'Decades' is vague temporal. Mixed opinion + fact."
        ),
    },

    # ── Decomposition patterns ─────────────────────────────────────────
    {
        "text": (
            "Texas's power grid failed during Winter Storm Uri because they "
            "refused to connect to the federal grid"
        ),
        "tests": [
            "decompose:causal",
            "judge:oversimplification",
        ],
        "notes": (
            "Causal claim with oversimplified mechanism. The grid DID fail, "
            "Texas IS independent from federal interconnections, but 'refused "
            "to connect' oversimplifies decades of regulatory history. Tests "
            "whether judge catches the nuance."
        ),
    },
    {
        "text": (
            "ExxonMobil's own scientists confirmed climate change was real "
            "in the 1970s but the company spent decades funding climate denial"
        ),
        "tests": [
            "decompose:temporal_sequence",
            "judge:interested_parties",
            "research:wikidata_expansion",
        ],
        "notes": (
            "Two temporal claims linked by contrast ('but'). ExxonMobil is "
            "an interested party in evidence about itself. Wikidata should "
            "expand to subsidiaries, executives."
        ),
    },
    {
        "text": (
            "Japan has the oldest population in the world and spends more "
            "per capita on elderly care than any European country"
        ),
        "tests": [
            "decompose:parallel_comparison",
            "decompose:superlative",
            "judge:quantitative",
        ],
        "notes": (
            "Two independent superlative claims joined by 'and'. Each needs "
            "exhaustive comparison. The first is likely true, the second is "
            "harder to verify and may be false."
        ),
    },
    {
        "text": (
            "Every single Republican voted against capping insulin prices "
            "at $35 in the Inflation Reduction Act"
        ),
        "tests": [
            "decompose:universal_quantifier",
            "research:legiscan",
            "judge:quantitative",
        ],
        "notes": (
            "Universal quantifier ('every single') — one counterexample "
            "makes it false. LegiScan should find roll call votes. The "
            "Senate vote was party-line but 'every single' across both "
            "chambers may not hold."
        ),
    },
    {
        "text": (
            "Since China stopped manipulating its currency in 2015, the "
            "trade deficit with the US has only gotten worse"
        ),
        "tests": [
            "decompose:presupposition",
            "decompose:temporal",
            "judge:cherry_picked_timeframe",
        ],
        "notes": (
            "'Since X stopped' presupposes X was doing it before. 'Since "
            "2015' is a specific temporal anchor. 'Only gotten worse' is "
            "a trend claim. Tests presupposition extraction and temporal "
            "cherry-pick detection."
        ),
    },
    {
        "text": (
            "No Republican president since Eisenhower has reduced the "
            "federal deficit during their term in office"
        ),
        "tests": [
            "decompose:negation",
            "decompose:temporal",
            "judge:exhaustive_check",
        ],
        "notes": (
            "Existential negation requiring checking every Republican "
            "president from Nixon through Trump. 'Reduced the deficit' "
            "vs 'eliminated' — operationalization matters."
        ),
    },
    {
        "text": (
            "Sure, unemployment is low, but if you count all the people "
            "who just gave up looking for work the real number is closer to 25%"
        ),
        "tests": [
            "normalization:colloquial_language",
            "decompose:concession_structure",
            "judge:statistical_framing",
            "judge:quantitative",
        ],
        "notes": (
            "Colloquial hedging ('sure, but'). The concession structure "
            "admits unemployment is low then pivots. '25%' is a specific "
            "number to verify against U-6 unemployment data. Tests whether "
            "judge does the math."
        ),
    },

    # ── Research + Judge scenarios ──────────────────────────────────────
    {
        "text": (
            "SpaceX has received over $15 billion in government contracts "
            "while Elon Musk runs the agency cutting government spending"
        ),
        "tests": [
            "research:wikidata_expansion",
            "judge:interested_parties",
            "judge:conflict_of_interest",
        ],
        "notes": (
            "Wikidata expansion should connect SpaceX → Elon Musk → DOGE. "
            "Tests interested party detection and conflict of interest "
            "flagging. Both sub-claims are individually verifiable."
        ),
    },
    {
        "text": (
            "Rupert Murdoch's media empire has systematically promoted "
            "climate skepticism across Fox News, Sky News, and the New York Post"
        ),
        "tests": [
            "research:wikidata_expansion",
            "judge:interested_parties",
            "decompose:generic_habitual",
        ],
        "notes": (
            "Wikidata should expand Murdoch → News Corp → Fox/Sky/NYP. "
            "'Systematically promoted' is a habitual/generic claim. "
            "Media outlets are interested parties re: their own editorial stance."
        ),
    },
    {
        "text": (
            "According to the WHO, COVID-19 most likely originated from a "
            "lab leak at the Wuhan Institute of Virology"
        ),
        "tests": [
            "decompose:attribution",
            "decompose:evidentiality_marker",
            "judge:source_verification",
        ],
        "notes": (
            "Attribution claim ('According to the WHO'). Must verify BOTH "
            "that the WHO said this AND whether it's substantively true. "
            "The WHO has NOT said this — tests whether the system catches "
            "false attribution."
        ),
    },
    {
        "text": (
            "Israel's treatment of Palestinians in the West Bank meets the "
            "legal definition of apartheid under international law"
        ),
        "tests": [
            "decompose:definition_claim",
            "judge:contested_category",
            "judge:legal_regulatory",
            "normalization:operationalize_vague",
        ],
        "notes": (
            "Contested category claim. 'Apartheid' has a specific legal "
            "definition (Rome Statute, Apartheid Convention) but application "
            "is disputed. Tests whether judge handles definition-dependent "
            "truth and notes competing frameworks."
        ),
    },
    {
        "text": (
            "States that implemented strict gun control laws saw a 40% "
            "reduction in gun violence within five years"
        ),
        "tests": [
            "decompose:causal",
            "judge:correlation_vs_causation",
            "judge:statistical_framing",
            "judge:quantitative",
        ],
        "notes": (
            "Causal claim with specific statistic (40%). Even if some "
            "states did see reductions, correlation ≠ causation. Tests "
            "whether judge flags this and checks the specific number."
        ),
    },
    {
        "text": (
            "The Inflation Reduction Act allocated $369 billion for climate "
            "and energy provisions, making it the largest climate investment "
            "in US history"
        ),
        "tests": [
            "research:legiscan",
            "decompose:superlative",
            "judge:quantitative",
        ],
        "notes": (
            "Specific dollar amount + superlative ('largest in US history'). "
            "LegiScan should find the bill. The $369B figure and superlative "
            "are both verifiable against CBO estimates."
        ),
    },
    {
        "text": (
            "Both Google and Meta have paid more in EU antitrust fines than "
            "they paid in EU corporate taxes over the past five years"
        ),
        "tests": [
            "decompose:parallel_comparison",
            "research:wikidata_expansion",
            "judge:quantitative",
            "judge:temporal",
        ],
        "notes": (
            "Parallel comparison (same assertion for two entities). "
            "Wikidata should expand both companies. Requires comparing "
            "two sets of numbers (fines vs taxes) for two companies "
            "over a five-year window. Likely very hard to verify."
        ),
    },

    # ── Synthesize scenarios ───────────────────────────────────────────
    {
        "text": (
            "Amazon pays its warehouse workers a minimum of $15 per hour, "
            "but the company's warehouse injury rate is three times the "
            "industry average and Jeff Bezos personally blocked unionization "
            "efforts at the Bessemer, Alabama facility"
        ),
        "tests": [
            "decompose:multi_part",
            "synthesize:core_vs_detail",
            "synthesize:mixed_verdicts",
            "research:wikidata_expansion",
        ],
        "notes": (
            "Three distinct sub-claims with likely different truth values: "
            "$15 minimum (true), 3x injury rate (roughly true), Bezos "
            "'personally blocked' (oversimplified). Tests whether synthesis "
            "weights them correctly and doesn't just count true/false."
        ),
    },
    {
        "text": (
            "The opioid epidemic started because Purdue Pharma lied about "
            "OxyContin being non-addictive in the late 1990s, and the "
            "Sackler family made over $10 billion from the drug while "
            "hundreds of thousands of Americans died"
        ),
        "tests": [
            "decompose:causal",
            "decompose:temporal",
            "judge:interested_parties",
            "judge:manufactured_recency",
            "synthesize:thesis_based",
            "research:wikidata_expansion",
        ],
        "notes": (
            "Complex causal + temporal + quantitative. Purdue/Sacklers are "
            "interested parties. 'Started because' oversimplifies a "
            "multi-factor epidemic. '$10 billion' and 'hundreds of thousands' "
            "are verifiable numbers. Wikidata should expand Sackler family."
        ),
    },
    {
        "text": (
            "If the United States had maintained its 1960s-era top marginal "
            "tax rate of 91%, the national debt would never have reached "
            "$30 trillion"
        ),
        "tests": [
            "decompose:conditional_counterfactual",
            "decompose:temporal",
            "judge:quantitative",
        ],
        "notes": (
            "Counterfactual conditional — the consequence is unverifiable. "
            "But the antecedent contains verifiable facts: the 91% rate "
            "existed, the debt did exceed $30T. Tests whether decomposer "
            "separates the verifiable from the unverifiable."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper: pretty table
# ---------------------------------------------------------------------------

def _verdict_color(v: str) -> str:
    """ANSI color for verdict."""
    colors = {
        "true": "\033[92m",        # green
        "mostly_true": "\033[93m", # yellow
        "mixed": "\033[33m",       # orange
        "mostly_false": "\033[91m",# red
        "false": "\033[31m",       # dark red
        "unverifiable": "\033[90m",# gray
    }
    reset = "\033[0m"
    return f"{colors.get(v, '')}{v}{reset}"


def _print_results(results: list[dict]):
    """Print results table."""
    print("\n" + "=" * 100)
    print("REGRESSION SUITE RESULTS")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        claim = r["claim"]
        verdict = r.get("verdict", "???")
        confidence = r.get("confidence", 0)
        tests = ", ".join(claim["tests"])

        # Truncate claim text for display
        text = claim["text"]
        if len(text) > 90:
            text = text[:87] + "..."

        print(f"\n{i:2d}. {text}")
        print(f"    Verdict: {_verdict_color(verdict)} "
              f"(confidence: {confidence:.2f})")
        print(f"    Tests: {tests}")

    print("\n" + "=" * 100)

    # Summary
    verdicts = [r.get("verdict", "???") for r in results]
    print(f"\nTotal: {len(results)} claims")
    for v in ["true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"]:
        count = verdicts.count(v)
        if count:
            print(f"  {_verdict_color(v)}: {count}")
    unknown = sum(1 for v in verdicts if v == "???")
    if unknown:
        print(f"  ???: {unknown}")
    print()


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def _http_post(url: str, body: dict) -> dict:
    """POST JSON using stdlib urllib."""
    import urllib.request
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _http_get(url: str) -> dict:
    """GET JSON using stdlib urllib."""
    import urllib.request
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


async def submit_batch(api_url: str, claims: list[dict]) -> list[dict]:
    """Submit claims via batch API. Returns list of {id, text, status}."""
    payload = {
        "claims": [{"text": c["text"]} for c in claims]
    }

    data = _http_post(f"{api_url}/claims/batch", payload)

    submitted = data["claims"]
    print(f"Submitted {len(submitted)} claims via batch API")
    for s in submitted:
        print(f"  {s['id'][:8]}... [{s['status']}] {s['text'][:70]}...")
    return submitted


async def poll_until_done(
    api_url: str,
    claim_ids: list[str],
    poll_interval: int = 15,
    timeout: int = 1800,  # 30 min
) -> list[dict]:
    """Poll claim statuses until all are verified or timeout."""
    start = time.monotonic()
    results = {}

    while time.monotonic() - start < timeout:
        pending = [cid for cid in claim_ids if cid not in results]
        if not pending:
            break

        for cid in pending:
            try:
                data = _http_get(f"{api_url}/claims/{cid}")

                status = data.get("status")
                if status in ("verified", "flagged"):
                    results[cid] = data
                    elapsed = int(time.monotonic() - start)
                    v = data.get("verdict", "???")
                    text = data.get("text", "")[:60]
                    print(f"  [{elapsed:4d}s] {cid[:8]}... "
                          f"{_verdict_color(v)} — {text}...")
            except Exception:
                pass  # Retry on next poll

        remaining = len(claim_ids) - len(results)
        if remaining > 0:
            elapsed = int(time.monotonic() - start)
            print(f"  [{elapsed:4d}s] Waiting... "
                  f"{len(results)}/{len(claim_ids)} done")
            await asyncio.sleep(poll_interval)

    return results


async def run_suite(api_url: str, count: int | None = None, timeout: int | None = None):
    """Run the regression suite (optionally a subset)."""
    claims = CLAIMS[:count] if count else CLAIMS
    n = len(claims)

    # Auto-calculate timeout: 35 min per claim, 40 min minimum
    if timeout is None:
        timeout = max(2400, n * 35 * 60)

    print(f"\nSpin Cycle Regression Suite — {n} of {len(CLAIMS)} claims")
    print(f"API: {api_url}")
    print(f"Timeout: {timeout}s ({timeout // 60} min)")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Submit
    submitted = await submit_batch(api_url, claims)
    claim_ids = [s["id"] for s in submitted]

    # Map IDs back to our claim metadata
    id_to_claim = {}
    for s, c in zip(submitted, claims):
        id_to_claim[s["id"]] = c

    # Poll
    print(f"\nPolling for results (up to {timeout // 60} min)...\n")
    raw_results = await poll_until_done(api_url, claim_ids, timeout=timeout)

    # Build results
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

    # Dump full results to JSON for later analysis
    output_file = f"tests/regression_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global CLAIMS

    parser = argparse.ArgumentParser(
        description="Run Spin Cycle regression test suite"
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
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1 claim (Texas power grid), 40 min timeout",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\nRegression Suite: {len(CLAIMS)} claims\n")
        for i, c in enumerate(CLAIMS, 1):
            text = c["text"]
            if len(text) > 90:
                text = text[:87] + "..."
            print(f"{i:2d}. {text}")
            print(f"    Tests: {', '.join(c['tests'])}")
            print(f"    Notes: {c['notes'][:100]}...")
            print()
        return

    count = args.count
    timeout = args.timeout

    if args.smoke:
        # Texas power grid claim is index 3 — move it to front
        # so --count 1 picks it up
        smoke_claim = CLAIMS[3]  # Texas power grid
        CLAIMS = [smoke_claim] + [c for i, c in enumerate(CLAIMS) if i != 3]
        count = 1
        timeout = timeout or 2400

    asyncio.run(run_suite(args.api_url, count=count, timeout=timeout))


if __name__ == "__main__":
    main()
