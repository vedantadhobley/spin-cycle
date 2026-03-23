"""Rev.com transcript claims — real claims from real transcripts.

Paste claims from Rev.com transcripts into the CLAIMS list below.
Each claim can optionally include a speaker name and source URL.

Run with: python -m tests.rev_claims [--api-url URL]

Usage:
  python -m tests.rev_claims                    # submit all, poll for results
  python -m tests.rev_claims --list             # just list claims
  python -m tests.rev_claims --count 5          # submit first 5 only
  python -m tests.rev_claims --timeout 7200     # custom poll timeout
"""

import argparse
import asyncio
import json
import sys
import time

# ---------------------------------------------------------------------------
# Claims — paste from Rev.com transcripts
#
# Format:
#   {"text": "the claim", "speaker": "who said it", "source": "https://..."}
#
# Only "text" is required. Speaker and source are optional.
# ---------------------------------------------------------------------------

CLAIMS = [
    # Example (delete and replace with real claims):
    # {
    #     "text": "We've created more jobs in three years than any president in history",
    #     "speaker": "Joe Biden",
    #     "source": "https://rev.com/...",
    # },
    # {
    #     "text": "Staff Sergeant Jorge Oliveira, he was one of my sergeants or one of my specialists in Guantanamo Bay. He deployed later to Afghanistan, where he was killed on 19 October 2011.",
    #     "speaker": "Pete Hegseth",
    #     "source": "https://rev.com/...",
    # },
    {
        "text": "Iran is a vast country, and just like Hamas and their tunnels, they've poured any aid, any economic development, humanitarian aid into tunnels and rockets.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "Iran has funneled decades of state resources not to their people, but into missiles and drones and proxies and buried facilities.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "To date, we've struck over 7,000 targets across Iran and its military infrastructure.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "Ballistic missile attacks against our forces down 90% since the start of the conflict. Same with one-way attack UAVs, think Kamikaze drones, down 90%.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "We've damaged or sunk over 120 of their Navy ships with battle damage assessments pending for many more.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "Iran has terrorized the United States and our interests for 47 years. Their core industries, not steel or agriculture or tourism. Their core industries are state sponsored terrorism, proxy militias, underground networks, ballistic missiles, and a violent, messianic, Islamist ideology chasing some sort of apocalyptic endgame.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "Iran has weaponized energy for decades.",
        "speaker": "Pete Hegseth",
        "source": "https://rev.com/transcripts/pentagon-press-briefing-for-3-19-26",
    },
    {
        "text": "In 2025, last year, new US vehicle sales rose by 2.4%, which is the biggest jump in American made automobiles since 2019, the last time Donald J. Trump was president.",
        "speaker": "JD Vance",
        "source": "https://rev.com/...",
    },
    {
        "text": "If you like no taxes on Social Security, remember, Republicans made it happen in Congress. If you like no taxes on overtime, because we've got a lot of great overtime workers just here in this place, you ought to thank a Republican in Congress because congressional Republicans made no taxes on overtime happen and Democrats fought us every single step of the way. And if you happen to believe that lowering taxes for people making less than $100,000 a year, and everybody else, by the way, is good policy, then again, we got to remember that it was congressional Republicans that made those victories happen.",
        "speaker": "JD Vance",
        "source": "https://rev.com/...",
    },
    {
        "text": "under the Biden administration, the average American worker lost $ 3,000 in take-home pay. That was a combination of two things, higher prices, that terrible inflation problem that we had under Joe Biden's leadership, but it was also because of higher taxes. So, every single one of us, we got about an average of $3,000 poorer while Joe Biden and the Democrats ran Washington, D.C.",
        "speaker": "JD Vance",
        "source": "https://rev.com/...",
    },
    {
        "text": "over the last 14 months under Donald Trump's leadership, the average American has actually increased their take-home pay by about $1,400.",
        "speaker": "JD Vance",
        "source": "https://rev.com/...",
    },
]

# ---------------------------------------------------------------------------
# Infrastructure (reused from regression_claims.py)
# ---------------------------------------------------------------------------

def _verdict_color(v: str) -> str:
    colors = {
        "true": "\033[92m",
        "mostly_true": "\033[93m",
        "mixed": "\033[33m",
        "mostly_false": "\033[91m",
        "false": "\033[31m",
        "unverifiable": "\033[90m",
    }
    reset = "\033[0m"
    return f"{colors.get(v, '')}{v}{reset}"


def _print_results(results: list[dict]):
    print("\n" + "=" * 100)
    print("REV.COM CLAIMS RESULTS")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        claim = r["claim"]
        verdict = r.get("verdict", "???")
        confidence = r.get("confidence", 0)
        speaker = claim.get("speaker", "")

        text = claim["text"]
        if len(text) > 85:
            text = text[:82] + "..."

        speaker_tag = f" [{speaker}]" if speaker else ""
        print(f"\n{i:2d}.{speaker_tag} {text}")
        print(f"    Verdict: {_verdict_color(verdict)} "
              f"(confidence: {confidence:.2f})")

    print("\n" + "=" * 100)

    verdicts = [r.get("verdict", "???") for r in results]
    print(f"\nTotal: {len(results)} claims")
    for v in ["true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"]:
        count = verdicts.count(v)
        if count:
            print(f"  {_verdict_color(v)}: {count}")
    print()


def _http_post(url: str, body: dict) -> dict:
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
    import urllib.request
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


async def submit_batch(api_url: str, claims: list[dict]) -> list[dict]:
    payload = {
        "claims": [
            {
                "text": c["text"],
                **({"source": c["source"]} if c.get("source") else {}),
                **({"speaker": c["speaker"]} if c.get("speaker") else {}),
            }
            for c in claims
        ]
    }

    data = _http_post(f"{api_url}/claims/batch", payload)

    submitted = data["claims"]
    print(f"Submitted {len(submitted)} claims via batch API")
    for s in submitted:
        # Find matching claim for speaker tag
        matching = next((c for c in claims if c["text"] == s["text"]), {})
        speaker = matching.get("speaker", "")
        tag = f" [{speaker}]" if speaker else ""
        print(f"  {s['id'][:8]}... [{s['status']}]{tag} {s['text'][:65]}...")
    return submitted


async def poll_until_done(
    api_url: str,
    claim_ids: list[str],
    poll_interval: int = 15,
    timeout: int = 7200,
) -> dict:
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
                pass

        remaining = len(claim_ids) - len(results)
        if remaining > 0:
            elapsed = int(time.monotonic() - start)
            print(f"  [{elapsed:4d}s] Waiting... "
                  f"{len(results)}/{len(claim_ids)} done")
            await asyncio.sleep(poll_interval)

    return results


async def run_suite(api_url: str, count: int | None = None, timeout: int = 7200):
    claims = CLAIMS[:count] if count else CLAIMS
    n = len(claims)

    if n == 0:
        print("No claims in CLAIMS list. Paste some in and try again.")
        return []

    print(f"\nRev.com Claims — {n} claims")
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

    # Save results
    from pathlib import Path
    results_dir = Path("tests/regression_results")
    results_dir.mkdir(exist_ok=True)
    output_file = str(
        results_dir / f"rev_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    serializable = []
    for r in results:
        serializable.append({
            "text": r["claim"]["text"],
            "speaker": r["claim"].get("speaker"),
            "source": r["claim"].get("source"),
            "verdict": r["verdict"],
            "confidence": r["confidence"],
        })
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Full results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run claims from Rev.com transcripts"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:4500",
        help="API base URL (default: http://localhost:4500)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Just list claims, don't submit",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Run only the first N claims",
    )
    parser.add_argument(
        "--timeout", type=int, default=7200,
        help="Poll timeout in seconds (default: 7200 / 2 hours)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\nRev.com Claims: {len(CLAIMS)} claims\n")
        for i, c in enumerate(CLAIMS, 1):
            text = c["text"]
            if len(text) > 85:
                text = text[:82] + "..."
            speaker = c.get("speaker", "")
            source = c.get("source", "")
            tag = f" [{speaker}]" if speaker else ""
            print(f"{i:2d}.{tag} {text}")
            if source:
                print(f"    Source: {source}")
            print()
        return

    asyncio.run(run_suite(args.api_url, count=args.count, timeout=args.timeout))


if __name__ == "__main__":
    main()
