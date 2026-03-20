"""Benchmark: sequential vs concurrent LLM inference.

Sends the same prompt twice to the LLM and measures:
  1. Single call latency (baseline)
  2. Two sequential calls (2x baseline expected)
  3. Two concurrent calls (if faster than sequential, concurrency helps)

Usage:
  python -m tests.benchmark_concurrency [--url URL] [--runs N]

Defaults to LLAMA_URL from .env or http://localhost:3101
"""

import argparse
import asyncio
import os
import time
import statistics

from openai import AsyncOpenAI

# A non-trivial prompt that takes real inference time — similar to what
# the research/judge agents process. Long enough to measure accurately.
PROMPT = """\
You are a fact-checker. Evaluate the following claim based on the evidence provided.

Claim: "The Great Wall of China is visible from space with the naked eye."

Evidence:
1. NASA astronaut Chris Hadfield stated he could not see the Great Wall from
   the International Space Station, which orbits at approximately 400 km altitude.
2. A 2004 study published in the journal Science found that the Great Wall is
   too narrow (about 6 meters wide) to be visible from low Earth orbit without aid.
3. Chinese astronaut Yang Liwei reported he could not see the Great Wall during
   his 2003 Shenzhou 5 mission.

Provide your assessment in 3-4 sentences, citing the evidence by number.
"""


async def single_call(client: AsyncOpenAI, model: str) -> float:
    """Make one LLM call, return elapsed seconds."""
    t0 = time.monotonic()
    await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=512,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return time.monotonic() - t0


async def run_benchmark(base_url: str, model: str, runs: int = 3):
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="not-needed")

    print(f"LLM: {base_url}")
    print(f"Model: {model}")
    print(f"Runs per test: {runs}")
    print()

    # Warmup
    print("Warming up...")
    await single_call(client, model)
    print()

    single_times = []
    sequential_times = []
    concurrent_times = []

    for i in range(runs):
        print(f"── Run {i+1}/{runs} ──")

        # 1. Single call
        t = await single_call(client, model)
        single_times.append(t)
        print(f"  Single:     {t:.2f}s")

        # 2. Two sequential calls
        t0 = time.monotonic()
        await single_call(client, model)
        await single_call(client, model)
        seq_total = time.monotonic() - t0
        sequential_times.append(seq_total)
        print(f"  Sequential: {seq_total:.2f}s (2 calls)")

        # 3. Two concurrent calls
        t0 = time.monotonic()
        await asyncio.gather(
            single_call(client, model),
            single_call(client, model),
        )
        con_total = time.monotonic() - t0
        concurrent_times.append(con_total)
        print(f"  Concurrent: {con_total:.2f}s (2 calls)")
        print()

    # Summary
    avg_single = statistics.mean(single_times)
    avg_seq = statistics.mean(sequential_times)
    avg_con = statistics.mean(concurrent_times)

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Single call avg:     {avg_single:.2f}s")
    print(f"  Sequential (2) avg:  {avg_seq:.2f}s  ({avg_seq/avg_single:.2f}x single)")
    print(f"  Concurrent (2) avg:  {avg_con:.2f}s  ({avg_con/avg_single:.2f}x single)")
    print()
    speedup = avg_seq / avg_con
    print(f"  Concurrent speedup:  {speedup:.2f}x vs sequential")
    if speedup > 1.3:
        print(f"  → Concurrency HELPS: {speedup:.2f}x faster, worth parallelizing")
    elif speedup > 1.05:
        print(f"  → Concurrency MARGINAL: {speedup:.2f}x faster, small benefit")
    else:
        print(f"  → Concurrency NO BENEFIT: sequential is ~same speed, use MAX_CONCURRENT=1")
    print()

    # Throughput comparison
    seq_tput = 2 / avg_seq
    con_tput = 2 / avg_con
    print(f"  Sequential throughput: {seq_tput:.2f} calls/sec")
    print(f"  Concurrent throughput: {con_tput:.2f} calls/sec")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM concurrency")
    parser.add_argument("--url", default=None, help="LLM base URL")
    parser.add_argument("--model", default="Qwen3.5-122B-A10B", help="Model name")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    args = parser.parse_args()

    url = args.url or os.getenv("LLAMA_URL", "http://localhost:3101")
    asyncio.run(run_benchmark(url, args.model, args.runs))


if __name__ == "__main__":
    main()
