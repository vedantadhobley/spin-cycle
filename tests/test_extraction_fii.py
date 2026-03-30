"""Test transcript extraction pipeline.

Two modes:
  1. Temporal workflows (--phase1, --classify, --review, --extract):
     End-to-end tests through real Temporal. Each re-runs all prior stages.
  2. Standalone (--classify-only, --dedup-only):
     Test individual stages using saved artifacts. Fast, no re-extraction.

Artifacts are saved between runs so downstream stages can reuse them:
  tests/artifacts/fii_phase1_output.json   — raw 4-field claims
  tests/artifacts/fii_classify_output.json — claims with classification
  tests/artifacts/fii_phase2_output.json   — dedup results

Usage (from Docker):
    docker compose -f docker-compose.dev.yml exec worker \
        python tests/test_extraction_fii.py <flag>
"""
import asyncio
import json
import sys
import os
import uuid

sys.path.insert(0, ".")

os.environ.setdefault("LOG_FORMAT", "pretty")
from src.utils.logging import configure_logging
configure_logging()

from collections import Counter

TRANSCRIPT_FILE = "tests/artifacts/trump_fii_2026"
TRANSCRIPT_URL = "https://www.c-span.org/program/white-house-event/president-trump-speaks-at-saudi-investors-forum-in-miami/676455"
TRANSCRIPT_TITLE = "President Trump Speaks at Saudi Investors Forum in Miami"
TRANSCRIPT_DATE = "2026-03-27"

PHASE1_OUTPUT = "tests/artifacts/fii_phase1_output.json"
PHASE2_OUTPUT = "tests/artifacts/fii_phase2_output.json"


async def run_workflow(stop_after: str) -> dict:
    """Start ExtractTranscriptWorkflow via Temporal and wait for result."""
    from temporalio.client import Client
    from src.config import TEMPORAL_HOST, TASK_QUEUE
    from src.workflows.extract_transcript import ExtractTranscriptWorkflow

    with open(TRANSCRIPT_FILE, "r") as f:
        raw_text = f.read()

    print(f"Connecting to Temporal at {TEMPORAL_HOST}...")
    client = await Client.connect(TEMPORAL_HOST)

    workflow_id = f"test-extract-{stop_after}-{uuid.uuid4().hex[:8]}"

    print(f"Starting ExtractTranscriptWorkflow (stop_after={stop_after})")
    print(f"  workflow_id: {workflow_id}")
    print(f"  task_queue: {TASK_QUEUE}")
    print(f"  transcript: {TRANSCRIPT_TITLE}")
    print()

    result = await client.execute_workflow(
        ExtractTranscriptWorkflow.run,
        args=[
            TRANSCRIPT_URL,      # url
            raw_text,            # raw_text
            TRANSCRIPT_TITLE,    # title
            TRANSCRIPT_DATE,     # date
            stop_after,          # stop_after
        ],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    return result


def print_phase1_results(result: dict):
    """Print and save Phase 1 results."""
    all_theses = result.get("all_theses", [])

    print(f"\n{'='*80}")
    print(f"PHASE 1 COMPLETE: {result['thesis_count']} claims extracted")
    print(f"{'='*80}")
    print(f"  Title: {result['title']}")
    print(f"  Words: {result['word_count']}")
    print(f"  Turns: {result['turn_count']}")
    print(f"  Speakers: {result['speakers']}")
    print(f"  Transcript ID: {result.get('transcript_id', 'N/A')}")

    if all_theses:
        # Speaker breakdown
        by_speaker: dict[str, int] = {}
        for t in all_theses:
            speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
            by_speaker[speaker] = by_speaker.get(speaker, 0) + 1
        print(f"\n  By speaker:")
        for speaker, count in sorted(by_speaker.items(), key=lambda x: -x[1]):
            print(f"    {speaker}: {count} claims")

        # Topic breakdown
        topic_counts = Counter(t.get("topic", "?") for t in all_theses)
        print(f"\n  Topics: {dict(topic_counts)}")

        # Print each claim (no classification at phase1)
        print(f"\n  Claims:")
        for i, t in enumerate(all_theses):
            topic = t.get("topic", "?")
            speaker = t["speakers"][0] if t.get("speakers") else "?"
            print(f"  [{i}] ({topic}) {speaker}: "
                  f"{t['thesis_statement'][:90]}")

        # Save
        with open(PHASE1_OUTPUT, "w") as f:
            json.dump(all_theses, f, indent=2)
        print(f"\n  Phase 1 output saved to {PHASE1_OUTPUT}")


def print_classify_results(result: dict):
    """Print and save classification results."""
    all_theses = result.get("all_theses", [])

    print(f"\n{'='*80}")
    print(f"CLASSIFICATION COMPLETE: {result['thesis_count']} claims classified")
    print(f"{'='*80}")
    print(f"  Title: {result['title']}")
    print(f"  Transcript ID: {result.get('transcript_id', 'N/A')}")

    if all_theses:
        # Classification breakdown
        class_counts = Counter(t.get("classification", "?") for t in all_theses)
        checkable_count = sum(1 for t in all_theses if t.get("checkable", True))
        print(f"\n  Classifications: {dict(class_counts)}")
        print(f"  Checkable: {checkable_count}/{len(all_theses)}")

        # Print each claim with classification
        print(f"\n  Claims:")
        for i, t in enumerate(all_theses):
            checkmark = "V" if t.get("checkable", True) else "X"
            cls = t.get("classification", "?")
            speaker = t["speakers"][0] if t.get("speakers") else "?"
            print(f"  [{i}] [{checkmark}] ({cls}) {speaker}: "
                  f"{t['thesis_statement'][:90]}")

        # Save
        with open(PHASE1_OUTPUT, "w") as f:
            json.dump(all_theses, f, indent=2)
        print(f"\n  Output saved to {PHASE1_OUTPUT}")


def print_dedup_results(result: dict):
    """Print dedup/review stage results."""
    groups = result.get("groups", [])

    print(f"\n{'='*80}")
    print(f"DEDUP COMPLETE: {result.get('group_count', len(groups))} groups")
    print(f"{'='*80}")
    print(f"  Thesis count: {result['thesis_count']}")
    print(f"  Group count: {result.get('group_count', '?')}")

    checkable = [g for g in groups if g.get("checkable")]
    multi = [g for g in groups if len(g.get("member_global_indices", [])) > 1]
    print(f"  Checkable: {len(checkable)}")
    print(f"  Multi-member: {len(multi)}")

    # Print groups
    for g in groups:
        members = g.get("member_global_indices", [])
        tag = "CHECKABLE" if g.get("checkable") else "SKIP"
        print(f"\n  {g['group_id']} ({tag}) | {g.get('topic', '?')} | "
              f"{len(members)} members")
        print(f"    claim_text: {g.get('claim_text', '')[:120]}")
        if g.get("original_quotes"):
            for q in g["original_quotes"][:2]:
                print(f"    quote: \"{q[:100]}\"")


def print_extract_results(result: dict):
    """Print full extraction (dedup + synthesis) results."""
    groups = result.get("groups", [])

    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Thesis count: {result['thesis_count']}")
    print(f"  Claim count: {result.get('claim_count', '?')}")
    print(f"  Group count: {result.get('group_count', '?')}")

    checkable = [g for g in groups if g.get("checkable")]
    multi = [g for g in groups if len(g.get("member_global_indices", [])) > 1]
    single = [g for g in groups if len(g.get("member_global_indices", [])) == 1]
    print(f"  Checkable: {len(checkable)}")
    print(f"  Multi-member (synthesized): {len(multi)}")
    print(f"  Single-member (no synthesis): {len(single)}")

    # Print checkable groups with final claim text
    print(f"\n  Checkable claims for verification:")
    for i, g in enumerate(checkable):
        members = g.get("member_global_indices", [])
        print(f"\n  [{i}] {g['speaker']} | {g.get('topic', '?')} | "
              f"{len(members)} members")
        print(f"    {g.get('claim_text', '')[:150]}")


CLASSIFY_OUTPUT = "tests/artifacts/fii_classify_output.json"


async def run_classify_only():
    """Standalone classification test using saved phase1 data (no Temporal)."""
    from src.transcript.claim_classifier import classify_claims_batch
    from src.config import CLASSIFY_BATCH_SIZE

    if not os.path.exists(PHASE1_OUTPUT):
        print(f"ERROR: No Phase 1 output at {PHASE1_OUTPUT}")
        print("Run --phase1 first.")
        sys.exit(1)

    with open(PHASE1_OUTPUT) as f:
        phase1_data = json.load(f)

    print(f"\n{'='*80}")
    print(f"STANDALONE CLASSIFY: {len(phase1_data)} claims")
    print(f"{'='*80}")

    # Batch classify (same batching as the workflow)
    all_classified = []
    for batch_start in range(0, len(phase1_data), CLASSIFY_BATCH_SIZE):
        batch = phase1_data[batch_start:batch_start + CLASSIFY_BATCH_SIZE]
        print(f"\n  Classifying batch {batch_start}-{batch_start + len(batch) - 1} "
              f"({len(batch)} claims)...")
        classified = await classify_claims_batch(batch)
        all_classified.extend(classified)

    # Print results
    class_counts = Counter(t.get("classification", "?") for t in all_classified)
    checkable_count = sum(1 for t in all_classified if t.get("checkable", True))
    print(f"\n  Classifications: {dict(class_counts)}")
    print(f"  Checkable: {checkable_count}/{len(all_classified)}")

    print(f"\n  Claims:")
    for i, t in enumerate(all_classified):
        checkmark = "V" if t.get("checkable", True) else "X"
        cls = t.get("classification", "?")
        speaker = t["speakers"][0] if t.get("speakers") else "?"
        rationale = t.get("check_rationale", "")
        print(f"  [{i}] [{checkmark}] ({cls}) {speaker}: "
              f"{t['thesis_statement'][:80]}")
        if not t.get("checkable", True):
            print(f"       reason: {rationale}")

    # Save
    with open(CLASSIFY_OUTPUT, "w") as f:
        json.dump(all_classified, f, indent=2)
    print(f"\n  Output saved to {CLASSIFY_OUTPUT}")


async def run_dedup_only():
    """Standalone dedup test using saved phase1 data (no Temporal)."""
    from src.transcript.claim_dedup import dedup_speaker_claims

    # Use classified data if available, fall back to raw phase1
    input_file = CLASSIFY_OUTPUT if os.path.exists(CLASSIFY_OUTPUT) else PHASE1_OUTPUT
    if not os.path.exists(input_file):
        print(f"ERROR: No input data at {input_file}")
        print("Run --phase1 first.")
        sys.exit(1)

    with open(input_file) as f:
        phase1_data = json.load(f)

    print(f"\n{'='*80}")
    print(f"STANDALONE DEDUP: {len(phase1_data)} claims")
    print(f"{'='*80}")

    # Group by speaker
    speaker_theses: dict[str, list[dict]] = {}
    for t in phase1_data:
        speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
        speaker_theses.setdefault(speaker, []).append(t)

    for speaker, sp_theses in speaker_theses.items():
        print(f"\n--- {speaker}: {len(sp_theses)} claims ---")

        if len(sp_theses) <= 1:
            print("  (trivial — 1 claim)")
            continue

        result = await dedup_speaker_claims(sp_theses, speaker)
        clusters = result["clusters"]
        checkable = sum(1 for c in clusters if c["checkable"])
        multi = sum(1 for c in clusters if len(c["member_indices"]) > 1)
        print(f"  → {len(clusters)} clusters "
              f"({checkable} checkable, {multi} multi-member)")

        for ci, cluster in enumerate(clusters):
            members = cluster["member_indices"]
            if len(members) > 1:
                tag = "CHECKABLE" if cluster["checkable"] else "SKIP"
                print(f"\n    C{ci} ({tag}) | {cluster['topic']} | "
                      f"{len(members)} members")
                for idx in members:
                    is_rep = (sp_theses[idx]["thesis_statement"]
                              == cluster["representative"]["thesis_statement"])
                    marker = " *" if is_rep else ""
                    print(f"      [{idx}] \"{sp_theses[idx]['thesis_statement'][:90]}\""
                          f"{marker}")


async def run_synthesize_only():
    """Standalone synthesis test: dedup + synthesize multi-member clusters."""
    from src.transcript.claim_dedup import dedup_speaker_claims
    from src.transcript.claim_synthesizer import synthesize_group_claim

    # Use classified data if available, fall back to raw phase1
    input_file = CLASSIFY_OUTPUT if os.path.exists(CLASSIFY_OUTPUT) else PHASE1_OUTPUT
    if not os.path.exists(input_file):
        print(f"ERROR: No input data at {input_file}")
        print("Run --phase1 or --classify-only first.")
        sys.exit(1)

    with open(input_file) as f:
        claims = json.load(f)

    print(f"\n{'='*80}")
    print(f"STANDALONE SYNTHESIZE: {len(claims)} claims")
    print(f"  Input: {input_file}")
    print(f"{'='*80}")

    # Group by speaker
    speaker_theses: dict[str, list[dict]] = {}
    for t in claims:
        speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
        speaker_theses.setdefault(speaker, []).append(t)

    # Dedup per speaker, collect multi-member clusters for synthesis
    all_results = []

    for speaker, sp_theses in speaker_theses.items():
        print(f"\n--- {speaker}: {len(sp_theses)} claims ---")

        if len(sp_theses) <= 1:
            stmt = sp_theses[0]["thesis_statement"]
            checkable = sp_theses[0].get("checkable", True)
            print(f"  (1 claim, {'checkable' if checkable else 'skip'})")
            if checkable:
                all_results.append({
                    "speaker": speaker,
                    "topic": sp_theses[0].get("topic", "?"),
                    "members": 1,
                    "claim_text": stmt,
                    "synthesized": False,
                })
            continue

        result = await dedup_speaker_claims(sp_theses, speaker)
        clusters = result["clusters"]
        checkable_clusters = [c for c in clusters if c["checkable"]]
        multi = [c for c in checkable_clusters if len(c["member_indices"]) > 1]
        single = [c for c in checkable_clusters if len(c["member_indices"]) == 1]

        print(f"  → {len(clusters)} clusters "
              f"({len(checkable_clusters)} checkable, {len(multi)} multi-member)")

        # Single-member: use thesis_statement directly
        for cluster in single:
            idx = cluster["member_indices"][0]
            all_results.append({
                "speaker": speaker,
                "topic": cluster.get("topic", "?"),
                "members": 1,
                "claim_text": sp_theses[idx]["thesis_statement"],
                "synthesized": False,
            })

        # Multi-member: synthesize
        for ci, cluster in enumerate(multi):
            member_stmts = [
                sp_theses[idx]["thesis_statement"]
                for idx in cluster["member_indices"]
            ]
            topic = cluster.get("topic", "?")

            print(f"\n  Synthesizing C{ci} ({topic}, "
                  f"{len(member_stmts)} members)...")
            for idx in cluster["member_indices"]:
                print(f"    [{idx}] \"{sp_theses[idx]['thesis_statement'][:90]}\"")

            synth = await synthesize_group_claim(member_stmts, topic, speaker)
            print(f"  → \"{synth.overarching_claim[:120]}\"")

            all_results.append({
                "speaker": speaker,
                "topic": topic,
                "members": len(member_stmts),
                "claim_text": synth.overarching_claim,
                "synthesized": True,
            })

    # Summary
    checkable_count = len(all_results)
    synth_count = sum(1 for r in all_results if r["synthesized"])
    print(f"\n{'='*80}")
    print(f"SYNTHESIS COMPLETE: {checkable_count} checkable claims")
    print(f"  Synthesized (multi-member): {synth_count}")
    print(f"  Direct (single-member): {checkable_count - synth_count}")
    print(f"{'='*80}")

    print(f"\n  Final claims for verification:")
    for i, r in enumerate(all_results):
        tag = "SYNTH" if r["synthesized"] else "DIRECT"
        print(f"\n  [{i}] ({tag}) {r['speaker']} | {r['topic']} | "
              f"{r['members']} members")
        print(f"    {r['claim_text'][:150]}")

    # Save
    with open(PHASE2_OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Output saved to {PHASE2_OUTPUT}")


async def main():
    args = sys.argv[1:]

    if "--phase1" in args:
        result = await run_workflow("phase1")
        print_phase1_results(result)

    elif "--classify" in args:
        result = await run_workflow("classify")
        print_classify_results(result)

    elif "--review" in args:
        result = await run_workflow("review")
        print_dedup_results(result)

    elif "--extract" in args:
        result = await run_workflow("extract")
        print_extract_results(result)

    elif "--classify-only" in args:
        await run_classify_only()

    elif "--dedup-only" in args:
        await run_dedup_only()

    elif "--synthesize-only" in args:
        await run_synthesize_only()

    else:
        print("Usage: python tests/test_extraction_fii.py <stage>")
        print()
        print("Stages (run real Temporal workflows):")
        print("  --phase1      Extract only (stop_after=phase1)")
        print("  --classify    Extract + classify (stop_after=classify)")
        print("  --review      Extract + classify + dedup (stop_after=review)")
        print("  --extract     Full extraction + synthesis (stop_after=extract)")
        print()
        print("Standalone (no Temporal, uses saved artifacts):")
        print("  --classify-only     Classify saved phase1 data")
        print("  --dedup-only        Dedup saved classified/phase1 data")
        print("  --synthesize-only   Dedup + synthesize multi-member clusters")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
