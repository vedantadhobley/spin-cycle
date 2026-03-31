"""Test transcript extraction pipeline.

Two modes:
  1. Temporal workflows (--extract, --classify, --synthesize, --full):
     End-to-end tests through real Temporal child workflows.
  2. Standalone (--classify-only, --dedup-only, --synthesize-only):
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


async def run_pipeline(stop_after: str) -> dict:
    """Start TranscriptPipelineWorkflow via Temporal and wait for result."""
    from temporalio.client import Client
    from src.config import TEMPORAL_HOST, TASK_QUEUE
    from src.workflows.transcript_pipeline import TranscriptPipelineWorkflow

    with open(TRANSCRIPT_FILE, "r") as f:
        raw_text = f.read()

    print(f"Connecting to Temporal at {TEMPORAL_HOST}...")
    client = await Client.connect(TEMPORAL_HOST)

    workflow_id = f"test-pipeline-{stop_after}-{uuid.uuid4().hex[:8]}"

    print(f"Starting TranscriptPipelineWorkflow (stop_after={stop_after})")
    print(f"  workflow_id: {workflow_id}")
    print(f"  task_queue: {TASK_QUEUE}")
    print(f"  transcript: {TRANSCRIPT_TITLE}")
    print()

    result = await client.execute_workflow(
        TranscriptPipelineWorkflow.run,
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


async def run_extract_workflow() -> dict:
    """Start ExtractClaimsWorkflow directly (requires fetch result)."""
    from temporalio.client import Client
    from src.config import TEMPORAL_HOST, TASK_QUEUE
    from src.workflows.transcript_pipeline import TranscriptPipelineWorkflow

    # Run pipeline stopping after extract
    return await run_pipeline("extract")


async def run_classify_workflow() -> dict:
    """Run pipeline stopping after dedup."""
    return await run_pipeline("dedup")


async def run_synthesize_workflow() -> dict:
    """Run pipeline stopping after synthesize."""
    return await run_pipeline("synthesize")


async def run_full_pipeline() -> dict:
    """Run full pipeline (no stop_after)."""
    from temporalio.client import Client
    from src.config import TEMPORAL_HOST, TASK_QUEUE
    from src.workflows.transcript_pipeline import TranscriptPipelineWorkflow

    with open(TRANSCRIPT_FILE, "r") as f:
        raw_text = f.read()

    print(f"Connecting to Temporal...")
    client = await Client.connect(TEMPORAL_HOST)

    workflow_id = f"test-pipeline-full-{uuid.uuid4().hex[:8]}"

    print(f"Starting TranscriptPipelineWorkflow (full pipeline)")
    print(f"  workflow_id: {workflow_id}")

    result = await client.execute_workflow(
        TranscriptPipelineWorkflow.run,
        args=[TRANSCRIPT_URL, raw_text, TRANSCRIPT_TITLE, TRANSCRIPT_DATE],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    return result


def print_extract_results(result: dict):
    """Print extraction results."""
    all_theses = result.get("all_theses", [])

    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE: {len(all_theses)} claims extracted")
    print(f"{'='*80}")

    if all_theses:
        by_speaker: dict[str, int] = {}
        for t in all_theses:
            speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
            by_speaker[speaker] = by_speaker.get(speaker, 0) + 1
        print(f"\n  By speaker:")
        for speaker, count in sorted(by_speaker.items(), key=lambda x: -x[1]):
            print(f"    {speaker}: {count} claims")

        topic_counts = Counter(t.get("topic", "?") for t in all_theses)
        print(f"\n  Topics: {dict(topic_counts)}")

        print(f"\n  Claims:")
        for i, t in enumerate(all_theses):
            topic = t.get("topic", "?")
            speaker = t["speakers"][0] if t.get("speakers") else "?"
            print(f"  [{i}] ({topic}) {speaker}: "
                  f"{t['thesis_statement'][:90]}")

        with open(PHASE1_OUTPUT, "w") as f:
            json.dump(all_theses, f, indent=2)
        print(f"\n  Output saved to {PHASE1_OUTPUT}")


def print_dedup_results(result: dict):
    """Print classify+dedup results."""
    theses = result.get("classified_theses", [])
    groups = result.get("dedup_groups", [])

    print(f"\n{'='*80}")
    print(f"CLASSIFY+DEDUP COMPLETE")
    print(f"{'='*80}")
    print(f"  Theses: {len(theses)}")
    print(f"  Groups: {len(groups)}")

    if theses:
        class_counts = Counter(t.get("classification", "?") for t in theses)
        checkable_count = sum(1 for t in theses if t.get("checkable", True))
        print(f"\n  Classifications: {dict(class_counts)}")
        print(f"  Checkable: {checkable_count}/{len(theses)}")

    checkable = [g for g in groups if g.get("checkable")]
    multi = [g for g in groups if len(g.get("member_global_indices", [])) > 1]
    print(f"  Checkable groups: {len(checkable)}")
    print(f"  Multi-member: {len(multi)}")

    for g in groups:
        members = g.get("member_global_indices", [])
        tag = "CHECKABLE" if g.get("checkable") else "SKIP"
        print(f"\n  {g['group_id']} ({tag}) | {g.get('topic', '?')} | "
              f"{len(members)} members")
        print(f"    claim_text: {g.get('claim_text', '')[:120]}")


def print_synth_results(result: dict):
    """Print synthesis results."""
    groups = result.get("checkable_groups", [])
    claim_ids = result.get("claim_ids", [])

    print(f"\n{'='*80}")
    print(f"SYNTHESIS COMPLETE")
    print(f"{'='*80}")
    print(f"  Checkable groups: {len(groups)}")
    print(f"  Claim records created: {len(claim_ids)}")

    for i, g in enumerate(groups):
        members = g.get("member_global_indices", [])
        print(f"\n  [{i}] {g['speaker']} | {g.get('topic', '?')} | "
              f"{len(members)} members")
        print(f"    {g.get('claim_text', '')[:150]}")


def print_full_results(result: dict):
    """Print full pipeline results."""
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"  Transcript ID: {result.get('transcript_id', 'N/A')}")
    print(f"  Title: {result.get('title', 'N/A')}")
    print(f"  Theses: {result.get('thesis_count', '?')}")
    print(f"  Claims: {result.get('claim_count', '?')}")


CLASSIFY_OUTPUT = "tests/artifacts/fii_classify_output.json"


async def run_classify_only():
    """Standalone classification test using saved phase1 data (no Temporal)."""
    from src.transcript.claim_classifier import classify_claims_batch
    from src.config import CLASSIFY_BATCH_SIZE

    if not os.path.exists(PHASE1_OUTPUT):
        print(f"ERROR: No Phase 1 output at {PHASE1_OUTPUT}")
        print("Run --extract first.")
        sys.exit(1)

    with open(PHASE1_OUTPUT) as f:
        phase1_data = json.load(f)

    print(f"\n{'='*80}")
    print(f"STANDALONE CLASSIFY: {len(phase1_data)} claims")
    print(f"{'='*80}")

    all_classified = []
    for batch_start in range(0, len(phase1_data), CLASSIFY_BATCH_SIZE):
        batch = phase1_data[batch_start:batch_start + CLASSIFY_BATCH_SIZE]
        print(f"\n  Classifying batch {batch_start}-{batch_start + len(batch) - 1} "
              f"({len(batch)} claims)...")
        classified = await classify_claims_batch(batch)
        all_classified.extend(classified)

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

    with open(CLASSIFY_OUTPUT, "w") as f:
        json.dump(all_classified, f, indent=2)
    print(f"\n  Output saved to {CLASSIFY_OUTPUT}")


async def run_dedup_only():
    """Standalone dedup test using saved phase1 data (no Temporal)."""
    from src.transcript.claim_dedup import dedup_speaker_claims

    input_file = CLASSIFY_OUTPUT if os.path.exists(CLASSIFY_OUTPUT) else PHASE1_OUTPUT
    if not os.path.exists(input_file):
        print(f"ERROR: No input data at {input_file}")
        print("Run --extract first.")
        sys.exit(1)

    with open(input_file) as f:
        phase1_data = json.load(f)

    print(f"\n{'='*80}")
    print(f"STANDALONE DEDUP: {len(phase1_data)} claims")
    print(f"{'='*80}")

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
        print(f"  -> {len(clusters)} clusters "
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

    input_file = CLASSIFY_OUTPUT if os.path.exists(CLASSIFY_OUTPUT) else PHASE1_OUTPUT
    if not os.path.exists(input_file):
        print(f"ERROR: No input data at {input_file}")
        print("Run --extract or --classify-only first.")
        sys.exit(1)

    with open(input_file) as f:
        claims = json.load(f)

    print(f"\n{'='*80}")
    print(f"STANDALONE SYNTHESIZE: {len(claims)} claims")
    print(f"  Input: {input_file}")
    print(f"{'='*80}")

    speaker_theses: dict[str, list[dict]] = {}
    for t in claims:
        speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
        speaker_theses.setdefault(speaker, []).append(t)

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

        print(f"  -> {len(clusters)} clusters "
              f"({len(checkable_clusters)} checkable, {len(multi)} multi-member)")

        for cluster in single:
            idx = cluster["member_indices"][0]
            all_results.append({
                "speaker": speaker,
                "topic": cluster.get("topic", "?"),
                "members": 1,
                "claim_text": sp_theses[idx]["thesis_statement"],
                "synthesized": False,
            })

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
            print(f"  -> \"{synth.overarching_claim[:120]}\"")

            all_results.append({
                "speaker": speaker,
                "topic": topic,
                "members": len(member_stmts),
                "claim_text": synth.overarching_claim,
                "synthesized": True,
            })

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

    with open(PHASE2_OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Output saved to {PHASE2_OUTPUT}")


async def main():
    args = sys.argv[1:]

    if "--extract" in args:
        result = await run_pipeline("extract")
        print_extract_results(result)

    elif "--classify" in args:
        result = await run_pipeline("dedup")
        print_dedup_results(result)

    elif "--synthesize" in args:
        result = await run_pipeline("synthesize")
        print_synth_results(result)

    elif "--full" in args:
        result = await run_full_pipeline()
        print_full_results(result)

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
        print("  --extract     ExtractClaimsWorkflow (stop_after=extract)")
        print("  --classify    ClassifyAndDedupWorkflow (stop_after=dedup)")
        print("  --synthesize  SynthesizeClaimsWorkflow (stop_after=synthesize)")
        print("  --full        TranscriptPipelineWorkflow (full pipeline)")
        print()
        print("Standalone (no Temporal, uses saved artifacts):")
        print("  --classify-only     Classify saved phase1 data")
        print("  --dedup-only        Dedup saved classified/phase1 data")
        print("  --synthesize-only   Dedup + synthesize multi-member clusters")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
