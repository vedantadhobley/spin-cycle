"""Test chunked extraction + claim review on FII transcript.

Runs Phase 1 (chunked extraction) and Phase 2 (claim review/grouping)
independently so each step can be inspected.

Usage (from Docker):
    # Phase 1 only (chunking + extraction):
    docker compose -f docker-compose.dev.yml exec worker \
        python tests/test_extraction_fii.py --phase1

    # Phase 2 only (review/grouping from saved Phase 1 output):
    docker compose -f docker-compose.dev.yml exec worker \
        python tests/test_extraction_fii.py --phase2

    # Both phases:
    docker compose -f docker-compose.dev.yml exec worker \
        python tests/test_extraction_fii.py
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, ".")

# Configure structured logging so we see validation error details
os.environ.setdefault("LOG_FORMAT", "pretty")
from src.utils.logging import configure_logging
configure_logging()

PHASE1_OUTPUT = "tests/artifacts/fii_phase1_output.json"


def parse_transcript():
    from src.transcript.parsers.raw_text import parse_raw_text

    with open("tests/artifacts/trump_fii_2026", "r") as f:
        raw_text = f.read()

    transcript = parse_raw_text(
        content=raw_text,
        url="https://www.c-span.org/program/white-house-event/president-trump-speaks-at-saudi-investors-forum-in-miami/676455",
        title="President Trump Speaks at Saudi Investors Forum in Miami",
        date="2026-03-27",
    )
    return transcript


async def run_phase1():
    """Phase 1: Build chunks and extract claims from each."""
    from src.transcript.thesis_extractor import build_chunks, extract_chunk
    from src.transcript.speakers import _enrich_speakers

    transcript = parse_transcript()

    print(f"Parsed: {transcript.turn_count} turns, "
          f"{transcript.word_count} words, "
          f"speakers: {transcript.speakers}")

    # Build chunks
    chunks = build_chunks(transcript.turns)
    print(f"\n{'='*80}")
    print(f"CHUNKING: {len(chunks)} chunks from {transcript.turn_count} turns")
    print(f"{'='*80}")
    for i, c in enumerate(chunks):
        target_words = len(c.target_text.split())
        print(f"  Chunk {i}: {target_words} target words, "
              f"{len(c.context_before.split())} before, "
              f"{len(c.context_after.split())} after")

    # Enrich speakers
    enriched = await _enrich_speakers(transcript.speakers)
    print(f"\nSpeakers enriched: {[s['name'] for s in enriched]}")

    # Extract from each chunk sequentially (for visibility)
    all_theses = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Extracting chunk {i} ---")
        theses = await extract_chunk(transcript, chunk, enriched)
        print(f"  → {len(theses)} claims")

        for j, t in enumerate(theses):
            print(f"  [{len(all_theses) + j}] ({t.topic}) {t.speakers[0] if t.speakers else '?'}: "
                  f"{t.thesis_statement[:100]}...")
            print(f"       quote: \"{t.original_quote[:80]}...\"")

        all_theses.extend(theses)

    print(f"\n{'='*80}")
    print(f"PHASE 1 COMPLETE: {len(all_theses)} total claims")
    print(f"{'='*80}")

    # Group by speaker for summary
    by_speaker: dict[str, int] = {}
    for t in all_theses:
        speaker = t.speakers[0] if t.speakers else "Unknown"
        by_speaker[speaker] = by_speaker.get(speaker, 0) + 1
    for speaker, count in sorted(by_speaker.items(), key=lambda x: -x[1]):
        print(f"  {speaker}: {count} claims")

    # Save for Phase 2
    output = []
    for t in all_theses:
        output.append({
            "thesis_statement": t.thesis_statement,
            "speakers": t.speakers,
            "original_quote": t.original_quote,
            "topic": t.topic,
        })

    with open(PHASE1_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nPhase 1 output saved to {PHASE1_OUTPUT}")

    return output


PHASE2_OUTPUT = "tests/artifacts/fii_phase2_output.json"


async def run_phase2(phase1_data=None):
    """Phase 2: Review + group claims per speaker (sequential batches)."""
    from collections import Counter
    from datetime import date
    from src.schemas.llm_outputs import ExtractedThesis
    from src.transcript.claim_reviewer import (
        review_batch, make_trivial_review, REVIEW_BATCH_SIZE,
    )

    if phase1_data is None:
        if not os.path.exists(PHASE1_OUTPUT):
            print(f"ERROR: No Phase 1 output at {PHASE1_OUTPUT}")
            print("Run with --phase1 first, or run without flags for both phases.")
            sys.exit(1)
        with open(PHASE1_OUTPUT) as f:
            phase1_data = json.load(f)

    print(f"\n{'='*80}")
    print(f"PHASE 2: Reviewing {len(phase1_data)} claims")
    print(f"{'='*80}")

    current_date = date.today().isoformat()

    # Group by speaker
    speaker_theses: dict[str, list[dict]] = {}
    for t in phase1_data:
        speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
        speaker_theses.setdefault(speaker, []).append(t)

    for speaker, claims in speaker_theses.items():
        print(f"\n  {speaker}: {len(claims)} claims")

    all_speaker_results = {}

    for speaker, sp_theses in speaker_theses.items():
        # Reconstruct ExtractedThesis objects
        extracted = []
        for t in sp_theses:
            extracted.append(ExtractedThesis(
                thesis_statement=t["thesis_statement"],
                speakers=t.get("speakers", [speaker]),
                original_quote=t.get("original_quote", ""),
                topic=t.get("topic", ""),
            ))

        if len(extracted) <= 1:
            print(f"\n--- {speaker}: 1 claim, trivial group ---")
            review = make_trivial_review(sp_theses[0])
            all_speaker_results[speaker] = review
            continue

        # Sequential batch review (mirrors ReviewClaimsWorkflow logic)
        total = len(extracted)
        batch_count = (total + REVIEW_BATCH_SIZE - 1) // REVIEW_BATCH_SIZE
        print(f"\n--- {speaker}: {total} claims → {batch_count} batches of ~{REVIEW_BATCH_SIZE} ---")

        all_dispositions: list[dict] = []
        groups: dict[str, dict] = {}

        for batch_start in range(0, total, REVIEW_BATCH_SIZE):
            batch_end = min(batch_start + REVIEW_BATCH_SIZE, total)
            batch_theses = extracted[batch_start:batch_end]
            batch_num = batch_start // REVIEW_BATCH_SIZE
            batch_label = f"b{batch_num}"

            # Build groups summary for context
            groups_summary = {}
            for gid, g in groups.items():
                groups_summary[gid] = {
                    "topic": g["topic"],
                    "member_count": len(g["member_indices"]),
                    "representative_statement": g["representative_statement"],
                }
            existing_group_ids = list(groups.keys())

            print(f"\n  Batch {batch_num+1}/{batch_count}: "
                  f"claims [{batch_start}:{batch_end}], "
                  f"{len(existing_group_ids)} existing groups")

            result = await review_batch(
                theses=batch_theses,
                speaker=speaker,
                current_date=current_date,
                existing_groups=groups_summary,
                existing_group_ids=existing_group_ids,
                batch_label=batch_label,
            )

            # Process dispositions — remap local to global indices
            action_counts: Counter = Counter()

            for disp in result.dispositions:
                global_index = disp.claim_index + batch_start
                global_disp = {
                    "claim_index": global_index,
                    "classification": disp.classification,
                    "action": disp.action,
                    "group_id": disp.group_id,
                    "rationale": disp.rationale,
                }
                all_dispositions.append(global_disp)
                action_counts[disp.action] += 1

                if disp.action == "new_group":
                    ng_def = next(
                        ng for ng in result.new_groups
                        if ng.group_id == disp.group_id
                    )
                    groups[disp.group_id] = {
                        "topic": ng_def.topic,
                        "checkable": ng_def.checkable,
                        "checkability_rationale": ng_def.checkability_rationale,
                        "member_indices": [global_index],
                        "representative_statement": sp_theses[global_index][
                            "thesis_statement"
                        ],
                    }
                elif disp.action == "add_to_group":
                    groups[disp.group_id]["member_indices"].append(global_index)

            print(f"    → {dict(action_counts)}, total groups: {len(groups)}")

        # Print final groups for this speaker
        checkable = sum(1 for g in groups.values() if g.get("checkable"))
        print(f"\n  {speaker} DONE: {len(groups)} groups ({checkable} checkable)")

        for gid, group in groups.items():
            tag = "CHECKABLE" if group.get("checkable") else "NOT CHECKABLE"
            members = group["member_indices"]
            print(f"    {gid} ({tag}) | {group['topic']} | {len(members)} members")
            for idx in members:
                print(f"      [{idx}] \"{sp_theses[idx]['thesis_statement'][:90]}\"")

        all_speaker_results[speaker] = {
            "dispositions": all_dispositions,
            "groups": groups,
        }

    # Save Phase 2 output
    with open(PHASE2_OUTPUT, "w") as f:
        json.dump(all_speaker_results, f, indent=2)
    print(f"\nPhase 2 output saved to {PHASE2_OUTPUT}")

    # Final summary
    total_groups = sum(
        len(r["groups"]) for r in all_speaker_results.values()
        if isinstance(r.get("groups"), dict)
    )
    total_checkable = sum(
        sum(1 for g in r["groups"].values() if g.get("checkable"))
        for r in all_speaker_results.values()
        if isinstance(r.get("groups"), dict)
    )
    total_dispositions = sum(
        len(r["dispositions"]) for r in all_speaker_results.values()
        if isinstance(r.get("dispositions"), list)
    )

    print(f"\n{'='*80}")
    print(f"PHASE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Total dispositions: {total_dispositions}")
    print(f"Total groups: {total_groups}")
    print(f"Checkable groups (→ verification): {total_checkable}")

    return all_speaker_results


async def main():
    args = sys.argv[1:]

    # --limit N: only process first N claims in phase2
    limit = None
    for i, a in enumerate(args):
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])

    if "--phase1" in args:
        await run_phase1()
    elif "--phase2" in args:
        phase1_data = None
        if limit:
            with open(PHASE1_OUTPUT) as f:
                phase1_data = json.load(f)[:limit]
            print(f"(Limited to first {limit} claims)")
        await run_phase2(phase1_data)
    else:
        # Run both
        phase1_data = await run_phase1()
        await run_phase2(phase1_data)


if __name__ == "__main__":
    asyncio.run(main())
