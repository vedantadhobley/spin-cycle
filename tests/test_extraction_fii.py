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

    print(f"Parsed: {transcript.segment_count} segments, "
          f"{transcript.word_count} words, "
          f"speakers: {transcript.speakers}")

    # Build chunks
    chunks = build_chunks(transcript.segments)
    print(f"\n{'='*80}")
    print(f"CHUNKING: {len(chunks)} chunks from {transcript.segment_count} segments")
    print(f"{'='*80}")
    for i, c in enumerate(chunks):
        # Count words in target segments
        target_segs = transcript.segments[c.target_start:c.target_end]
        target_words = sum(len(s.text.split()) for s in target_segs)
        print(f"  Chunk {i}: target [{c.target_start}-{c.target_end}) "
              f"({c.target_end - c.target_start} segs, {target_words} words), "
              f"context [{c.context_start}-{c.context_end})")

    # Enrich speakers
    enriched = await _enrich_speakers(transcript.speakers)
    print(f"\nSpeakers enriched: {[s['name'] for s in enriched]}")

    # Extract from each chunk sequentially (for visibility)
    all_theses = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Extracting chunk {i} "
              f"[{chunk.target_start}-{chunk.target_end}) ---")
        theses = await extract_chunk(transcript, chunk, enriched)
        print(f"  → {len(theses)} claims")

        for j, t in enumerate(theses):
            refs = ", ".join(f"[{r.segment_index}]" for r in t.supporting_references)
            print(f"  [{len(all_theses) + j}] ({t.topic}) {t.speakers[0] if t.speakers else '?'}: "
                  f"{t.thesis_statement[:100]}...")
            print(f"       refs: {refs}")

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
            "supporting_references": [
                {"segment_index": r.segment_index, "excerpt": r.excerpt}
                for r in t.supporting_references
            ],
            "topic": t.topic,
        })

    with open(PHASE1_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nPhase 1 output saved to {PHASE1_OUTPUT}")

    return output


async def run_phase2(phase1_data=None):
    """Phase 2: Review + group claims per speaker."""
    from src.transcript.claim_reviewer import review_claims, make_trivial_review
    from src.schemas.llm_outputs import ExtractedThesis, SupportingReference

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

    # Group by speaker
    speaker_theses: dict[str, list[dict]] = {}
    for t in phase1_data:
        speaker = t["speakers"][0] if t.get("speakers") else "Unknown"
        speaker_theses.setdefault(speaker, []).append(t)

    for speaker, claims in speaker_theses.items():
        print(f"\n  {speaker}: {len(claims)} claims")

    # Review each speaker
    all_groups = []
    all_classifications = []

    for speaker, sp_theses in speaker_theses.items():
        # Reconstruct ExtractedThesis objects
        extracted = []
        for t in sp_theses:
            refs = [SupportingReference(**r) for r in t.get("supporting_references", [])]
            extracted.append(ExtractedThesis(
                thesis_statement=t["thesis_statement"],
                speakers=t.get("speakers", [speaker]),
                supporting_references=refs,
                topic=t.get("topic", ""),
            ))

        if len(extracted) <= 1:
            print(f"\n--- {speaker}: 1 claim, trivial group ---")
            review = make_trivial_review(extracted)
        else:
            print(f"\n--- {speaker}: Reviewing {len(extracted)} claims via LLM ---")
            review = await review_claims(extracted, speaker, "2026-03-27")

        # Print classifications
        print(f"\n  CLASSIFICATIONS:")
        verifiable_count = 0
        duplicate_count = 0
        for cls in review.classifications:
            marker = ""
            if cls.is_duplicate:
                marker = f" [DUP of {cls.duplicate_of}]"
                duplicate_count += 1
            if cls.classification == "verifiable_fact" and not cls.is_duplicate:
                verifiable_count += 1
            thesis_text = sp_theses[cls.claim_index]["thesis_statement"][:80]
            print(f"    [{cls.claim_index}] {cls.classification}{marker}")
            print(f"        \"{thesis_text}...\"")
            if cls.classification != "verifiable_fact":
                print(f"        Rationale: {cls.rationale[:100]}")

        print(f"\n  SUMMARY: {verifiable_count} verifiable, "
              f"{duplicate_count} duplicates, "
              f"{len(review.classifications) - verifiable_count - duplicate_count} filtered")

        # Print groups
        print(f"\n  GROUPS ({len(review.groups)}):")
        for gi, group in enumerate(review.groups):
            checkable_marker = "CHECKABLE" if group.checkable else "NOT CHECKABLE"
            print(f"\n    Group {gi} ({checkable_marker}) | Topic: {group.topic}")
            print(f"    Rationale: {group.group_rationale[:120]}")
            print(f"    Checkability: {group.checkability_rationale[:120]}")
            print(f"    Members ({len(group.member_indices)}):")
            for idx in group.member_indices:
                thesis_text = sp_theses[idx]["thesis_statement"][:100]
                print(f"      [{idx}] \"{thesis_text}...\"")

            # Build group text (what would be sent to verification)
            member_statements = [sp_theses[idx]["thesis_statement"] for idx in group.member_indices]
            group_text = "\n".join(member_statements)

            # Collect all refs
            all_refs = set()
            for idx in group.member_indices:
                for ref in sp_theses[idx].get("supporting_references", []):
                    all_refs.add(ref["segment_index"])

            all_groups.append({
                "speaker": speaker,
                "topic": group.topic,
                "checkable": group.checkable,
                "member_count": len(group.member_indices),
                "group_text": group_text,
                "ref_segments": sorted(all_refs),
                "group_rationale": group.group_rationale,
            })

        all_classifications.extend([
            {"speaker": speaker, **cls.model_dump()}
            for cls in review.classifications
        ])

    # Final summary
    checkable_groups = [g for g in all_groups if g["checkable"]]
    print(f"\n{'='*80}")
    print(f"PHASE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Total groups: {len(all_groups)}")
    print(f"Checkable groups (→ verification): {len(checkable_groups)}")
    print(f"Not checkable: {len(all_groups) - len(checkable_groups)}")

    # Classification breakdown
    class_counts: dict[str, int] = {}
    dup_count = 0
    for c in all_classifications:
        class_counts[c["classification"]] = class_counts.get(c["classification"], 0) + 1
        if c["is_duplicate"]:
            dup_count += 1
    print(f"\nClassification breakdown:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print(f"  duplicates: {dup_count}")

    print(f"\n{'='*80}")
    print(f"CLAIMS SUBMITTED TO VERIFICATION PIPELINE ({len(checkable_groups)}):")
    print(f"{'='*80}")
    for i, g in enumerate(checkable_groups, 1):
        print(f"\n--- Verification Claim {i} | {g['speaker']} | {g['topic']} | "
              f"{g['member_count']} member(s) | refs: {g['ref_segments']}")
        print(f"    Rationale: {g['group_rationale'][:120]}")
        print(f"    Text sent to decompose:")
        for line in g["group_text"].split("\n"):
            print(f"      > {line}")

    return all_groups


async def main():
    args = sys.argv[1:]

    if "--phase1" in args:
        await run_phase1()
    elif "--phase2" in args:
        await run_phase2()
    else:
        # Run both
        phase1_data = await run_phase1()
        await run_phase2(phase1_data)


if __name__ == "__main__":
    asyncio.run(main())
