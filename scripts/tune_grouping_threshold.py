#!/usr/bin/env python3
"""Sweep cosine similarity thresholds for sentence grouping.

Two modes:
  1. Fast (default): sentencize → embed raw sentences → sweep thresholds
     No LLM calls — runs in ~30 seconds. Good enough for threshold tuning
     since raw and decontextualized text have similar embedding neighborhoods.

  2. Full (--decontext): sentencize → decontextualize via LLM → embed → sweep
     Takes ~10 minutes. Use for final validation after picking a threshold.

Run inside Docker:
  docker exec spin-cycle-dev-worker python scripts/tune_grouping_threshold.py [transcript_id]
  docker exec spin-cycle-dev-worker python scripts/tune_grouping_threshold.py --detail 0.35 [transcript_id]
  docker exec spin-cycle-dev-worker python scripts/tune_grouping_threshold.py --decontext [transcript_id]

Flags:
  --decontext     Run LLM decontextualization before embedding (slow, cached)
  --detail THRESH Print every group at a specific threshold
  --bridge N      Override bridge_max_words (default: 10)
  --no-cache      Skip loading cached embeddings

If no transcript_id is given, uses the most recent transcript.
Decontextualized embeddings are cached to /tmp/grouping_cache_<id>.npz.
"""

import asyncio
import sys
import os
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    args = {
        "decontext": False,
        "detail": None,
        "bridge": 10,
        "transcript_id": None,
        "no_cache": False,
    }
    positional = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--decontext":
            args["decontext"] = True
        elif sys.argv[i] == "--detail":
            i += 1
            args["detail"] = float(sys.argv[i])
        elif sys.argv[i] == "--bridge":
            i += 1
            args["bridge"] = int(sys.argv[i])
        elif sys.argv[i] == "--no-cache":
            args["no_cache"] = True
        else:
            positional.append(sys.argv[i])
        i += 1
    if positional:
        args["transcript_id"] = positional[0]
    return args


def load_transcript(transcript_id: str | None):
    """Load transcript data from DB using sync session."""
    from sqlalchemy import text
    from src.db.session import get_sync_session
    from src.transcript.parsers import SpeakerTurn

    with get_sync_session() as session:
        if transcript_id:
            row = session.execute(
                text("SELECT id, title, date, description, url, source_format, "
                     "segments_data, enriched_speakers, speakers "
                     "FROM transcripts WHERE id = :id"),
                {"id": transcript_id},
            ).fetchone()
        else:
            row = session.execute(
                text("SELECT id, title, date, description, url, source_format, "
                     "segments_data, enriched_speakers, speakers "
                     "FROM transcripts ORDER BY created_at DESC LIMIT 1"),
            ).fetchone()

        if not row:
            return None, None, None

        meta = {
            "id": str(row.id),
            "title": row.title,
            "date": row.date,
            "description": row.description,
            "url": row.url or "",
            "source_format": row.source_format or "rev",
            "speakers": row.speakers or [],
        }

        segments = row.segments_data or []
        turns = [
            SpeakerTurn(
                speaker=t["speaker"],
                text=t["text"],
                section_header=t.get("section_header"),
            )
            for t in segments
        ]

        enriched = row.enriched_speakers or [{"name": s} for s in meta["speakers"]]

        return meta, turns, enriched


async def main():
    args = parse_args()

    from src.transcript.thesis_extractor import (
        sentencize_transcript, build_sentence_chunks,
        NumberedSentence,
    )
    from src.transcript.parsers import TranscriptData, SpeakerTurn
    from src.transcript.sentence_grouper import group_sentences
    from src.llm.embeddings import embed_texts
    from src.utils.logging import configure_logging, get_logger

    configure_logging()
    logger = get_logger()

    meta, turns, enriched_speakers = load_transcript(args["transcript_id"])
    if not meta:
        print("No transcript found")
        return

    print(f"Transcript: {meta['title']} ({meta['id']})")

    if not turns:
        print("No turns stored (segments_data is empty)")
        return

    # Sentencize
    sentences = sentencize_transcript(turns)
    print(f"Sentences: {len(sentences)}")
    speakers = sorted({s.speaker for s in sentences})
    print(f"Speakers: {', '.join(speakers)}")

    # Build ordered sentence list
    ordered = []
    for s in sentences:
        ordered.append({
            "global_index": s.global_index,
            "speaker": s.speaker,
            "text": s.text,
            "standalone": s.text,  # overwritten if --decontext
        })

    # Check for cached embeddings first
    cache_suffix = "_decontext" if args["decontext"] else "_raw"
    cache_path = f"/tmp/grouping_cache_{meta['id']}{cache_suffix}.npz"

    if not args["no_cache"] and os.path.exists(cache_path):
        print(f"\nLoading cache from {cache_path}...", end="", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = cached["embeddings"]
        if "standalone" in cached:
            cached_standalone = json.loads(str(cached["standalone"]))
            for entry in ordered:
                idx = entry["global_index"]
                if str(idx) in cached_standalone:
                    entry["standalone"] = cached_standalone[str(idx)]
        print(f" shape {embeddings.shape}")
    else:
        # Run decontextualization if requested
        if args["decontext"]:
            from src.transcript.thesis_extractor import decontextualize_chunk

            td = TranscriptData(
                url=meta["url"],
                title=meta["title"],
                date=meta.get("date"),
                description=meta.get("description"),
                speakers=speakers,
                turns=turns,
                source_format=meta["source_format"],
            )

            chunks = build_sentence_chunks(sentences, logger=logger)
            print(f"\nDecontextualizing {len(chunks)} chunks...")

            standalone_map = {}
            for chunk in chunks:
                print(f"  Chunk {chunk.chunk_index + 1}/{chunk.total_chunks} "
                      f"(S{chunk.target_range[0]}-S{chunk.target_range[1] - 1})...",
                      end="", flush=True)
                output = await decontextualize_chunk(
                    td, chunk, enriched_speakers, logger=logger,
                )
                for s in output.sentences:
                    standalone_map[s.index] = s.standalone
                print(f" {len(output.sentences)} sentences")

            for entry in ordered:
                dc = standalone_map.get(entry["global_index"])
                if dc:
                    entry["standalone"] = dc

            print(f"Decontextualized: {len(standalone_map)}/{len(sentences)}")

        # Embed
        print("\nEmbedding...", end="", flush=True)
        texts = [s["standalone"] for s in ordered]
        embeddings = await embed_texts(texts)
        print(f" shape {embeddings.shape}")

        # Cache for future runs
        standalone_cache = {str(s["global_index"]): s["standalone"] for s in ordered}
        np.savez(
            cache_path,
            embeddings=embeddings,
            standalone=json.dumps(standalone_cache),
        )
        print(f"  Cached to {cache_path}")

    bridge = args["bridge"]

    # Detail mode: show every group at a specific threshold
    if args["detail"] is not None:
        threshold = args["detail"]
        groups = group_sentences(ordered, embeddings, threshold, bridge)
        print(f"\n{'='*80}")
        print(f"Threshold {threshold:.2f} — {len(groups)} groups (bridge={bridge})")
        print(f"{'='*80}\n")

        for g in groups:
            indices = g.sentence_indices
            size = len(indices)
            print(f"--- Group {g.group_id} ({size} sentences, {g.speaker}) ---")
            for idx in indices:
                s = ordered[idx]
                text = s["standalone"][:140]
                print(f"  S{idx}: {text}")
            print()

        return

    # Sweep mode
    print(f"\n{'Threshold':>10} | {'Groups':>6} | {'Avg':>5} | {'Max':>4} | "
          f"{'1-sent':>6} | {'2-5':>4} | {'6-15':>5} | {'16+':>4}")
    print("-" * 75)

    for threshold_int in range(20, 75, 5):
        threshold = threshold_int / 100.0
        groups = group_sentences(ordered, embeddings, threshold, bridge)

        sizes = [len(g.sentence_indices) for g in groups]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        singles = sum(1 for s in sizes if s == 1)
        small = sum(1 for s in sizes if 2 <= s <= 5)
        medium = sum(1 for s in sizes if 6 <= s <= 15)
        large = sum(1 for s in sizes if s >= 16)

        print(f"{threshold:>10.2f} | {len(groups):>6} | {avg_size:>5.1f} | "
              f"{max_size:>4} | {singles:>6} | {small:>4} | {medium:>5} | {large:>4}")


if __name__ == "__main__":
    asyncio.run(main())
