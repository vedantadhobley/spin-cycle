"""Temporal activities for transcript extraction.

Four activities:
  1. fetch_transcript          — fetch + parse a Rev.com transcript
  2. extract_transcript_batch  — extract claims from one batch of segments
  3. finalize_extraction       — filter + deduplicate claims across all batches
  4. store_transcript          — persist cleaned transcript to DB

Each batch is a separate activity so it's visible in Temporal UI.
The workflow orchestrates batches — Temporal's max_concurrent_activities
naturally limits GPU contention.
"""

from temporalio import activity

from src.utils.logging import log


@activity.defn
async def fetch_transcript(url: str) -> dict:
    """Fetch and parse a transcript from Rev.com.

    Returns a serialized transcript dict with segments, metadata, and word count.
    """
    log.info(activity.logger, "transcript", "fetch_start", "Fetching transcript",
             url=url)

    from src.transcript.fetcher import fetch_transcript as _fetch

    transcript = await _fetch(url)

    result = {
        "url": transcript.url,
        "title": transcript.title,
        "date": transcript.date,
        "speakers": transcript.speakers,
        "word_count": transcript.word_count,
        "display_text": transcript.display_text,
        "segments": [
            {
                "speaker": s.speaker,
                "timestamp": s.timestamp,
                "timestamp_secs": s.timestamp_secs,
                "text": s.text,
            }
            for s in transcript.segments
        ],
    }

    log.info(activity.logger, "transcript", "fetch_done", "Transcript fetched",
             url=url, title=transcript.title,
             word_count=transcript.word_count,
             segment_count=len(transcript.segments),
             speaker_count=len(transcript.speakers))

    return result


@activity.defn
async def extract_transcript_batch(
    transcript_data: dict,
    target_start: int,
    target_end: int,
    text_start: int,
    text_end: int,
    batch_label: str,
) -> list[dict]:
    """Extract claims from one batch of transcript segments.

    Takes the full transcript data + indices defining which segments are
    targets (in manifest) vs context-only (overlap for bracket resolution).

    Each batch is a separate Temporal activity for UI visibility.
    """
    from src.transcript.fetcher import TranscriptSegment
    from src.transcript.extractor import extract_batch

    # Reconstruct segment objects
    all_segments = [TranscriptSegment(**s) for s in transcript_data["segments"]]
    text_segments = all_segments[text_start:text_end]
    target_segments = all_segments[target_start:target_end]

    log.info(activity.logger, "transcript", "batch_start",
             "Extracting batch",
             batch_label=batch_label,
             target_count=len(target_segments),
             text_count=len(text_segments),
             overlap=len(text_segments) - len(target_segments))

    claims = await extract_batch(
        text_segments=text_segments,
        target_segments=target_segments,
        batch_label=batch_label,
    )

    result = [
        {
            "claim_text": c.claim_text,
            "original_quote": c.original_quote,
            "speaker": c.speaker,
            "timestamp": c.timestamp,
            "claim_type": c.claim_type,
            "worth_checking": c.worth_checking,
            "skip_reason": c.skip_reason,
        }
        for c in claims
    ]

    worth = sum(1 for c in claims if c.worth_checking)
    log.info(activity.logger, "transcript", "batch_done",
             "Batch extraction complete",
             batch_label=batch_label,
             total_assertions=len(result),
             worth_checking=worth)

    return result


@activity.defn
async def finalize_extraction(
    transcript_data: dict,
    all_batch_claims: list[list[dict]],
) -> list[dict]:
    """Filter + deduplicate claims from all batches into final claim list.

    Runs after all batch activities complete.  Applies consistency enforcement,
    filters to worth_checking, deduplicates across batch boundaries, and
    converts to the final TranscriptClaim format.
    """
    from src.transcript.fetcher import Transcript, TranscriptSegment
    from src.transcript.extractor import (
        ExtractedClaim, finalize_claims,
    )

    # Reconstruct Transcript
    transcript = Transcript(
        url=transcript_data["url"],
        title=transcript_data["title"],
        date=transcript_data.get("date"),
        speakers=transcript_data["speakers"],
        segments=[TranscriptSegment(**s) for s in transcript_data["segments"]],
    )

    # Reconstruct ExtractedClaim objects from all batches
    all_claims: list[ExtractedClaim] = []
    for batch_claims in all_batch_claims:
        for c in batch_claims:
            all_claims.append(ExtractedClaim(
                claim_text=c["claim_text"],
                original_quote=c["original_quote"],
                speaker=c["speaker"],
                timestamp=c["timestamp"],
                claim_type=c["claim_type"],
                worth_checking=c["worth_checking"],
                skip_reason=c.get("skip_reason"),
                # Fields not needed for finalization but required by schema
                supports_argument=False,
                checkable=c["worth_checking"],
                consequence_if_wrong="high" if c["worth_checking"] else "low",
            ))

    log.info(activity.logger, "transcript", "finalize_start",
             "Finalizing extraction",
             total_claims=len(all_claims),
             batch_count=len(all_batch_claims))

    final_claims = finalize_claims(all_claims, transcript)

    result = [
        {
            "claim_text": c.claim_text,
            "original_quote": c.original_quote,
            "speaker": c.speaker,
            "timestamp": c.timestamp,
            "timestamp_secs": c.timestamp_secs,
            "claim_type": c.claim_type,
            "source_url": c.source_url,
        }
        for c in final_claims
    ]

    log.info(activity.logger, "transcript", "finalize_done",
             "Extraction finalized",
             input_claims=len(all_claims),
             output_claims=len(result))

    return result


@activity.defn
async def store_transcript(transcript_data: dict) -> str:
    """Persist cleaned transcript to the database.

    Upserts by URL — if the transcript already exists, updates it.
    Returns the transcript record ID.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord

    url = transcript_data["url"]

    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.url == url)
        )
        record = result.scalar_one_or_none()

        if record:
            record.title = transcript_data["title"]
            record.date = transcript_data.get("date")
            record.speakers = transcript_data["speakers"]
            record.word_count = transcript_data["word_count"]
            record.segment_count = len(transcript_data["segments"])
            record.display_text = transcript_data["display_text"]
        else:
            record = TranscriptRecord(
                url=url,
                title=transcript_data["title"],
                date=transcript_data.get("date"),
                speakers=transcript_data["speakers"],
                word_count=transcript_data["word_count"],
                segment_count=len(transcript_data["segments"]),
                display_text=transcript_data["display_text"],
            )
            session.add(record)

        await session.commit()
        record_id = str(record.id)

    log.info(activity.logger, "transcript", "stored",
             "Transcript stored in database",
             url=url, transcript_id=record_id)

    return record_id


@activity.defn
async def store_transcript_claims(
    transcript_id: str,
    claims: list[dict],
) -> int:
    """Persist extracted claims linked to their transcript.

    Deletes existing claims for this transcript (re-extraction replaces old results)
    and inserts the new set. Returns the number of claims stored.
    """
    import uuid as _uuid
    from sqlalchemy import delete
    from src.db.session import async_session
    from src.db.models import TranscriptClaim

    tid = _uuid.UUID(transcript_id)

    async with async_session() as session:
        # Clear old claims for this transcript (idempotent re-runs)
        await session.execute(
            delete(TranscriptClaim).where(TranscriptClaim.transcript_id == tid)
        )

        for c in claims:
            session.add(TranscriptClaim(
                transcript_id=tid,
                claim_text=c["claim_text"],
                original_quote=c["original_quote"],
                speaker=c["speaker"],
                timestamp=c["timestamp"],
                timestamp_secs=c["timestamp_secs"],
                claim_type=c.get("claim_type"),
            ))

        await session.commit()

    log.info(activity.logger, "transcript", "claims_stored",
             "Transcript claims stored",
             transcript_id=transcript_id, claim_count=len(claims))

    return len(claims)
