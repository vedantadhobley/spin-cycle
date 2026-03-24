"""Temporal activities for transcript extraction.

Activities:
  1. fetch_transcript              — fetch + parse a Rev.com transcript
  2. extract_transcript_batch      — extract claims from one batch of segments
  3. finalize_extraction           — filter + deduplicate claims across all batches
  4. store_transcript              — persist cleaned transcript to DB
  5. store_transcript_claims       — persist extracted claims linked to transcript
  6. create_claims_for_transcript  — batch-create Claim records + link FKs
  7. update_transcript_status      — set transcript status field
  8. finish_transcript_and_start_next — mark transcript complete, start next queued

Each batch is a separate activity so it's visible in Temporal UI.
The workflow orchestrates batches — Temporal's max_concurrent_activities
naturally limits GPU contention.
"""

import uuid as _uuid_mod

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
            "argument_summary": c.argument_summary,
            "supports_argument": c.supports_argument,
            "checkable": c.checkable,
            "checkability_rationale": c.checkability_rationale,
            "consequence_if_wrong": c.consequence_if_wrong,
            "consequence_rationale": c.consequence_rationale,
            "segment_gist": getattr(c, "_segment_gist", None),
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
) -> dict:
    """Filter + deduplicate claims from all batches into final claim list.

    Runs after all batch activities complete.  Applies consistency enforcement,
    filters to worth_checking, deduplicates across batch boundaries, and
    converts to the final TranscriptClaim format.

    Returns dict with:
        - worth_checking: list of dicts for verification pipeline
        - all_claims: list of ALL claims (including skipped) with full metadata for DB storage
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

    # Build timestamp → seconds lookup
    ts_lookup: dict[str, float] = {}
    for seg in transcript.segments:
        ts_lookup[seg.timestamp] = seg.timestamp_secs

    # Reconstruct ExtractedClaim objects from all batches
    all_claims: list[ExtractedClaim] = []
    # Keep raw batch data for full metadata preservation
    all_raw_claims: list[dict] = []
    for batch_claims in all_batch_claims:
        for c in batch_claims:
            all_raw_claims.append(c)
            all_claims.append(ExtractedClaim(
                claim_text=c["claim_text"],
                original_quote=c["original_quote"],
                speaker=c["speaker"],
                timestamp=c["timestamp"],
                claim_type=c["claim_type"],
                worth_checking=c["worth_checking"],
                skip_reason=c.get("skip_reason"),
                argument_summary=c.get("argument_summary"),
                supports_argument=c.get("supports_argument", False),
                checkable=c.get("checkable", c["worth_checking"]),
                checkability_rationale=c.get("checkability_rationale", ""),
                consequence_if_wrong=c.get("consequence_if_wrong", "high" if c["worth_checking"] else "low"),
                consequence_rationale=c.get("consequence_rationale", ""),
            ))

    log.info(activity.logger, "transcript", "finalize_start",
             "Finalizing extraction",
             total_claims=len(all_claims),
             batch_count=len(all_batch_claims))

    final_claims = finalize_claims(all_claims, transcript)

    worth_checking = [
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

    # Build all_claims list with full metadata for DB storage
    all_claims_for_storage = [
        {
            "claim_text": c["claim_text"],
            "original_quote": c["original_quote"],
            "speaker": c["speaker"],
            "timestamp": c["timestamp"],
            "timestamp_secs": ts_lookup.get(c["timestamp"], 0.0),
            "claim_type": c.get("claim_type"),
            "worth_checking": c.get("worth_checking", True),
            "skip_reason": c.get("skip_reason"),
            "argument_summary": c.get("argument_summary"),
            "supports_argument": c.get("supports_argument"),
            "checkable": c.get("checkable"),
            "checkability_rationale": c.get("checkability_rationale"),
            "consequence_if_wrong": c.get("consequence_if_wrong"),
            "consequence_rationale": c.get("consequence_rationale"),
            "segment_gist": c.get("segment_gist"),
        }
        for c in all_raw_claims
    ]

    log.info(activity.logger, "transcript", "finalize_done",
             "Extraction finalized",
             input_claims=len(all_claims),
             worth_checking=len(worth_checking),
             all_for_storage=len(all_claims_for_storage))

    return {
        "worth_checking": worth_checking,
        "all_claims": all_claims_for_storage,
    }


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
    log.info(activity.logger, "transcript", "store_start", "Storing transcript",
             url=url, title=transcript_data.get("title"))

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
            record.status = "extracting"
        else:
            record = TranscriptRecord(
                url=url,
                title=transcript_data["title"],
                date=transcript_data.get("date"),
                speakers=transcript_data["speakers"],
                word_count=transcript_data["word_count"],
                segment_count=len(transcript_data["segments"]),
                display_text=transcript_data["display_text"],
                status="extracting",
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
) -> list[str]:
    """Persist extracted claims linked to their transcript.

    Deletes existing claims for this transcript (re-extraction replaces old results)
    and inserts the new set. Returns list of transcript_claim IDs (insertion order).
    """
    from sqlalchemy import delete
    from src.db.session import async_session
    from src.db.models import TranscriptClaim

    tid = _uuid_mod.UUID(transcript_id)
    tc_ids: list[str] = []

    async with async_session() as session:
        # Clear old claims for this transcript (idempotent re-runs)
        await session.execute(
            delete(TranscriptClaim).where(TranscriptClaim.transcript_id == tid)
        )

        for c in claims:
            tc = TranscriptClaim(
                transcript_id=tid,
                claim_text=c["claim_text"],
                original_quote=c["original_quote"],
                speaker=c["speaker"],
                timestamp=c["timestamp"],
                timestamp_secs=c["timestamp_secs"],
                claim_type=c.get("claim_type"),
                worth_checking=c.get("worth_checking", True),
                skip_reason=c.get("skip_reason"),
                argument_summary=c.get("argument_summary"),
                supports_argument=c.get("supports_argument"),
                checkable=c.get("checkable"),
                checkability_rationale=c.get("checkability_rationale"),
                consequence_if_wrong=c.get("consequence_if_wrong"),
                consequence_rationale=c.get("consequence_rationale"),
                segment_gist=c.get("segment_gist"),
            )
            session.add(tc)
            await session.flush()
            tc_ids.append(str(tc.id))

        await session.commit()

    log.info(activity.logger, "transcript", "claims_stored",
             "Transcript claims stored",
             transcript_id=transcript_id, claim_count=len(claims))

    return tc_ids


@activity.defn
async def create_claims_for_transcript(
    transcript_id: str,
    transcript_claim_ids: list[str],
    claims: list[dict],
    transcript_date: str | None = None,
) -> list[str]:
    """Batch-create Claim records and link them to TranscriptClaims via FK.

    Single transaction: creates all Claim records with status="queued",
    sets speaker/source_url/claim_date, and writes claim_id back to each
    TranscriptClaim. Returns list of claim_id strings (insertion order).
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import Claim, TranscriptClaim

    claim_ids: list[str] = []
    log.info(activity.logger, "transcript", "create_claims_start",
             "Creating Claim records for verification",
             transcript_id=transcript_id, claim_count=len(claims))

    async with async_session() as session:
        async with session.begin():
            for tc_id_str, claim_data in zip(transcript_claim_ids, claims):
                # Create the Claim record
                claim = Claim(
                    text=claim_data["claim_text"],
                    speaker=claim_data.get("speaker"),
                    source_url=claim_data.get("source_url"),
                    claim_date=transcript_date,
                    status="queued",
                )
                session.add(claim)
                await session.flush()
                claim_ids.append(str(claim.id))

                # Link TranscriptClaim → Claim
                tc_id = _uuid_mod.UUID(tc_id_str)
                result = await session.execute(
                    select(TranscriptClaim).where(TranscriptClaim.id == tc_id)
                )
                tc = result.scalar_one()
                tc.claim_id = claim.id

    log.info(activity.logger, "transcript", "claims_created",
             "Batch-created Claim records and linked FKs",
             transcript_id=transcript_id, claim_count=len(claim_ids))

    return claim_ids


@activity.defn
async def update_transcript_status(transcript_id: str, status: str) -> None:
    """Update a transcript's status field."""
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord

    tid = _uuid_mod.UUID(transcript_id)
    log.info(activity.logger, "transcript", "status_update",
             "Updating transcript status",
             transcript_id=transcript_id, status=status)

    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.id == tid)
        )
        record = result.scalar_one()
        record.status = status
        await session.commit()

    log.info(activity.logger, "transcript", "status_updated",
             "Transcript status updated",
             transcript_id=transcript_id, status=status)


@activity.defn
async def finish_transcript_and_start_next() -> str | None:
    """Mark completed transcripts and start the next queued one.

    1. Find transcripts with status='verifying' where ALL linked claims are verified
    2. Mark them 'complete'
    3. Find oldest 'queued' transcript and start its ExtractTranscriptWorkflow
    4. Return transcript_id if started, None if pipeline is idle
    """
    import os
    from sqlalchemy import select, func
    from temporalio.client import Client as TemporalClient
    from src.db.session import async_session
    from src.db.models import TranscriptRecord, TranscriptClaim, Claim

    TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
    TASK_QUEUE = "spin-cycle-verify"

    async with async_session() as session:
        # Step 1: Find verifying transcripts where all claims are done
        verifying = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.status == "verifying")
        )
        for transcript in verifying.scalars().all():
            # Count total linked claims vs verified claims
            total = await session.execute(
                select(func.count()).select_from(TranscriptClaim)
                .where(TranscriptClaim.transcript_id == transcript.id)
                .where(TranscriptClaim.claim_id.isnot(None))
            )
            total_count = total.scalar()

            verified = await session.execute(
                select(func.count()).select_from(TranscriptClaim)
                .join(Claim, TranscriptClaim.claim_id == Claim.id)
                .where(TranscriptClaim.transcript_id == transcript.id)
                .where(Claim.status == "verified")
            )
            verified_count = verified.scalar()

            if total_count > 0 and total_count == verified_count:
                transcript.status = "complete"
                log.info(activity.logger, "transcript", "complete",
                         "Transcript verification complete",
                         transcript_id=str(transcript.id),
                         verified_claims=verified_count)

        await session.commit()

    # Step 2: Find next queued transcript
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord)
            .where(TranscriptRecord.status == "queued")
            .order_by(TranscriptRecord.created_at.asc())
            .with_for_update()
            .limit(1)
        )
        queued = result.scalar_one_or_none()

        if not queued:
            log.info(activity.logger, "transcript", "queue_empty",
                     "No queued transcripts")
            return None

        transcript_id = str(queued.id)
        url = queued.url
        queued.status = "extracting"
        await session.commit()

    # Step 3: Start extraction workflow
    from src.workflows.extract_transcript import ExtractTranscriptWorkflow

    temporal = await TemporalClient.connect(TEMPORAL_HOST)
    await temporal.start_workflow(
        ExtractTranscriptWorkflow.run,
        args=[url],
        id=f"extract-{transcript_id}",
        task_queue=TASK_QUEUE,
    )

    log.info(activity.logger, "transcript", "next_started",
             "Started next queued transcript",
             transcript_id=transcript_id, url=url)

    return transcript_id
