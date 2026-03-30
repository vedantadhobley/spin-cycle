"""Temporal activities for transcript extraction.

Activities:
  1. fetch_transcript              — fetch + parse a C-SPAN transcript (Playwright WAF)
  2. fetch_raw_transcript          — parse raw text into TranscriptData
  3. extract_chunk_activity        — extract claims from one chunk of transcript
  4. classify_claims_activity      — batch LLM classification (checkability)
  5. dedup_claims_activity         — embedding-based dedup per speaker (Phase 2)
  6. synthesize_claim_activity     — synthesize overarching claim per group
  7. store_transcript              — persist cleaned transcript to DB
  8. store_transcript_claims       — persist extracted claims linked to transcript
  9. create_claims_for_transcript  — batch-create Claim records + link FKs
 10. update_transcript_status      — set transcript status field
 11. finish_transcript_and_start_next — mark transcript complete, start next queued

Each chunk is a separate activity so it's visible in Temporal UI.
The workflow orchestrates chunks — Temporal's max_concurrent_activities
naturally limits GPU contention.
"""

import uuid as _uuid_mod

from temporalio import activity

from src.utils.logging import log


@activity.defn
async def fetch_transcript(url: str) -> dict:
    """Fetch and parse a transcript from C-SPAN.

    Uses Playwright to solve the CloudFront WAF JS challenge, then fetches
    program metadata and structured transcript JSON.

    Returns a serialized TranscriptData-shaped dict with speaker turns.
    """
    from src.transcript.cspan import fetch_cspan_transcript, is_cspan_url

    log.info(activity.logger, "transcript", "fetch_start", "Fetching transcript",
             url=url)

    if is_cspan_url(url):
        td = await fetch_cspan_transcript(url)
    else:
        raise ValueError(f"Unsupported transcript URL: {url}")

    result = {
        "url": td.url,
        "title": td.title,
        "date": td.date,
        "speakers": td.speakers,
        "word_count": td.word_count,
        "display_text": td.display_text,
        "source_format": td.source_format,
        "speaker_aliases": td.speaker_aliases,
        "turns": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "section_header": t.section_header,
            }
            for t in td.turns
        ],
    }

    log.info(activity.logger, "transcript", "fetch_done", "Transcript fetched",
             url=url, title=td.title,
             word_count=td.word_count,
             turn_count=td.turn_count,
             speaker_count=len(td.speakers))

    return result


@activity.defn
async def fetch_raw_transcript(
    content: str,
    url: str = "",
    title: str = "",
    date: str | None = None,
) -> dict:
    """Parse raw text content into a TranscriptData structure.

    Returns a serialized transcript dict compatible with the thesis workflow.
    """
    from src.transcript.parsers.raw_text import parse_raw_text

    log.info(activity.logger, "transcript", "parse_raw_start",
             "Parsing raw text transcript",
             title=title, content_length=len(content))

    td = parse_raw_text(content, url=url, title=title, date=date)

    result = {
        "url": td.url,
        "title": td.title,
        "date": td.date,
        "speakers": td.speakers,
        "word_count": td.word_count,
        "display_text": td.display_text,
        "source_format": td.source_format,
        "speaker_aliases": td.speaker_aliases,
        "turns": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "section_header": t.section_header,
            }
            for t in td.turns
        ],
    }

    log.info(activity.logger, "transcript", "parse_raw_done",
             "Raw text parsed",
             title=td.title,
             word_count=td.word_count,
             turn_count=td.turn_count,
             speaker_count=len(td.speakers))

    return result


@activity.defn
async def extract_chunk_activity(
    transcript_data: dict,
    chunk_dict: dict,
    enriched_speakers: list[dict],
) -> list[dict]:
    """Extract claims from a single chunk of a transcript.

    Takes serialized TranscriptData, Chunk dict, and enriched speakers.
    Returns list of thesis dicts with original_quote.
    """
    from src.transcript.thesis_extractor import Chunk, extract_chunk

    # Reconstruct TranscriptData
    td = _reconstruct_transcript_data(transcript_data)
    chunk = Chunk(**chunk_dict)

    log.info(activity.logger, "transcript", "chunk_extraction_start",
             "Starting chunk extraction",
             title=td.title,
             chunk_index=chunk.chunk_index,
             total_chunks=chunk.total_chunks)

    theses = await extract_chunk(td, chunk, enriched_speakers)

    # Serialize for Temporal transport
    result = _serialize_theses(theses)

    log.info(activity.logger, "transcript", "chunk_extraction_done",
             "Chunk extraction complete",
             chunk_index=chunk.chunk_index,
             thesis_count=len(result))

    return result


@activity.defn
async def dedup_claims_activity(
    claims: list[dict],
    speaker: str,
) -> dict:
    """Deduplicate claims for one speaker using embedding similarity.

    Returns dict with "clusters" list — each cluster has a representative,
    member_indices, checkable flag, and topic.
    """
    from src.transcript.claim_dedup import dedup_speaker_claims

    log.info(activity.logger, "transcript", "dedup_start",
             "Starting embedding dedup",
             speaker=speaker, claim_count=len(claims))

    result = await dedup_speaker_claims(claims, speaker)

    cluster_count = len(result["clusters"])
    multi = sum(1 for c in result["clusters"] if len(c["member_indices"]) > 1)

    log.info(activity.logger, "transcript", "dedup_done",
             "Embedding dedup complete",
             speaker=speaker,
             cluster_count=cluster_count,
             multi_member=multi)

    return result


@activity.defn
async def classify_claims_activity(claims: list[dict]) -> list[dict]:
    """Classify a batch of claims. Returns claims with classification fields added."""
    from src.transcript.claim_classifier import classify_claims_batch

    log.info(activity.logger, "transcript", "classify_start",
             "Starting claim classification",
             claim_count=len(claims))

    result = await classify_claims_batch(claims)

    log.info(activity.logger, "transcript", "classify_done",
             "Claim classification complete",
             claim_count=len(result))

    return result


@activity.defn
async def synthesize_claim_activity(
    member_statements: list[str],
    topic: str,
    speaker: str,
) -> dict:
    """Synthesize an overarching claim for one group.

    Returns dict with overarching_claim and rationale.
    """
    from src.transcript.claim_synthesizer import synthesize_group_claim

    log.info(activity.logger, "transcript", "synthesize_start",
             "Starting claim synthesis",
             speaker=speaker, topic=topic,
             member_count=len(member_statements))

    output = await synthesize_group_claim(member_statements, topic, speaker)

    log.info(activity.logger, "transcript", "synthesize_done",
             "Claim synthesis complete",
             speaker=speaker, topic=topic,
             claim_preview=output.overarching_claim[:80])

    return output.model_dump()


def _reconstruct_transcript_data(transcript_data: dict):
    """Reconstruct TranscriptData from a serialized dict."""
    from src.transcript.parsers import TranscriptData, SpeakerTurn

    turns = [
        SpeakerTurn(
            speaker=t["speaker"],
            text=t["text"],
            section_header=t.get("section_header"),
        )
        for t in transcript_data["turns"]
    ]
    return TranscriptData(
        url=transcript_data["url"],
        title=transcript_data["title"],
        date=transcript_data.get("date"),
        speakers=transcript_data["speakers"],
        turns=turns,
        source_format=transcript_data.get("source_format", "revcom"),
        speaker_aliases=transcript_data.get("speaker_aliases", {}),
    )


def _serialize_theses(theses) -> list[dict]:
    """Serialize ExtractedThesis objects for Temporal transport."""
    result = []
    for t in theses:
        result.append({
            "thesis_statement": t.thesis_statement,
            "speakers": t.speakers,
            "original_quote": t.original_quote,
            "topic": t.topic,
        })
    return result


@activity.defn
async def store_transcript(transcript_data: dict) -> dict:
    """Persist cleaned transcript to the database.

    Upserts by URL — if the transcript already exists, updates it.
    Enriches speaker names with Wikidata descriptions before storing.
    Returns dict with transcript_id and enriched speakers.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord
    from src.transcript.speakers import _enrich_speakers

    url = transcript_data["url"]
    log.info(activity.logger, "transcript", "store_start", "Storing transcript",
             url=url, title=transcript_data.get("title"))

    # Enrich speakers with Wikidata descriptions before storing
    raw_speakers = transcript_data.get("speakers", [])
    enriched_speakers = await _enrich_speakers(raw_speakers)
    transcript_data["speakers"] = enriched_speakers

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
            record.segment_count = len(transcript_data["turns"])
            record.display_text = transcript_data["display_text"]
            record.status = "extracting"
            # v2 fields
            if "turns" in transcript_data:
                record.segments_data = transcript_data["turns"]
            if "source_format" in transcript_data:
                record.source_format = transcript_data["source_format"]
            if "speaker_aliases" in transcript_data:
                record.speaker_aliases = transcript_data["speaker_aliases"]
        else:
            record = TranscriptRecord(
                url=url,
                title=transcript_data["title"],
                date=transcript_data.get("date"),
                speakers=transcript_data["speakers"],
                word_count=transcript_data["word_count"],
                segment_count=len(transcript_data["turns"]),
                display_text=transcript_data["display_text"],
                status="extracting",
                # v2 fields
                segments_data=transcript_data.get("turns"),
                source_format=transcript_data.get("source_format", "revcom"),
                speaker_aliases=transcript_data.get("speaker_aliases"),
            )
            session.add(record)

        await session.commit()
        record_id = str(record.id)

    log.info(activity.logger, "transcript", "stored",
             "Transcript stored in database",
             url=url, transcript_id=record_id,
             enriched_speakers=len(enriched_speakers))

    return {"transcript_id": record_id, "speakers": enriched_speakers}


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
                claim_text=c.get("claim_text") or c.get("thesis_statement", ""),
                original_quote=c.get("original_quote", ""),
                speaker=c.get("speaker", c.get("speakers", [""])[0] if c.get("speakers") else ""),
                claim_type=c.get("claim_type"),
                worth_checking=c.get("worth_checking", True),
                skip_reason=c.get("skip_reason"),
                checkable=c.get("checkable"),
                checkability_rationale=c.get("checkability_rationale"),
                is_restatement=c.get("is_restatement", False),
                segment_gist=c.get("segment_gist"),
                topic=c.get("topic"),
                thesis_version=c.get("thesis_version", 3),
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
    groups: list[dict],
    transcript_date: str | None = None,
    transcript_title: str | None = None,
    speaker_descriptions: dict | None = None,
    source_url: str | None = None,
) -> list[str]:
    """Create one Claim record per group, linking member TranscriptClaims.

    Each group maps to one Claim. All member TranscriptClaims point to
    that Claim via claim_id FK (many-to-one).

    Args:
        transcript_id: UUID of the transcript.
        transcript_claim_ids: All TC IDs (indexed by thesis position).
        groups: List of group dicts with member_tc_ids, claim_text, speaker, topic.
        transcript_date: When the transcript was recorded.
        transcript_title: Title of the transcript.
        speaker_descriptions: Speaker name → description mapping.
        source_url: Transcript URL.

    Returns:
        List of Claim IDs (one per group, insertion order).
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import Claim, TranscriptClaim

    speaker_descriptions = speaker_descriptions or {}
    claim_ids: list[str] = []
    log.info(activity.logger, "transcript", "create_claims_start",
             "Creating Claim records for verification (group-based)",
             transcript_id=transcript_id, group_count=len(groups))

    async with async_session() as session:
        async with session.begin():
            for group in groups:
                speaker_name = group.get("speaker")
                # Create one Claim per group
                claim = Claim(
                    text=group["claim_text"],
                    speaker=speaker_name,
                    speaker_description=speaker_descriptions.get(speaker_name, "") if speaker_name else None,
                    source_url=source_url,
                    claim_date=transcript_date,
                    transcript_title=transcript_title,
                    status="queued",
                )
                session.add(claim)
                await session.flush()
                claim_ids.append(str(claim.id))

                # Link all member TranscriptClaims to this Claim
                for tc_id_str in group["member_tc_ids"]:
                    tc_id = _uuid_mod.UUID(tc_id_str)
                    result = await session.execute(
                        select(TranscriptClaim).where(TranscriptClaim.id == tc_id)
                    )
                    tc = result.scalar_one()
                    tc.claim_id = claim.id

    log.info(activity.logger, "transcript", "claims_created",
             "Created Claim records (group-based) and linked FKs",
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

    from src.config import TEMPORAL_HOST, TASK_QUEUE

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


@activity.defn
async def notify_frontend_refresh() -> bool:
    """Notify the vedanta-systems frontend to broadcast a refresh to SSE clients.

    Called after verification or transcript completion. The frontend receives
    a POST and broadcasts a lightweight refresh signal to all connected
    browser clients, which then refetch data via REST.

    Gracefully handles frontend being unavailable (not a fatal error).
    """
    import os
    import httpx

    api_url = os.getenv("FRONTEND_API_URL", "")
    if not api_url:
        log.debug(activity.logger, "notify", "skip",
                  "FRONTEND_API_URL not set, skipping refresh notification")
        return False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{api_url}/api/spin-cycle/refresh")
            if resp.is_success:
                result = resp.json()
                log.info(activity.logger, "notify", "frontend_notified",
                         "Frontend notified",
                         clients=result.get("clientsNotified", 0))
                return True
            else:
                log.warning(activity.logger, "notify", "frontend_notify_failed",
                            "Frontend notify failed",
                            status_code=resp.status_code)
                return False
    except httpx.ConnectError:
        log.debug(activity.logger, "notify", "frontend_unavailable",
                  "Frontend API not available")
        return False
    except Exception as e:
        log.warning(activity.logger, "notify", "frontend_notify_error",
                    "Frontend notify error", error=str(e))
        return False
