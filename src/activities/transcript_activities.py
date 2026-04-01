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
 12. load_extract_inputs           — load DB state for ExtractClaimsWorkflow
 13. load_classify_inputs          — load DB state for ClassifyAndDedupWorkflow
 14. load_synthesize_inputs        — load DB state for SynthesizeClaimsWorkflow
 15. load_verify_inputs            — load DB state for VerifyAllClaimsWorkflow

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
    from src.transcript.rev import fetch_rev_transcript, is_rev_url

    log.info(activity.logger, "fetch", "start", "Fetching transcript",
             url=url)

    try:
        if is_cspan_url(url):
            td = await fetch_cspan_transcript(url)
        elif is_rev_url(url):
            td = await fetch_rev_transcript(url)
        else:
            raise ValueError(f"Unsupported transcript URL: {url}")
    except Exception as e:
        log.error(activity.logger, "fetch", "failed", "Transcript fetch failed",
                  url=url, error=str(e))
        raise

    result = {
        "url": td.url,
        "title": td.title,
        "date": td.date,
        "description": td.description,
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

    log.info(activity.logger, "fetch", "done", "Transcript fetched",
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

    log.info(activity.logger, "fetch", "raw_start",
             "Parsing raw text transcript",
             title=title, content_length=len(content))

    td = parse_raw_text(content, url=url, title=title, date=date)

    result = {
        "url": td.url,
        "title": td.title,
        "date": td.date,
        "description": td.description,
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

    log.info(activity.logger, "fetch", "raw_done",
             "Raw text parsed",
             title=td.title,
             word_count=td.word_count,
             turn_count=td.turn_count,
             speaker_count=len(td.speakers))

    return result


def _normalize_transcript(transcript_data: dict) -> dict:
    """Merge consecutive same-speaker turns and update derived fields.

    This is the single normalization point for all transcript formats.
    Parsers return raw turns; this runs after any LLM attribution.
    """
    from src.transcript.parsers import clean_speaker_name

    turns = transcript_data.get("turns", [])
    if not turns:
        return transcript_data

    # Clean speaker names (strip quoted nicknames, etc.) before merging
    for t in turns:
        if t.get("speaker"):
            t["speaker"] = clean_speaker_name(t["speaker"])

    merged: list[dict] = [dict(turns[0])]  # shallow copy first turn
    for t in turns[1:]:
        prev = merged[-1]
        if (t.get("speaker") == prev.get("speaker")
                and t.get("section_header") is None):
            prev["text"] = prev["text"] + "\n\n" + t["text"]
        else:
            merged.append(dict(t))

    transcript_data["turns"] = merged
    transcript_data["speakers"] = list(dict.fromkeys(
        t.get("speaker", "") for t in merged
    ))
    transcript_data["display_text"] = "\n\n".join(
        f"{t.get('speaker', 'Unknown')}: {t.get('text', '')}" for t in merged
    )
    transcript_data["word_count"] = sum(
        len(t.get("text", "").split()) for t in merged
    )
    return transcript_data


@activity.defn
async def attribute_speakers(transcript_data: dict) -> dict:
    """Attribute Unknown speaker turns using an LLM.

    Receives the transcript_data dict from fetch_transcript, calls the LLM
    to attribute any "Unknown" speaker turns based on content signals, then
    returns the modified transcript_data with speakers filled in.

    If no Unknown turns exist (e.g. Rev.com transcripts), returns unchanged.
    """
    turns = transcript_data.get("turns", [])

    # Collect Unknown turn indices
    unknown_indices = [
        i for i, t in enumerate(turns) if t.get("speaker") == "Unknown"
    ]

    if not unknown_indices:
        log.info(activity.logger, "attribution", "skip",
                 "No Unknown turns — skipping speaker attribution")
        return _normalize_transcript(transcript_data)

    log.info(activity.logger, "attribution", "start",
             "Attributing unknown speaker turns",
             unknown_count=len(unknown_indices),
             total_turns=len(turns))

    from src.llm.invoker import invoke_llm
    from src.schemas.llm_outputs import AttributeSpeakersOutput
    from src.llm.validators import validate_speaker_attribution
    from src.prompts.speaker_attribution import (
        SPEAKER_ATTRIBUTION_SYSTEM,
        SPEAKER_ATTRIBUTION_USER,
    )

    # Build known speaker list from named turns + transcript-level speakers
    # (C-SPAN extracts person names from HTML even when all cc_names are ">>")
    skip = {"Unknown", "Narrator"}
    known_speakers: set[str] = set()
    for t in turns:
        speaker = t.get("speaker", "")
        if speaker and speaker not in skip:
            known_speakers.add(speaker)
    for s in transcript_data.get("speakers", []):
        if isinstance(s, str) and s and s not in skip:
            known_speakers.add(s)

    speaker_list = ", ".join(sorted(known_speakers)) if known_speakers else "(none identified)"

    # Build turn list string — all turns, truncated text for long ones
    turn_lines = []
    for i, t in enumerate(turns):
        text = t.get("text", "")
        total_len = len(text)
        if total_len > 300:
            display_text = text[:300].rstrip() + f"... [{total_len} chars total]"
        else:
            display_text = text
        turn_lines.append(f"[{i}] {t.get('speaker', 'Unknown')}: {display_text}")

    turn_list = "\n".join(turn_lines)

    # Build a validator that knows the allowed speaker names
    def _validator(output: AttributeSpeakersOutput) -> tuple[bool, str]:
        return validate_speaker_attribution(output, known_speakers=known_speakers)

    output = await invoke_llm(
        system_prompt=SPEAKER_ATTRIBUTION_SYSTEM,
        user_prompt=SPEAKER_ATTRIBUTION_USER.format(
            title=transcript_data.get("title", ""),
            date=transcript_data.get("date", "unknown"),
            description=transcript_data.get("description", "") or "",
            speaker_list=speaker_list,
            turn_list=turn_list,
        ),
        schema=AttributeSpeakersOutput,
        semantic_validator=_validator,
        max_retries=2,
        max_tokens=4096,
        activity_name="attribute_speakers",
    )

    # Apply attributions (skip "Unknown" — LLM saying "I can't tell")
    attributed_indices: set[int] = set()
    for attr in output.attributions:
        idx = attr.turn_index
        if 0 <= idx < len(turns) and turns[idx].get("speaker") == "Unknown":
            if attr.speaker != "Unknown":
                turns[idx]["speaker"] = attr.speaker
            attributed_indices.add(idx)

    # Any Unknown turns the LLM didn't attribute stay as "Unknown"
    missed = [i for i in unknown_indices if i not in attributed_indices]
    if missed:
        log.info(activity.logger, "attribution", "unattributed_turns",
                 "Some turns remain Unknown (LLM could not identify speaker)",
                 count=len(missed))

    log.info(activity.logger, "attribution", "done",
             "Speaker attribution complete",
             format_type=output.format_type,
             attributed=len(attributed_indices),
             missed=len(missed))

    return _normalize_transcript(transcript_data)


@activity.defn
async def extract_chunk_activity(
    transcript_meta: dict,
    chunk_dict: dict,
    enriched_speakers: list[dict],
) -> list[dict]:
    """Extract claims from a single chunk of a transcript.

    Takes slim transcript_meta (7 fields), Chunk dict, and enriched speakers.
    Returns list of thesis dicts with original_quote.
    """
    from src.transcript.thesis_extractor import Chunk, extract_chunk
    from src.transcript.parsers import TranscriptData

    # Reconstruct minimal TranscriptData from slim meta
    td = TranscriptData(
        url=transcript_meta.get("url", ""),
        title=transcript_meta["title"],
        date=transcript_meta.get("date"),
        description=transcript_meta.get("description"),
        speakers=transcript_meta["speakers"],
        turns=[],  # not needed by extract_chunk — only metadata used
        source_format=transcript_meta.get("source_format", "rev"),
        speaker_aliases=transcript_meta.get("speaker_aliases", {}),
        _word_count_override=transcript_meta.get("word_count"),
        _turn_count_override=transcript_meta.get("turn_count"),
    )
    chunk = Chunk(**chunk_dict)

    log.info(activity.logger, "extract", "chunk_start",
             "Starting chunk extraction",
             title=td.title,
             chunk_index=chunk.chunk_index,
             total_chunks=chunk.total_chunks)

    try:
        theses = await extract_chunk(td, chunk, enriched_speakers, logger=activity.logger)
    except Exception as e:
        log.error(activity.logger, "extract", "chunk_failed",
                  "Chunk extraction failed",
                  chunk_index=chunk.chunk_index, error=str(e))
        raise

    # Serialize for Temporal transport
    result = []
    for t in theses:
        result.append({
            "thesis_statement": t.thesis_statement,
            "speakers": t.speakers,
            "original_quote": t.original_quote,
            "topic": t.topic,
        })

    log.info(activity.logger, "extract", "chunk_done",
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

    log.info(activity.logger, "dedup", "start",
             "Starting embedding dedup",
             speaker=speaker, claim_count=len(claims))

    try:
        result = await dedup_speaker_claims(claims, speaker, logger=activity.logger)
    except Exception as e:
        log.error(activity.logger, "dedup", "failed",
                  "Embedding dedup failed",
                  speaker=speaker, error=str(e))
        raise

    cluster_count = len(result["clusters"])
    multi = sum(1 for c in result["clusters"] if len(c["member_indices"]) > 1)

    log.info(activity.logger, "dedup", "done",
             "Embedding dedup complete",
             speaker=speaker,
             cluster_count=cluster_count,
             multi_member=multi)

    return result


@activity.defn
async def classify_claims_activity(claims: list[dict]) -> list[dict]:
    """Classify a batch of claims. Returns claims with classification fields added."""
    from src.transcript.claim_classifier import classify_claims_batch

    log.info(activity.logger, "classify", "start",
             "Starting claim classification",
             claim_count=len(claims))

    try:
        result = await classify_claims_batch(claims, logger=activity.logger)
    except Exception as e:
        log.error(activity.logger, "classify", "failed",
                  "Claim classification failed",
                  claim_count=len(claims), error=str(e))
        raise

    log.info(activity.logger, "classify", "done",
             "Claim classification complete",
             claim_count=len(result))

    return result


@activity.defn
async def synthesize_claim_activity(
    member_statements: list[str],
    topic: str,
    speaker: str,
    transcript_title: str = "",
    transcript_description: str = "",
    transcript_date: str = "",
) -> dict:
    """Synthesize an overarching claim for one group.

    Returns dict with overarching_claim and rationale.
    """
    from src.transcript.claim_synthesizer import synthesize_group_claim

    log.info(activity.logger, "synthesize", "start",
             "Starting claim synthesis",
             speaker=speaker, topic=topic,
             member_count=len(member_statements))

    try:
        output = await synthesize_group_claim(
            member_statements, topic, speaker,
            transcript_title=transcript_title,
            transcript_description=transcript_description,
            transcript_date=transcript_date,
            logger=activity.logger,
        )
    except Exception as e:
        log.error(activity.logger, "synthesize", "failed",
                  "Claim synthesis failed",
                  speaker=speaker, topic=topic, error=str(e))
        raise

    log.info(activity.logger, "synthesize", "done",
             "Claim synthesis complete",
             speaker=speaker, topic=topic,
             claim_preview=output.overarching_claim[:80])

    return output.model_dump()


@activity.defn
async def update_transcript_claims_classification(
    transcript_id: str,
    updates: list[dict],
) -> None:
    """Update classification fields on existing transcript_claims.

    Each update dict has: tc_id, classification, checkable, checkability_rationale,
    and optionally factual_anchor.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptClaim

    async with async_session() as session:
        for u in updates:
            tc_id = _uuid_mod.UUID(u["tc_id"])
            result = await session.execute(
                select(TranscriptClaim).where(TranscriptClaim.id == tc_id)
            )
            tc = result.scalar_one()
            tc.classification = u.get("classification")
            tc.checkable = u.get("checkable")
            tc.checkability_rationale = u.get("checkability_rationale")
            if "factual_anchor" in u:
                tc.factual_anchor = u.get("factual_anchor")
        await session.commit()

    log.info(activity.logger, "classify", "updated",
             "Classification updated on transcript_claims",
             transcript_id=transcript_id, count=len(updates))


@activity.defn
async def update_transcript_claims_dedup(
    transcript_id: str,
    updates: list[dict],
) -> None:
    """Update dedup fields on existing transcript_claims.

    Each update dict has: tc_id, is_duplicate, worth_checking, dedup_group_id.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptClaim

    async with async_session() as session:
        for u in updates:
            tc_id = _uuid_mod.UUID(u["tc_id"])
            result = await session.execute(
                select(TranscriptClaim).where(TranscriptClaim.id == tc_id)
            )
            tc = result.scalar_one()
            tc.is_duplicate = u.get("is_duplicate", False)
            tc.worth_checking = u.get("worth_checking", True)
            if "dedup_group_id" in u:
                tc.dedup_group_id = u["dedup_group_id"]
        await session.commit()

    log.info(activity.logger, "dedup", "updated",
             "Dedup flags updated on transcript_claims",
             transcript_id=transcript_id, count=len(updates))


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
    log.info(activity.logger, "store", "start", "Storing transcript",
             url=url, title=transcript_data.get("title"))

    try:
        # Enrich speakers with Wikidata descriptions before storing
        raw_speakers = transcript_data.get("speakers", [])
        enriched_speakers = await _enrich_speakers(raw_speakers)
        transcript_data["speakers"] = enriched_speakers
    except Exception as e:
        log.error(activity.logger, "store", "enrich_failed",
                  "Speaker enrichment failed",
                  url=url, error=str(e))
        raise

    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.url == url)
        )
        record = result.scalar_one_or_none()

        if record:
            record.title = transcript_data["title"]
            record.date = transcript_data.get("date")
            record.description = transcript_data.get("description")
            record.speakers = transcript_data["speakers"]
            record.enriched_speakers = enriched_speakers
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
                description=transcript_data.get("description"),
                speakers=transcript_data["speakers"],
                enriched_speakers=enriched_speakers,
                word_count=transcript_data["word_count"],
                segment_count=len(transcript_data["turns"]),
                display_text=transcript_data["display_text"],
                status="extracting",
                # v2 fields
                segments_data=transcript_data.get("turns"),
                source_format=transcript_data.get("source_format", "rev"),
                speaker_aliases=transcript_data.get("speaker_aliases"),
            )
            session.add(record)

        await session.commit()
        record_id = str(record.id)

    log.info(activity.logger, "store", "done",
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
                classification=c.get("classification"),
                topic=c.get("topic"),
                checkable=c.get("checkable"),
                checkability_rationale=c.get("checkability_rationale"),
                worth_checking=c.get("worth_checking", True),
                is_duplicate=c.get("is_duplicate", False),
                factual_anchor=c.get("factual_anchor"),
            )
            session.add(tc)
            await session.flush()
            tc_ids.append(str(tc.id))

        await session.commit()

    log.info(activity.logger, "store", "claims_stored",
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
    log.info(activity.logger, "claims", "create_start",
             "Creating Claim records for verification",
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
                    supporting_quotes=group.get("original_quotes"),
                    status="extracted",
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

    log.info(activity.logger, "claims", "created",
             "Claim records created and FKs linked",
             transcript_id=transcript_id, claim_count=len(claim_ids))

    return claim_ids


@activity.defn
async def queue_claims_for_verification(claim_ids: list[str]) -> int:
    """Flip claims from 'extracted' to 'queued' so verification picks them up.

    Called by the orchestrator right before starting VerifyAllClaimsWorkflow.
    This is the only place claims transition to 'queued' status.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import Claim

    count = 0
    async with async_session() as session:
        for cid_str in claim_ids:
            cid = _uuid_mod.UUID(cid_str)
            result = await session.execute(
                select(Claim).where(Claim.id == cid)
            )
            claim = result.scalar_one_or_none()
            if claim and claim.status == "extracted":
                claim.status = "queued"
                count += 1
        await session.commit()

    log.info(activity.logger, "claims", "queued",
             "Claims queued for verification",
             count=count, total=len(claim_ids))
    return count


@activity.defn
async def update_transcript_status(transcript_id: str, status: str) -> None:
    """Update a transcript's status field."""
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord

    tid = _uuid_mod.UUID(transcript_id)
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.id == tid)
        )
        record = result.scalar_one()
        record.status = status
        await session.commit()

    log.info(activity.logger, "status", "updated",
             "Transcript status updated",
             transcript_id=transcript_id, status=status)


@activity.defn
async def finish_transcript_and_start_next() -> str | None:
    """Mark completed transcripts and start the next queued one.

    1. Find transcripts with status='verifying' where ALL linked claims are verified
    2. Mark them 'complete'
    3. Find oldest 'queued' transcript and start its TranscriptPipelineWorkflow
    4. Return transcript_id if started, None if pipeline is idle
    """
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
                log.info(activity.logger, "queue", "transcript_complete",
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
            log.info(activity.logger, "queue", "empty",
                     "No queued transcripts")
            return None

        transcript_id = str(queued.id)
        url = queued.url
        queued.status = "extracting"
        await session.commit()

    # Step 3: Start extraction workflow
    from src.workflows.transcript_pipeline import TranscriptPipelineWorkflow

    temporal = await TemporalClient.connect(TEMPORAL_HOST)
    await temporal.start_workflow(
        TranscriptPipelineWorkflow.run,
        args=[url],
        id=f"extract-{transcript_id}",
        task_queue=TASK_QUEUE,
    )

    log.info(activity.logger, "queue", "next_started",
             "Started next queued transcript",
             transcript_id=transcript_id, url=url)

    return transcript_id


@activity.defn
async def load_extract_inputs(transcript_id: str) -> dict:
    """Load inputs for ExtractClaimsWorkflow from DB.

    Returns transcript_meta, enriched_speakers, and turns for a stored transcript.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord

    tid = _uuid_mod.UUID(transcript_id)
    async with async_session() as session:
        result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.id == tid)
        )
        record = result.scalar_one()

        transcript_meta = {
            "url": record.url,
            "title": record.title,
            "date": record.date,
            "description": record.description,
            "speakers": record.speakers,
            "word_count": record.word_count,
            "turn_count": record.segment_count,
            "source_format": record.source_format or "rev",
        }
        enriched_speakers = record.enriched_speakers or record.speakers or []
        turns = record.segments_data or []

    log.info(activity.logger, "load", "extract_inputs",
             "Loaded extract inputs from DB",
             transcript_id=transcript_id,
             turn_count=len(turns))

    return {
        "transcript_meta": transcript_meta,
        "enriched_speakers": enriched_speakers,
        "turns": turns,
    }


@activity.defn
async def load_classify_inputs(transcript_id: str) -> dict:
    """Load inputs for ClassifyAndDedupWorkflow from DB.

    Reconstructs tc_ids, all_theses, and enriched_speakers from stored
    TranscriptRecord + TranscriptClaim rows.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord, TranscriptClaim

    tid = _uuid_mod.UUID(transcript_id)
    async with async_session() as session:
        # Load transcript for enriched_speakers
        t_result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.id == tid)
        )
        record = t_result.scalar_one()
        enriched_speakers = record.enriched_speakers or record.speakers or []

        # Load transcript_claims in creation order
        tc_result = await session.execute(
            select(TranscriptClaim)
            .where(TranscriptClaim.transcript_id == tid)
            .order_by(TranscriptClaim.created_at.asc())
        )
        claims = tc_result.scalars().all()

        tc_ids = [str(tc.id) for tc in claims]
        all_theses = []
        for tc in claims:
            all_theses.append({
                "thesis_statement": tc.claim_text,
                "original_quote": tc.original_quote or "",
                "speakers": [tc.speaker] if tc.speaker else [],
                "topic": tc.topic,
                "classification": tc.classification,
                "checkable": tc.checkable,
                "check_rationale": tc.checkability_rationale or "",
                "factual_anchor": tc.factual_anchor,
                "worth_checking": tc.worth_checking,
                "is_duplicate": tc.is_duplicate,
            })

    log.info(activity.logger, "load", "classify_inputs",
             "Loaded classify inputs from DB",
             transcript_id=transcript_id,
             claim_count=len(tc_ids))

    return {
        "tc_ids": tc_ids,
        "all_theses": all_theses,
        "enriched_speakers": enriched_speakers,
    }


@activity.defn
async def load_synthesize_inputs(transcript_id: str) -> dict:
    """Load inputs for SynthesizeClaimsWorkflow from DB.

    Reconstructs dedup_groups, classified_theses, enriched_speakers, tc_ids,
    and transcript metadata from stored records.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord, TranscriptClaim

    tid = _uuid_mod.UUID(transcript_id)
    async with async_session() as session:
        # Load transcript
        t_result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.id == tid)
        )
        record = t_result.scalar_one()
        enriched_speakers = record.enriched_speakers or record.speakers or []

        # Build speaker descriptions
        speaker_descriptions = {}
        for s in enriched_speakers:
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

        # Load transcript_claims in creation order
        tc_result = await session.execute(
            select(TranscriptClaim)
            .where(TranscriptClaim.transcript_id == tid)
            .order_by(TranscriptClaim.created_at.asc())
        )
        claims = tc_result.scalars().all()

        tc_ids = [str(tc.id) for tc in claims]
        classified_theses = []
        for tc in claims:
            classified_theses.append({
                "thesis_statement": tc.claim_text,
                "original_quote": tc.original_quote or "",
                "speakers": [tc.speaker] if tc.speaker else [],
                "topic": tc.topic,
                "classification": tc.classification,
                "checkable": tc.checkable,
                "check_rationale": tc.checkability_rationale or "",
                "factual_anchor": tc.factual_anchor,
                "worth_checking": tc.worth_checking,
                "is_duplicate": tc.is_duplicate,
            })

        # Rebuild dedup_groups from dedup_group_id + classification data
        group_map: dict[str, list[int]] = {}
        for i, tc in enumerate(claims):
            gid = tc.dedup_group_id
            if gid:
                group_map.setdefault(gid, []).append(i)
            else:
                # No group — treat as singleton
                group_map.setdefault(f"{tc.speaker}_C{i}", []).append(i)

        dedup_groups = []
        for gid, member_indices in group_map.items():
            # Collect original quotes (deduped)
            original_quotes = []
            seen_quotes: set[str] = set()
            for gi in member_indices:
                quote = classified_theses[gi].get("original_quote", "")
                if quote and quote not in seen_quotes:
                    original_quotes.append(quote)
                    seen_quotes.add(quote)

            first = classified_theses[member_indices[0]]
            speaker = first["speakers"][0] if first.get("speakers") else "Unknown"
            dedup_groups.append({
                "group_id": gid,
                "local_group_id": gid.split("_", 1)[-1] if "_" in gid else gid,
                "speaker": speaker,
                "topic": first.get("topic", ""),
                "checkable": first.get("checkable", False),
                "member_global_indices": member_indices,
                "claim_text": "\n".join(
                    classified_theses[gi]["thesis_statement"]
                    for gi in member_indices
                ),
                "original_quotes": original_quotes,
            })

    log.info(activity.logger, "load", "synthesize_inputs",
             "Loaded synthesize inputs from DB",
             transcript_id=transcript_id,
             group_count=len(dedup_groups))

    return {
        "dedup_groups": dedup_groups,
        "classified_theses": classified_theses,
        "enriched_speakers": enriched_speakers,
        "tc_ids": tc_ids,
        "source_url": record.url,
        "transcript_date": record.date,
        "transcript_title": record.title,
        "transcript_description": record.description or "",
        "speaker_descriptions": speaker_descriptions,
    }


@activity.defn
async def load_verify_inputs(transcript_id: str) -> list[dict]:
    """Load inputs for VerifyAllClaimsWorkflow from DB.

    Returns list of claim dicts ready for sequential verification.
    Only includes claims with status='queued' linked to this transcript.
    """
    from sqlalchemy import select
    from src.db.session import async_session
    from src.db.models import TranscriptRecord, TranscriptClaim, Claim

    tid = _uuid_mod.UUID(transcript_id)
    async with async_session() as session:
        # Load transcript for metadata
        t_result = await session.execute(
            select(TranscriptRecord).where(TranscriptRecord.id == tid)
        )
        record = t_result.scalar_one()

        # Build speaker descriptions from enriched speakers
        speaker_descriptions = {}
        enriched = record.enriched_speakers or []
        for s in enriched:
            if isinstance(s, dict) and s.get("description"):
                speaker_descriptions[s["name"]] = s["description"]

        # Load linked claims via transcript_claims FK
        tc_result = await session.execute(
            select(TranscriptClaim)
            .where(TranscriptClaim.transcript_id == tid)
            .where(TranscriptClaim.claim_id.isnot(None))
        )
        tc_rows = tc_result.scalars().all()

        # Unique claim IDs (multiple TCs can link to one Claim)
        claim_ids = list({tc.claim_id for tc in tc_rows})
        if not claim_ids:
            return []

        c_result = await session.execute(
            select(Claim).where(Claim.id.in_(claim_ids))
        )
        claims = c_result.scalars().all()

        verify_claims = []
        for claim in claims:
            verify_claims.append({
                "claim_id": str(claim.id),
                "claim_text": claim.text,
                "speaker": claim.speaker,
                "speaker_description": (
                    speaker_descriptions.get(claim.speaker, "")
                    if claim.speaker else ""
                ),
                "transcript_date": record.date or "unknown",
                "transcript_title": record.title,
                "transcript_description": record.description or "",
                "supporting_quotes": claim.supporting_quotes or [],
            })

    log.info(activity.logger, "load", "verify_inputs",
             "Loaded verify inputs from DB",
             transcript_id=transcript_id,
             claim_count=len(verify_claims))

    return verify_claims


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
