"""Temporal activities — thin wrappers around domain logic.

Each activity is a single unit of work that the Temporal worker executes.
Activities can be retried independently if they fail.

The verification pipeline has 7 activities:
  0. create_claim             — creates a claim record in the DB (when started from Temporal UI)
  1. decompose_claim          — normalize + extract facts + Wikidata expansion (2 LLM calls)
  2. research_subclaim        — seed search + MBFC→Wikidata + rank + ReAct agent + LegiScan + evidence NER
  3. judge_subclaim           — evidence ranking + annotation + LLM verdict
  4. synthesize_verdict       — LLM combines sub-verdicts into final verdict
  5. store_result             — writes result tree to Postgres
  6. start_next_queued_claim  — picks up next queued claim and starts its workflow

The pipeline is flat: decompose once → research each fact → judge each fact → synthesize.
Follows Google's SAFE and FActScore.

Domain logic lives in dedicated modules:
  - src/agent/decompose.py   — normalize + extract facts + Wikidata expansion
  - src/agent/research.py    — seed search, ranking, ReAct agent, evidence extraction
  - src/agent/judge.py       — evidence ranking, annotation, LLM verdict
  - src/agent/synthesize.py  — verdict synthesis
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from temporalio import activity

from src.db.session import async_session
from src.db.models import Claim, SubClaim, Evidence, Verdict
from src.utils.logging import log


@activity.defn
async def create_claim(
    claim_text: str,
    speaker: str | None = None,
    source_url: str | None = None,
) -> str:
    """Create a claim record in the database and return its ID.

    This lets workflows be self-contained — you can start a VerifyClaimWorkflow
    from Temporal UI with just the claim text, and it will create the DB record
    automatically. When started via the API, the claim already exists so this
    activity is skipped.
    """
    log.info(activity.logger, "create", "start", "Creating claim record",
             claim=claim_text)

    async with async_session() as session:
        async with session.begin():
            claim = Claim(
                text=claim_text,
                speaker=speaker,
                source_url=source_url,
                status="pending",
            )
            session.add(claim)
            await session.flush()
            claim_id = str(claim.id)

    log.info(activity.logger, "create", "done", "Claim record created",
             claim_id=claim_id)
    return claim_id


@activity.defn
async def decompose_claim(claim_text: str, speaker: str | None = None) -> dict:
    """Normalize and extract atomic verifiable facts and thesis from a claim.

    Delegates to src/agent/decompose.decompose() for all domain logic.
    If speaker is provided, they're automatically added as an interested party.
    """
    from src.agent.decompose import decompose
    return await decompose(claim_text, speaker=speaker)


@activity.defn
async def research_subclaim(
    sub_claim: str,
    interested_parties: dict | None = None,
    categories: list[str] | None = None,
    seed_queries: list[str] | None = None,
    speaker: str | None = None,
) -> dict:
    """Research evidence for a sub-claim using the LangGraph ReAct agent.

    Delegates to src/agent/research.research_claim() for all domain logic.

    Returns dict with:
        - evidence: list of evidence dicts
        - enriched_parties: InterestedPartiesDict with new parties/media
          discovered from MBFC ownership and evidence NER
    """
    log.info(activity.logger, "research", "start", "Researching sub-claim",
             sub_claim=sub_claim, categories=categories,
             seed_query_count=len(seed_queries) if seed_queries else 0)

    from src.agent.research import research_claim
    evidence, enriched_parties = await research_claim(
        sub_claim, interested_parties,
        categories=categories,
        seed_queries=seed_queries,
        speaker=speaker,
    )

    log.info(activity.logger, "research", "done", "Research complete",
             sub_claim=sub_claim, evidence_count=len(evidence))
    return {"evidence": evidence, "enriched_parties": dict(enriched_parties)}


@activity.defn
async def judge_subclaim(
    claim_text: str,
    sub_claim: str,
    evidence: list[dict],
    interested_parties: dict | list | None = None,
    speaker: str | None = None,
) -> dict:
    """Judge a sub-claim based on collected evidence.

    Normalizes interested_parties for backward compatibility, then
    delegates to src/agent/judge.judge() for all domain logic.
    """
    from src.agent.decompose import normalize_interested_parties
    from src.agent.judge import judge

    if interested_parties is None:
        interested_parties = normalize_interested_parties([])
    elif isinstance(interested_parties, list):
        interested_parties = normalize_interested_parties(interested_parties)

    return await judge(claim_text, sub_claim, evidence, interested_parties,
                       speaker=speaker)


@activity.defn
async def synthesize_verdict(
    claim_text: str,
    child_results: list[dict],
    thesis_info: dict | None = None,
) -> dict:
    """Combine child verdicts into a final overall verdict.

    Delegates to src/agent/synthesize.synthesize() for all domain logic.
    """
    from src.agent.synthesize import synthesize
    return await synthesize(claim_text, child_results, thesis_info)


@activity.defn
async def store_result(claim_id: str, result: dict) -> None:
    """Store the verification result in the database.

    The result dict is either:
      - A single fact verdict: {"sub_claim": ..., "verdict": ..., "evidence": [...]}
      - A synthesis: {"sub_claim": ..., "verdict": ..., "child_results": [...]}

    Sub-claims with child_results get is_leaf=False, leaf nodes get
    is_leaf=True. Evidence is stored for leaf nodes.
    """
    log.info(activity.logger, "store", "start", "Storing verification result",
             claim_id=claim_id, verdict=result.get("verdict"))

    async with async_session() as session:
        async with session.begin():
            # Fetch the claim
            claim_uuid = uuid.UUID(claim_id)
            db_result = await session.execute(
                select(Claim).where(Claim.id == claim_uuid)
            )
            claim = db_result.scalar_one_or_none()
            if not claim:
                log.error(activity.logger, "store", "claim_not_found",
                          "Claim not found in database",
                          error="claim_not_found", claim_id=claim_id)
                return

            # Recursive storage of the result tree
            async def _store_node(sub: dict, parent_id=None):
                """Store a sub-result node (leaf or synthesis) with evidence and children."""
                has_children = "child_results" in sub and sub["child_results"]
                sub_claim = SubClaim(
                    claim_id=claim_uuid,
                    parent_id=parent_id,
                    is_leaf=not has_children,
                    text=sub["sub_claim"],
                    verdict=sub.get("verdict"),
                    confidence=sub.get("confidence"),
                    reasoning=sub.get("reasoning"),
                )
                session.add(sub_claim)
                await session.flush()  # get sub_claim.id

                # Store evidence (leaves have evidence, synthesis nodes don't)
                for ev in sub.get("evidence", []):
                    source_type = ev.get("source_type", "web")
                    if source_type not in ("web", "wikipedia", "news_api"):
                        source_type = "web"
                    # Truncate string fields to fit DB column limits
                    title = ev.get("title")
                    if title and len(title) > 512:
                        title = title[:509] + "..."
                    domain = ev.get("domain")
                    if domain and len(domain) > 256:
                        domain = domain[:256]
                    evidence = Evidence(
                        sub_claim_id=sub_claim.id,
                        source_type=source_type,
                        source_url=ev.get("source_url"),
                        content=ev.get("content"),
                        title=title,
                        domain=domain,
                        bias=ev.get("bias"),
                        factual=ev.get("factual"),
                        tier=ev.get("tier"),
                        judge_index=ev.get("judge_index"),
                        assessment=ev.get("assessment"),
                        is_independent=ev.get("is_independent"),
                        key_point=ev.get("key_point"),
                    )
                    session.add(evidence)

                # Recurse into children for synthesis nodes
                if has_children:
                    for child in sub["child_results"]:
                        await _store_node(child, parent_id=sub_claim.id)

            # Store sub-claims recursively
            child_results = result.get("child_results")
            if child_results:
                # Complex claim — top-level children are the first level of sub-claims
                for sub in child_results:
                    await _store_node(sub)
            else:
                # Simple atomic claim — result itself is the only sub-claim
                await _store_node(result)

            # Write verdict
            verdict_row = Verdict(
                claim_id=claim_uuid,
                verdict=result.get("verdict", "unverifiable"),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning"),
                reasoning_chain=result.get("reasoning_chain"),
                citations=result.get("citations"),
            )
            session.add(verdict_row)

            # Update claim status
            claim.status = "verified"
            claim.updated_at = datetime.now(timezone.utc)

        log.info(activity.logger, "store", "done", "Result stored in database",
                 claim_id=claim_id, verdict=result.get("verdict"))


@activity.defn
async def start_next_queued_claim() -> str | None:
    """Check for queued claims and start the next one.

    Called at the end of each workflow to trigger the next claim in queue.
    Uses SELECT ... FOR UPDATE for race safety.
    Returns the claim_id if a workflow was started, None otherwise.
    """
    import os
    from temporalio.client import Client as TemporalClient

    TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
    TASK_QUEUE = "spin-cycle-verify"

    async with async_session() as session:
        # Find oldest queued claim (locked to prevent races)
        result = await session.execute(
            select(Claim)
            .where(Claim.status == "queued")
            .order_by(Claim.created_at.asc())
            .with_for_update()
            .limit(1)
        )
        claim = result.scalar_one_or_none()

        if not claim:
            log.info(activity.logger, "queue", "empty", "No queued claims")
            return None

        claim_id = str(claim.id)
        claim_text = claim.text
        claim_speaker = claim.speaker

        # Update status to pending before starting workflow
        claim.status = "pending"
        claim.updated_at = datetime.now(timezone.utc)
        await session.commit()

        log.info(activity.logger, "queue", "starting_next",
                 "Starting next queued claim",
                 claim_id=claim_id)

    # Connect to Temporal and start the workflow
    # Import here to avoid circular dependency with workflow module
    from src.workflows.verify import VerifyClaimWorkflow

    temporal = await TemporalClient.connect(TEMPORAL_HOST)
    await temporal.start_workflow(
        VerifyClaimWorkflow.run,
        args=[claim_id, claim_text, claim_speaker],
        id=f"verify-{claim_id}",
        task_queue=TASK_QUEUE,
    )

    log.info(activity.logger, "queue", "workflow_started",
             "Queued claim workflow started",
             claim_id=claim_id)

    return claim_id
