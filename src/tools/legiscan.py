"""LegiScan — programmatic legislative evidence gathering.

LegiScan (https://legiscan.com) provides structured access to US legislation:
  - Bill search across all 50 states + US Congress
  - Bill details (status, sponsors, history)
  - Bill text (the actual legislative language)
  - Roll call votes with individual member positions (yea/nay/absent/NV)

## Architecture: programmatic, not agentic

Like Wikidata and MBFC, LegiScan runs PROGRAMMATICALLY — not as an agent
tool. Every subclaim gets a LegiScan search after the research agent finishes.
If the subclaim matches legislation, bill details + votes + text are injected
as evidence items alongside the agent's web search results.

Why not an agent tool:
  - The agent might not use it (as we saw — it skipped LegiScan even for
    voting record claims)
  - When it does use it, it burns tool calls from the evidence budget
  - Structured data sources should be deterministic, not optional

The bill TEXT is the key value-add. Web search finds articles *about* a bill,
but only LegiScan provides the actual legislative language. This lets the judge
identify "poison pills" — provisions slipped into otherwise popular bills that
explain otherwise puzzling voting patterns.

  - Free "Civic API" tier: 30,000 queries/month
  - Data is immutable (past votes don't change) → highly cacheable
  - Covers both House and Senate roll call votes

Env var gated: no LEGISCAN_API_KEY → search_legislation() returns [].
"""

import base64
import os
import re
import time as _time

import httpx

from src.utils.logging import log, get_logger

MODULE = "legiscan"
logger = get_logger()

LEGISCAN_API_KEY = os.getenv("LEGISCAN_API_KEY", "")
LEGISCAN_URL = "https://api.legiscan.com/"

# Bill text truncation limit — enough to capture key provisions and
# potential riders, but not so much that it overwhelms the judge prompt.
BILL_TEXT_MAX_CHARS = 8000


def is_available() -> bool:
    """Check if the LegiScan API key is configured."""
    return bool(LEGISCAN_API_KEY)


# ---------------------------------------------------------------------------
# Low-level API functions
# ---------------------------------------------------------------------------

async def _legiscan_request(params: dict) -> dict | None:
    """Make a request to the LegiScan API.

    All LegiScan API calls use the same endpoint with different `op` params.
    Returns the parsed JSON response or None on failure.
    """
    params["key"] = LEGISCAN_API_KEY

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(LEGISCAN_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "ERROR":
                log.warning(logger, MODULE, "api_error",
                            "LegiScan API error",
                            error=data.get("alert", {}).get("message", "unknown"))
                return None

            return data
        except Exception as e:
            log.warning(logger, MODULE, "request_failed",
                        "LegiScan request failed",
                        error=str(e), error_type=type(e).__name__)
            return None


async def search_bills(query: str, state: str = "US") -> list[dict]:
    """Search for bills matching a query.

    Args:
        query: Search terms (e.g., "healthcare appropriations bill")
        state: State code or "US" for federal (default: "US")

    Returns:
        List of bill summary dicts sorted by relevance.
    """
    data = await _legiscan_request({
        "op": "search",
        "state": state,
        "query": query,
    })

    if not data:
        return []

    search_result = data.get("searchresult", {})

    # LegiScan returns results as numbered keys "0", "1", "2"...
    # plus a "summary" key with metadata
    bills = []
    for key, val in search_result.items():
        if key == "summary":
            continue
        if isinstance(val, dict) and val.get("bill_id"):
            bills.append(val)

    bills.sort(key=lambda b: b.get("relevance", 0), reverse=True)
    return bills


async def get_bill(bill_id: int) -> dict | None:
    """Get full bill details including sponsors, history, and roll call IDs."""
    data = await _legiscan_request({"op": "getBill", "id": bill_id})
    if not data:
        return None
    return data.get("bill")


async def get_roll_call(roll_call_id: int) -> dict | None:
    """Get roll call vote details with individual member positions."""
    data = await _legiscan_request({"op": "getRollCall", "id": roll_call_id})
    if not data:
        return None
    return data.get("roll_call")


async def get_bill_text(doc_id: int) -> str | None:
    """Get the text content of a bill document.

    LegiScan returns bill text as base64-encoded content. We decode it
    and extract plain text from HTML. PDFs are skipped (would need a
    PDF parser dependency we don't want).

    Returns:
        Plain text of the bill, or None if unavailable/PDF.
    """
    data = await _legiscan_request({"op": "getBillText", "id": doc_id})
    if not data:
        return None

    text_data = data.get("text", {})
    doc_b64 = text_data.get("doc")
    mime = text_data.get("mime", "")

    if not doc_b64:
        return None

    try:
        raw = base64.b64decode(doc_b64).decode("utf-8", errors="replace")
    except Exception:
        return None

    if "html" in mime.lower():
        return _extract_text_from_html(raw)
    elif "text" in mime.lower():
        return raw
    else:
        # PDF or other binary — skip
        log.debug(logger, MODULE, "skip_binary",
                  "Skipping non-text bill document", mime=mime, doc_id=doc_id)
        return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_html(html: str) -> str:
    """Extract text from HTML, removing script/style elements."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_bill_detail(bill: dict) -> str:
    """Format full bill details as readable text."""
    parts = [
        f"Bill: {bill.get('bill_number', 'N/A')}",
        f"Title: {bill.get('title', 'N/A')}",
        f"State: {bill.get('state', 'N/A')}",
        f"Status: {bill.get('status_desc', 'N/A')} "
        f"(as of {bill.get('status_date', 'N/A')})",
    ]

    # Sponsors
    sponsors = bill.get("sponsors", [])
    if sponsors:
        sponsor_strs = []
        for s in sponsors[:10]:
            name = s.get("name", "Unknown")
            party = s.get("party", "")
            role = s.get("role", "")
            role_label = " (primary)" if role == "Primary" else ""
            sponsor_strs.append(f"{name} ({party}){role_label}")
        parts.append(f"Sponsors: {', '.join(sponsor_strs)}")

    if bill.get("description"):
        parts.append(f"Description: {bill['description']}")

    # Key history events (last 5)
    history = bill.get("history", [])
    if history:
        parts.append("Recent history:")
        for h in history[-5:]:
            date = h.get("date", "")
            action = h.get("action", "")
            chamber = h.get("chamber", "")
            parts.append(f"  {date} [{chamber}]: {action}")

    # Roll call vote summaries
    votes = bill.get("votes", [])
    if votes:
        parts.append(f"Roll call votes: {len(votes)} recorded")
        for v in votes[:5]:
            desc = v.get("desc", "Vote")
            date = v.get("date", "")
            yea = v.get("yea", 0)
            nay = v.get("nay", 0)
            parts.append(f"  {date}: {desc} — Yea: {yea}, Nay: {nay}")

    if bill.get("url"):
        parts.append(f"URL: {bill['url']}")

    return "\n".join(parts)


def _format_roll_call(rc: dict) -> str:
    """Format a roll call vote as readable text."""
    parts = [
        f"Vote: {rc.get('desc', 'N/A')}",
        f"Date: {rc.get('date', 'N/A')}",
        f"Chamber: {rc.get('chamber', 'N/A')}",
        f"Result: {'PASSED' if rc.get('passed') else 'FAILED'}",
        f"Yea: {rc.get('yea', 0)}, Nay: {rc.get('nay', 0)}, "
        f"NV: {rc.get('nv', 0)}, Absent: {rc.get('absent', 0)}",
    ]

    # Individual votes — group by vote type
    votes = rc.get("votes", [])
    if votes:
        by_vote: dict[str, list[str]] = {}
        for v in votes:
            name = v.get("name", "Unknown")
            party = v.get("party", "")
            vote_text = v.get("vote_text", "")
            entry = f"{name} ({party})"

            if vote_text not in by_vote:
                by_vote[vote_text] = []
            by_vote[vote_text].append(entry)

        for vote_type in ("Yea", "Nay", "Not Voting", "Absent"):
            members = by_vote.get(vote_type, [])
            if members:
                # Show all for Nay (usually fewer), cap others
                if vote_type == "Nay" or len(members) <= 20:
                    parts.append(f"\n{vote_type} ({len(members)}):")
                    parts.append(f"  {', '.join(members)}")
                else:
                    parts.append(f"\n{vote_type} ({len(members)}):")
                    parts.append(
                        f"  {', '.join(members[:20])} "
                        f"(+{len(members)-20} more)"
                    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point: programmatic evidence gathering
# ---------------------------------------------------------------------------

async def search_legislation(subclaim: str) -> list[dict]:
    """Search LegiScan for legislation matching a subclaim.

    This is the main entry point, called programmatically after the
    research agent finishes. Returns evidence items in the same format
    as the agent's web search results.

    For the top result:
      - Full bill details (sponsors, status, history)
      - Roll call votes with individual member positions
      - Bill TEXT (the actual legislative language — for poison pill detection)

    For the next 2 results:
      - Bill details only (conserves API quota)

    Args:
        subclaim: The subclaim text to search for.

    Returns:
        List of evidence dicts compatible with the judge:
        [{source_type, source_url, title, content, supports_claim}]
    """
    if not is_available():
        return []

    _t0 = _time.monotonic()

    bills = await search_bills(subclaim, state="US")
    if not bills:
        log.debug(logger, MODULE, "no_results",
                  "No legislation found", subclaim=subclaim)
        return []

    api_calls = 1  # search
    evidence = []

    # --- Top bill: full treatment (details + votes + text) ---
    top = bills[0]
    top_id = top.get("bill_id")
    if top_id:
        bill = await get_bill(top_id)
        api_calls += 1

        if bill:
            content_parts = [_format_bill_detail(bill)]

            # Roll call votes (cap at 2)
            for vote_info in bill.get("votes", [])[:2]:
                rc_id = vote_info.get("roll_call_id")
                if not rc_id:
                    continue
                rc = await get_roll_call(rc_id)
                api_calls += 1
                if rc:
                    content_parts.append(_format_roll_call(rc))

            # Bill text — the actual legislative language
            texts = bill.get("texts", [])
            if texts:
                # Prefer the most recent version (enrolled > engrossed > introduced)
                latest = texts[-1]
                doc_id = latest.get("doc_id")
                if doc_id:
                    bill_text = await get_bill_text(doc_id)
                    api_calls += 1
                    if bill_text:
                        truncated = bill_text[:BILL_TEXT_MAX_CHARS]
                        if len(bill_text) > BILL_TEXT_MAX_CHARS:
                            truncated += (
                                f"\n\n[... truncated at {BILL_TEXT_MAX_CHARS} chars, "
                                f"full text: {bill.get('url', 'see LegiScan')}]"
                            )
                        content_parts.append(
                            "BILL TEXT (legislative language):\n" + truncated
                        )

            evidence.append({
                "source_type": "web",
                "source_url": bill.get("url") or top.get("url"),
                "title": (f"{bill.get('bill_number', '')}: "
                          f"{bill.get('title', 'N/A')}"),
                "content": "\n\n".join(content_parts)[:15000],
                "supports_claim": None,
            })

    # --- Next 2 bills: details only (conserve quota) ---
    for bill_summary in bills[1:3]:
        bid = bill_summary.get("bill_id")
        if not bid:
            continue
        bill = await get_bill(bid)
        api_calls += 1
        if bill:
            evidence.append({
                "source_type": "web",
                "source_url": bill.get("url") or bill_summary.get("url"),
                "title": (f"{bill.get('bill_number', '')}: "
                          f"{bill.get('title', 'N/A')}"),
                "content": _format_bill_detail(bill)[:5000],
                "supports_claim": None,
            })

    log.info(logger, MODULE, "done",
             "Legislative evidence gathered",
             subclaim=subclaim,
             bills_found=len(bills),
             evidence_items=len(evidence),
             api_calls=api_calls,
             latency_ms=int((_time.monotonic() - _t0) * 1000))

    return evidence
