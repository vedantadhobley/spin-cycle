"""Wikidata tool for discovering entity relationships.

This module provides tools for the research agent to query Wikidata for:
- Ownership chains (who owns this company?)
- Media holdings (what news outlets does this person/company own?)
- Institutional relationships (parent organizations, subsidiaries)
- Key personnel (CEO, board members)

This helps identify conflicts of interest when evaluating evidence sources.
For example, if a claim is about Company X, and we find that the CEO also owns
a major newspaper, then that newspaper's coverage is NOT independent.

Uses Wikidata's:
- Search API to find entities by name
- SPARQL endpoint for relationship queries

Caching:
- Results cached in PostgreSQL (wikidata_cache table)
- TTL: 7 days (entity relationships change infrequently)
- Cache hit = instant, cache miss = API query + store
"""

import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import quote

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.db.session import get_sync_session
from src.db.models import WikidataCache
from src.utils.logging import log, get_logger

MODULE = "wikidata"
logger = get_logger()

# Wikidata endpoints
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# User-Agent for Wikidata API (required, or you get 403)
USER_AGENT = "SpinCycle/1.0 (https://github.com/spin-cycle; fact-checking research tool)"

# Cache TTL — entities change less often than news bias
CACHE_TTL_DAYS = 7

# Relationship properties we care about
# Split into corporate/political and family for transitive expansion control
CORPORATE_PROPERTIES = {
    "P127": "owned_by",           # owned by (organization)
    "P749": "parent_org",         # parent organization
    "P355": "subsidiary",         # subsidiary
    "P1830": "owner_of",          # owner of (person owns X)
    "P169": "ceo",                # chief executive officer
    "P488": "chairperson",        # chairperson
    "P112": "founder",            # founded by
    "P463": "member_of",          # member of (organization)
    "P108": "employer",           # employer (person works for)
    "P39": "position_held",       # position held (political office)
    "P102": "political_party",    # member of political party
}

FAMILY_PROPERTIES = {
    "P26": "spouse",              # spouse / partner (married)
    "P451": "partner",            # unmarried partner
    "P22": "father",              # father
    "P25": "mother",              # mother
    "P40": "child",               # child
    "P3373": "sibling",           # sibling
    "P1038": "relative",          # general relative
}

# Corporate roles to also check for family members (hop-2).
# These capture orgs that family members LEAD (e.g., "Person A FOUNDED Org X").
# Excludes employer/member_of/political_party (too noisy for family-of-family).
FAMILY_CORPORATE_PROPERTIES = {
    "P112": "founder",
    "P488": "chairperson",
    "P169": "ceo",
    "P1830": "owner_of",
}

# Combined for SPARQL queries
PROPERTIES = {**CORPORATE_PROPERTIES, **FAMILY_PROPERTIES}

# In-memory cache for QID lookups within a single session (fast path)
_entity_cache: dict[str, Optional[str]] = {}


def _strip_leading_article(name: str) -> str:
    """Strip leading 'the', 'a', 'an' (case-insensitive)."""
    stripped = name.lstrip()
    for article in ("the ", "a ", "an "):
        if stripped.lower().startswith(article):
            return stripped[len(article):]
    return stripped


def _is_cache_stale(scraped_at: datetime) -> bool:
    """Check if cached data is older than TTL."""
    now = datetime.now(timezone.utc)
    if scraped_at.tzinfo is None:
        scraped_at = scraped_at.replace(tzinfo=timezone.utc)
    return (now - scraped_at) > timedelta(days=CACHE_TTL_DAYS)


def _get_cached_entity(entity_name: str) -> Optional[dict]:
    """Check PostgreSQL cache for entity data."""
    try:
        with get_sync_session() as session:
            stmt = select(WikidataCache).where(WikidataCache.entity_name == entity_name.lower())
            cached = session.execute(stmt).scalar_one_or_none()
            
            if cached and not _is_cache_stale(cached.scraped_at):
                log.debug(logger, MODULE, "cache_hit",
                         "Wikidata cache hit", entity=entity_name)
                return cached.relationships
            return None
    except Exception as e:
        log.warning(logger, MODULE, "cache_read_failed",
                    "Wikidata cache read failed", entity=entity_name, error=str(e))
        return None


def _store_cached_entity(entity_name: str, qid: Optional[str], relationships: dict) -> None:
    """Store entity data in PostgreSQL cache."""
    try:
        with get_sync_session() as session:
            stmt = insert(WikidataCache).values(
                entity_name=entity_name.lower(),
                qid=qid,
                relationships=relationships,
                scraped_at=datetime.now(timezone.utc),
            ).on_conflict_do_update(
                index_elements=["entity_name"],
                set_={
                    "qid": qid,
                    "relationships": relationships,
                    "scraped_at": datetime.now(timezone.utc),
                }
            )
            session.execute(stmt)
            session.commit()
            log.debug(logger, MODULE, "cache_write",
                     "Wikidata cached", entity=entity_name, qid=qid)
    except Exception as e:
        log.warning(logger, MODULE, "cache_write_failed",
                    "Wikidata cache write failed", entity=entity_name, error=str(e))


async def search_entity(name: str) -> Optional[str]:
    """Search Wikidata for an entity by name, return its QID.
    
    Args:
        name: Entity name (e.g., "Acme Corp", "John Smith", "FBI")
    
    Returns:
        Wikidata QID (e.g., "Q312") or None if not found
    """
    # Strip leading articles — NER often extracts "the Cato Institute" but
    # Wikidata search needs "Cato Institute"
    clean_name = _strip_leading_article(name)

    if clean_name in _entity_cache:
        return _entity_cache[clean_name]
    # Also check original name in cache
    if name in _entity_cache:
        return _entity_cache[name]

    params = {
        "action": "wbsearchentities",
        "search": clean_name,
        "language": "en",
        "format": "json",
        "limit": 1,
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                WIKIDATA_SEARCH,
                params=params,
                headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("search", [])
            if results:
                qid = results[0].get("id")
                _entity_cache[name] = qid
                _entity_cache[clean_name] = qid
                log.debug(logger, MODULE, "entity_found",
                         "Wikidata entity resolved", entity=clean_name, qid=qid)
                return qid

            _entity_cache[name] = None
            _entity_cache[clean_name] = None
            return None
            
    except Exception as e:
        log.warning(logger, MODULE, "search_failed",
                    "Wikidata search failed", entity=name, error=str(e))
        return None



async def get_entity_relationships(qid: str) -> dict:
    """Query Wikidata SPARQL for entity relationships.

    Args:
        qid: Wikidata QID (e.g., "Q312" for Amazon)

    Returns:
        Dict with relationship types as keys, lists of related entities as values
    """
    # Build VALUES clause dynamically from all properties
    prop_values = " ".join(f"wdt:{pid}" for pid in PROPERTIES)
    query = f"""
    SELECT ?prop ?value ?valueLabel WHERE {{
      VALUES ?prop {{ {prop_values} }}
      wd:{qid} ?prop ?value .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = {}
            for binding in data.get("results", {}).get("bindings", []):
                prop_uri = binding.get("prop", {}).get("value", "")
                prop_id = prop_uri.split("/")[-1] if prop_uri else None
                
                value_label = binding.get("valueLabel", {}).get("value", "")
                value_qid = binding.get("value", {}).get("value", "").split("/")[-1]
                
                if prop_id and prop_id in PROPERTIES:
                    prop_name = PROPERTIES[prop_id]
                    if prop_name not in results:
                        results[prop_name] = []
                    results[prop_name].append({
                        "name": value_label,
                        "qid": value_qid,
                    })
            
            return results
            
    except Exception as e:
        log.warning(logger, MODULE, "sparql_failed",
                    "Wikidata SPARQL query failed", qid=qid, error=str(e))
        return {}


async def get_media_owned_by(person_or_org: str) -> list[str]:
    """Find media outlets owned by a person or organization.
    
    This is the key query for conflict of interest detection.
    
    Args:
        person_or_org: Name of person or org (e.g., "John Smith", "Acme Corp")
    
    Returns:
        List of media outlet names owned by this entity
    """
    qid = await search_entity(person_or_org)
    if not qid:
        return []
    
    # Query for things this entity owns, then filter to media
    # P31 = instance of, Q11032 = newspaper, Q1002697 = news media, Q1616075 = television station
    query = f"""
    SELECT DISTINCT ?mediaLabel WHERE {{
      wd:{qid} wdt:P1830 ?media .
      ?media wdt:P31/wdt:P279* ?type .
      VALUES ?type {{ wd:Q11032 wd:Q1002697 wd:Q1616075 wd:Q5398426 wd:Q17232649 }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            data = resp.json()
            
            media = []
            for binding in data.get("results", {}).get("bindings", []):
                label = binding.get("mediaLabel", {}).get("value", "")
                if label:
                    media.append(label)
            
            return media
            
    except Exception as e:
        log.warning(logger, MODULE, "media_query_failed",
                    "Wikidata media query failed", entity=person_or_org, error=str(e))
        return []


def _extract_names(relationships: dict, key: str) -> list[str]:
    """Extract name strings from a relationship list."""
    return [r["name"] for r in relationships.get(key, [])]


def _extract_with_qids(relationships: dict, key: str) -> list[dict]:
    """Extract name+qid dicts from a relationship list."""
    return [{"name": r["name"], "qid": r["qid"]} for r in relationships.get(key, [])]


async def get_ownership_chain(entity_name: str) -> dict:
    """Get comprehensive ownership and relationship information for an entity.

    This is the main entry point for entity expansion. Returns corporate
    relationships, family ties, and media holdings.

    Args:
        entity_name: Name of company, organization, or person

    Returns:
        {
            "entity": "Person A",
            "qid": "Q...",
            "owned_by": [...], "parent_org": [...], "subsidiary": [...],
            "owner_of": [...], "ceo": [...], "founder": [...],
            "chairperson": [...], "employer": [...],
            "political_party": [...], "position_held": [...], "member_of": [...],
            "spouse": [...], "father": [...], "mother": [...],
            "child": [...], "sibling": [...], "relative": [...], "partner": [...],
            "media_holdings": [...],
            "family_expanded": {...},  # 2nd-hop family connections
        }
    """
    # Strip leading articles to match search_entity behavior
    clean_name = _strip_leading_article(entity_name)

    # Check cache with cleaned name first, then original
    cached = _get_cached_entity(clean_name) or _get_cached_entity(entity_name)
    if cached:
        return cached

    qid = await search_entity(entity_name)
    if not qid:
        return {
            "entity": entity_name,
            "qid": None,
            "error": f"Entity '{entity_name}' not found in Wikidata",
        }

    # Get direct relationships (corporate + family)
    relationships = await get_entity_relationships(qid)

    # Build result with all relationship types
    result = {
        "entity": entity_name,
        "qid": qid,
        # Corporate
        "owned_by": _extract_names(relationships, "owned_by"),
        "parent_org": _extract_names(relationships, "parent_org"),
        "subsidiary": _extract_names(relationships, "subsidiary"),
        "owner_of": _extract_names(relationships, "owner_of"),
        "ceo": _extract_names(relationships, "ceo"),
        "founder": _extract_names(relationships, "founder"),
        "chairperson": _extract_names(relationships, "chairperson"),
        "employer": _extract_names(relationships, "employer"),
        "political_party": _extract_names(relationships, "political_party"),
        "position_held": _extract_names(relationships, "position_held"),
        "member_of": _extract_names(relationships, "member_of"),
        # Family
        "spouse": _extract_names(relationships, "spouse"),
        "partner": _extract_names(relationships, "partner"),
        "father": _extract_names(relationships, "father"),
        "mother": _extract_names(relationships, "mother"),
        "child": _extract_names(relationships, "child"),
        "sibling": _extract_names(relationships, "sibling"),
        "relative": _extract_names(relationships, "relative"),
        "media_holdings": [],
        "family_expanded": {},
    }

    # Check for DIRECT media ownership by this entity
    direct_media = await get_media_owned_by(entity_name)
    result["media_holdings"].extend(direct_media)

    # Also check what media the owners/founders/CEOs own
    people_to_check = result["owned_by"] + result["founder"] + result["ceo"]
    for person in people_to_check[:3]:
        media = await get_media_owned_by(person)
        result["media_holdings"].extend(media)

    # 2-hop family expansion: for each family member, get THEIR family
    # relationships AND corporate roles (founder/chair/CEO/owner — control
    # relationships only, not employer/member_of which would explode).
    # This catches in-laws: Person A → Spouse B (spouse) → Parent C (father)
    # AND orgs they lead: Parent C → founder: Org X, Org Y
    family_members = _extract_with_qids(relationships, "spouse") + \
                     _extract_with_qids(relationships, "partner") + \
                     _extract_with_qids(relationships, "father") + \
                     _extract_with_qids(relationships, "mother") + \
                     _extract_with_qids(relationships, "child") + \
                     _extract_with_qids(relationships, "sibling")

    for member in family_members[:6]:  # Cap to avoid too many queries
        member_qid = member.get("qid")
        member_name = member.get("name")
        if not member_qid or not member_name:
            continue

        try:
            member_rels = await get_entity_relationships(member_qid)
            hop2_family = {}

            # Family relationships (existing)
            for prop_id, prop_name in FAMILY_PROPERTIES.items():
                names = _extract_names(member_rels, prop_name)
                if names:
                    hop2_family[prop_name] = names

            # Corporate roles — orgs this family member leads/founded/owns
            # Data is already in member_rels (same SPARQL call), zero extra queries
            family_corp_roles = {}
            for prop_id, prop_name in FAMILY_CORPORATE_PROPERTIES.items():
                names = _extract_names(member_rels, prop_name)
                if names:
                    family_corp_roles[prop_name] = names
            if family_corp_roles:
                hop2_family["corporate_roles"] = family_corp_roles

            # Also check media holdings of family members
            member_media = await get_media_owned_by(member_name)
            if member_media:
                hop2_family["media_holdings"] = member_media
                result["media_holdings"].extend(member_media)

            if hop2_family:
                result["family_expanded"][member_name] = hop2_family
                log.debug(logger, MODULE, "hop2_expanded",
                         "Wikidata hop-2 expansion", member=member_name,
                         relationships=list(hop2_family.keys()))
        except Exception as e:
            log.warning(logger, MODULE, "hop2_failed",
                        "Wikidata hop-2 expansion failed",
                        member=member_name, error=str(e))
            continue

    # Deduplicate media holdings
    result["media_holdings"] = list(set(result["media_holdings"]))

    _store_cached_entity(clean_name, qid, result)
    return result


def format_wikidata_result(result: dict) -> str:
    """Format Wikidata result as readable text.

    Args:
        result: Output from get_ownership_chain()

    Returns:
        Human-readable summary
    """
    if result.get("error"):
        return f"Wikidata: {result['error']}"

    lines = [f"**Wikidata: {result['entity']}** (QID: {result['qid']})"]

    if result.get("owned_by"):
        lines.append(f"- Owned by: {', '.join(result['owned_by'])}")

    if result.get("parent_org"):
        lines.append(f"- Parent organization: {', '.join(result['parent_org'])}")

    if result.get("subsidiary"):
        subs = result['subsidiary'][:5]
        if len(result['subsidiary']) > 5:
            lines.append(f"- Subsidiaries: {', '.join(subs)} (+{len(result['subsidiary'])-5} more)")
        else:
            lines.append(f"- Subsidiaries: {', '.join(subs)}")

    if result.get("owner_of"):
        lines.append(f"- Owner of: {', '.join(result['owner_of'])}")

    if result.get("ceo"):
        lines.append(f"- CEO: {', '.join(result['ceo'])}")

    if result.get("founder"):
        lines.append(f"- Founder: {', '.join(result['founder'])}")

    if result.get("employer"):
        lines.append(f"- Employer: {', '.join(result['employer'])}")

    if result.get("political_party"):
        lines.append(f"- Political party: {', '.join(result['political_party'])}")

    if result.get("position_held"):
        positions = result['position_held'][:3]
        lines.append(f"- Positions: {', '.join(positions)}")

    # Family relationships
    for key, label in [("spouse", "Spouse"), ("partner", "Partner"),
                       ("father", "Father"), ("mother", "Mother"),
                       ("child", "Children"), ("sibling", "Siblings"),
                       ("relative", "Relatives")]:
        if result.get(key):
            lines.append(f"- {label}: {', '.join(result[key])}")

    # 2nd-hop family connections
    if result.get("family_expanded"):
        lines.append("- **Extended family connections:**")
        for member, rels in result["family_expanded"].items():
            for rel_type, names_or_dict in rels.items():
                if rel_type == "media_holdings":
                    continue  # Shown separately below
                elif rel_type == "corporate_roles":
                    for role, org_names in names_or_dict.items():
                        lines.append(f"  - {member} → {role}: {', '.join(org_names)}")
                else:
                    lines.append(f"  - {member} → {rel_type}: {', '.join(names_or_dict)}")

    if result.get("media_holdings"):
        lines.append(f"- **Media holdings** (via owners/founders/family): {', '.join(result['media_holdings'])}")

    if len(lines) == 1:
        lines.append("- No significant relationships found")

    return "\n".join(lines)


def collect_all_connected_parties(result: dict) -> dict:
    """Extract all connected party names from a Wikidata result.

    Returns a flat structure ready for interested_parties expansion:
    - people: all connected people (executives, family, etc.)
    - media: all affiliated media outlets
    - orgs: all connected organizations

    This is the bridge between Wikidata's raw result and the
    interested_parties format used by decompose/judge.
    """
    if result.get("error"):
        return {"people": [], "media": [], "orgs": []}

    people = set()
    media = set(result.get("media_holdings", []))
    orgs = set()

    # Corporate connections → people
    for key in ("ceo", "founder", "chairperson"):
        for name in result.get(key, []):
            people.add(name)

    # Family connections → people
    for key in ("spouse", "partner", "father", "mother", "child", "sibling", "relative"):
        for name in result.get(key, []):
            people.add(name)

    # 2nd-hop family → people + orgs from corporate roles
    for member_name, rels in result.get("family_expanded", {}).items():
        for rel_type, names_or_dict in rels.items():
            if rel_type == "media_holdings":
                media.update(names_or_dict)
            elif rel_type == "corporate_roles":
                # Orgs that family members lead/founded/own
                # e.g., Person A → founder: ["Org X", "Org Y"]
                for role_name, org_names in names_or_dict.items():
                    orgs.update(org_names)
            else:
                people.update(names_or_dict)

    # Corporate connections → orgs
    for key in ("owned_by", "parent_org", "subsidiary", "owner_of", "employer", "member_of"):
        for name in result.get(key, []):
            orgs.add(name)

    # Don't include the entity itself
    entity_name = result.get("entity", "")
    people.discard(entity_name)
    orgs.discard(entity_name)

    return {
        "people": list(people),
        "media": list(media),
        "orgs": list(orgs),
    }


