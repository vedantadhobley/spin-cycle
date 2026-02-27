"""Wikidata tool for discovering entity relationships.

This module provides tools for the research agent to query Wikidata for:
- Ownership chains (who owns this company?)
- Media holdings (what news outlets does this person/company own?)
- Institutional relationships (parent organizations, subsidiaries)
- Key personnel (CEO, board members)

This helps identify conflicts of interest when evaluating evidence sources.
For example, if a claim is about Amazon, and we find that Jeff Bezos owns
both Amazon and the Washington Post, then WaPo coverage is NOT independent.

Uses Wikidata's:
- Search API to find entities by name
- SPARQL endpoint for relationship queries
"""

import httpx
import logging
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Wikidata endpoints
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# User-Agent for Wikidata API (required, or you get 403)
USER_AGENT = "SpinCycle/1.0 (https://github.com/spin-cycle; fact-checking research tool)"

# Relationship properties we care about
PROPERTIES = {
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

# Cache for entity lookups (entity name → Q ID)
_entity_cache: dict[str, Optional[str]] = {}


async def search_entity(name: str) -> Optional[str]:
    """Search Wikidata for an entity by name, return its QID.
    
    Args:
        name: Entity name (e.g., "Jeff Bezos", "Amazon", "FBI")
    
    Returns:
        Wikidata QID (e.g., "Q312") or None if not found
    """
    if name in _entity_cache:
        return _entity_cache[name]
    
    params = {
        "action": "wbsearchentities",
        "search": name,
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
                logger.debug(f"Wikidata: '{name}' → {qid}")
                return qid
            
            _entity_cache[name] = None
            return None
            
    except Exception as e:
        logger.warning(f"Wikidata search failed for '{name}': {e}")
        return None


async def get_entity_relationships(qid: str) -> dict:
    """Query Wikidata SPARQL for entity relationships.
    
    Args:
        qid: Wikidata QID (e.g., "Q312" for Amazon)
    
    Returns:
        Dict with relationship types as keys, lists of related entities as values
    """
    # SPARQL query to get all relevant relationships
    # We query both directions: what this entity owns AND who owns this entity
    query = f"""
    SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
      VALUES ?prop {{ wd:P127 wd:P749 wd:P355 wd:P1830 wd:P169 wd:P488 wd:P112 wd:P463 wd:P108 wd:P39 wd:P102 }}
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
        logger.warning(f"Wikidata SPARQL failed for {qid}: {e}")
        return {}


async def get_media_owned_by(person_or_org: str) -> list[str]:
    """Find media outlets owned by a person or organization.
    
    This is the key query for conflict of interest detection.
    
    Args:
        person_or_org: Name of person or org (e.g., "Jeff Bezos", "News Corp")
    
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
        logger.warning(f"Wikidata media query failed for '{person_or_org}': {e}")
        return []


async def get_ownership_chain(entity_name: str) -> dict:
    """Get comprehensive ownership information for an entity.
    
    This is the main function for the research agent to use.
    
    Args:
        entity_name: Name of company, organization, or person
    
    Returns:
        {
            "entity": "Amazon",
            "qid": "Q3884",
            "owned_by": ["Jeff Bezos"],
            "parent_org": ["Alphabet Inc"],
            "subsidiaries": ["AWS", "Twitch", "Whole Foods"],
            "ceo": ["Andy Jassy"],
            "founder": ["Jeff Bezos"],
            "media_holdings": ["Washington Post"],  # Via owner
            "political_party": [...],  # For politicians
            "relationships_raw": {...}  # Full relationship data
        }
    """
    qid = await search_entity(entity_name)
    if not qid:
        return {
            "entity": entity_name,
            "qid": None,
            "error": f"Entity '{entity_name}' not found in Wikidata",
        }
    
    # Get direct relationships
    relationships = await get_entity_relationships(qid)
    
    # Build result
    result = {
        "entity": entity_name,
        "qid": qid,
        "owned_by": [r["name"] for r in relationships.get("owned_by", [])],
        "parent_org": [r["name"] for r in relationships.get("parent_org", [])],
        "subsidiary": [r["name"] for r in relationships.get("subsidiary", [])],
        "owner_of": [r["name"] for r in relationships.get("owner_of", [])],
        "ceo": [r["name"] for r in relationships.get("ceo", [])],
        "founder": [r["name"] for r in relationships.get("founder", [])],
        "chairperson": [r["name"] for r in relationships.get("chairperson", [])],
        "employer": [r["name"] for r in relationships.get("employer", [])],
        "political_party": [r["name"] for r in relationships.get("political_party", [])],
        "position_held": [r["name"] for r in relationships.get("position_held", [])],
        "member_of": [r["name"] for r in relationships.get("member_of", [])],
        "media_holdings": [],
    }
    
    # Check for DIRECT media ownership by this entity
    # (e.g., News Corp directly owns Fox News)
    direct_media = await get_media_owned_by(entity_name)
    result["media_holdings"].extend(direct_media)
    
    # Also check what media the owners/founders/CEOs own
    # (e.g., Jeff Bezos owns Amazon AND Washington Post)
    people_to_check = result["owned_by"] + result["founder"] + result["ceo"]
    for person in people_to_check[:3]:  # Limit to avoid too many queries
        media = await get_media_owned_by(person)
        result["media_holdings"].extend(media)
    
    # Deduplicate
    result["media_holdings"] = list(set(result["media_holdings"]))
    
    return result


def format_wikidata_result(result: dict) -> str:
    """Format Wikidata result as readable text for the research agent.
    
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
    
    if result.get("media_holdings"):
        lines.append(f"- **Media holdings** (via owners/founders): {', '.join(result['media_holdings'])}")
        lines.append("  ⚠️ Coverage from these outlets may have conflicts of interest")
    
    if len(lines) == 1:
        lines.append("- No significant relationships found")
    
    return "\n".join(lines)


def get_wikidata_tool():
    """Get a LangChain tool that wraps Wikidata lookup.

    Returns a @tool-decorated function compatible with LangGraph agents.
    The tool returns formatted text — not raw dicts — because LangGraph
    tool nodes pass tool results as text to the next LLM call.
    """
    from langchain_core.tools import tool

    @tool
    async def wikidata_lookup(entity: str) -> str:
        """Look up entity relationships in Wikidata to find potential conflicts of interest.

        Use this tool to discover:
        - WHO OWNS a company (important for identifying media bias)
        - WHAT MEDIA a person/company owns (critical for source independence)
        - PARENT organizations and subsidiaries
        - Political affiliations of people and organizations
        - Employment history

        WHEN TO USE THIS:
        - Claims about corporations: check owner/founder to find their media holdings
        - Claims about politicians: check party affiliation, positions held
        - Claims about wealthy individuals: check what companies/media they own
        - Evaluating news sources: check if the publisher has ownership ties to claim subject

        EXAMPLE: For a claim about Amazon, query "Jeff Bezos" to discover he owns
        Washington Post — then WaPo coverage of Amazon is NOT independent.

        Args:
            entity: Name of person, company, or organization to look up
        """
        logger.debug(f"Wikidata lookup: {entity}")
        try:
            result = await get_ownership_chain(entity)
            return format_wikidata_result(result)
        except Exception as e:
            logger.warning(f"Wikidata lookup failed for '{entity}': {e}")
            return f"Wikidata lookup failed for '{entity}': {e}"

    return wikidata_lookup
