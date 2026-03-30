"""Speaker enrichment via Wikidata lookups.

Shared by the legacy extractor and the thesis extractor. Resolves speaker
names against Wikidata to get role/title descriptions for pronoun resolution
during claim decontextualization.
"""

import re

_ANONYMOUS_SPEAKER = re.compile(
    r"^(?:speaker\s*\w{0,3}|unknown|unidentified|moderator|host|interviewer"
    r"|caller|audience\s*member|voice(?:\s*over)?)$",
    re.IGNORECASE,
)

_JUNK_DESCRIPTION = re.compile(
    r"(?:^(?:male|female)\s+given\s+name$"
    r"|^given\s+name$"
    r"|^(?:family|sur)\s*name"
    r"|scientific\s+article"
    r"|^Wikimedia\s+disambiguation"
    r"|^human\s+name$"
    r")",
    re.IGNORECASE,
)


async def _enrich_speakers(speakers: list[str]) -> list[dict]:
    """Look up Wikidata descriptions for speakers.

    Skips anonymous/generic names (Speaker 1, Unknown, etc.) and filters
    out junk Wikidata hits (male given name, scientific article, etc.).

    Returns list of dicts like:
        [{"name": "Donald Trump", "description": "45th and 47th president of the United States"},
         {"name": "Speaker 1", "description": null}]
    """
    import asyncio
    from src.tools.wikidata import get_entity_description

    async def _lookup(name: str) -> dict:
        stripped = name.strip()
        if _ANONYMOUS_SPEAKER.match(stripped):
            return {"name": name, "description": None}
        if len(stripped.split()) < 2:
            return {"name": name, "description": None}
        try:
            desc = await get_entity_description(name)
            if desc and _JUNK_DESCRIPTION.search(desc):
                desc = None
            return {"name": name, "description": desc}
        except Exception:
            return {"name": name, "description": None}

    results = await asyncio.gather(*[_lookup(s) for s in speakers])
    return list(results)
