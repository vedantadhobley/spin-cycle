"""Typed schema for the interested_parties dict that flows through the pipeline.

This dict is built incrementally:
  - decompose: LLM populates direct, institutional, affiliated_media, reasoning
  - expand_interested_parties: Wikidata adds all_parties, wikidata_context
  - judge: reads all fields for conflict detection and annotation

Using TypedDict (not dataclass/pydantic) because Temporal passes it as a
plain dict over JSON. TypedDict gives IDE support and type checking without
runtime overhead or serialization concerns.

total=False because early pipeline stages (pre-Wikidata) don't have
all_parties or wikidata_context yet.
"""

from typing import TypedDict


class InterestedPartiesDict(TypedDict, total=False):
    direct: list[str]
    institutional: list[str]
    affiliated_media: list[str]
    reasoning: str | None
    all_parties: list[str]
    wikidata_context: str
