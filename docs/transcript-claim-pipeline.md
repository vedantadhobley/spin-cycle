# Transcript Claim Pipeline

Design document for automated claim extraction from public transcripts.

## Overview

Extend Spin Cycle to automatically discover, extract, and verify claims from
public transcripts (initially Rev.com). The pipeline adds three new stages
before the existing verification flow:

```
Transcript Discovery → Claim Extraction → Claim Contextualization → [existing pipeline]
                                                                      ↓
                                                              decompose → research → judge → synthesize
```

## Data Source: Rev.com

- **Transcript index**: `https://www.rev.com/transcripts`
- **Individual transcripts**: `https://www.rev.com/transcripts/{slug}`
- Transcripts include speaker labels, timestamps, and full text
- Content: press briefings, congressional hearings, political speeches,
  interviews, debates

### Polling Strategy

- Poll the transcript index once per day
- Track already-processed transcript slugs to avoid reprocessing
- Store transcript metadata (URL, title, date, speakers) in the database

### Transcript Selection

Which transcripts are worth processing? Options to explore:

1. **Speaker-based**: Curated list of speakers whose claims matter
   (elected officials, cabinet members, agency heads)
2. **Category-based**: Rev.com may categorize transcripts (political,
   legal, etc.) — filter by relevant categories
3. **Keyword-based**: Scan titles/descriptions for keywords indicating
   factual claims (policy announcements, press briefings, hearings)
4. **Hybrid**: Combine approaches — e.g., any transcript featuring a
   tracked speaker, plus any press briefing

The selection criteria will evolve. Start simple (curated speaker list)
and expand.

## Stage 1: Transcript Fetching

Fetch and parse the transcript HTML from Rev.com.

### Input
- Transcript URL (from discovery/polling)

### Output
- Structured transcript: list of `{speaker, timestamp, text}` segments
- Transcript metadata: title, date, URL, speakers list

### Considerations
- Rev.com HTML structure may change — parser should be resilient
- Rate limiting / politeness (1 req/sec or similar)
- Store raw transcript text for audit/reprocessing

## Stage 2: Claim Extraction

LLM pass over the transcript to identify verifiable factual assertions.

### Input
- Structured transcript (speaker-labeled segments)

### Output
- List of claims: `{text, speaker, timestamp, context_window}`
- Each claim is a verifiable factual assertion with bracketed context

### What Is a Claim?

A claim is a **specific, verifiable factual assertion**. Not every sentence
in a transcript is a claim:

| Type | Example | Claim? |
|------|---------|--------|
| Factual assertion | "We've struck over 7,000 targets" | Yes |
| Quantitative claim | "Inflation is down 40% from its peak" | Yes |
| Historical claim | "Iran has terrorized the US for 47 years" | Yes |
| Opinion | "This is the greatest country on earth" | No |
| Greeting | "Thank you for being here today" | No |
| Rhetoric | "We will never back down" | No |
| Prediction | "We will win this war" | No (unless testable) |
| Policy statement | "We are committed to diplomacy" | No |

### Contextual Bracketing

When extracting claims from transcripts, pronouns and references must be
resolved using surrounding context. Use the standard journalistic convention
of square brackets for editorial insertions:

**Before** (raw transcript):
> "We've damaged or sunk over 120 of their Navy ships"

**After** (contextualized claim):
> "[The United States has] damaged or sunk over 120 of [Iran's] Navy ships"

Rules:
- Square brackets `[]` indicate editorial insertions/clarifications
- Original words stay intact outside brackets
- Only add context that is unambiguously clear from the transcript
- Preserve the speaker's actual words where possible
- When the speaker uses "we"/"our"/"my", resolve to the specific entity
- When the speaker references "they"/"their"/"them", resolve to the
  specific entity being discussed

### Windowed Processing

Transcripts can be long (30-60+ minutes). Process in overlapping windows:
- Window size: ~2000-3000 tokens of transcript text
- Overlap: ~500 tokens to catch claims that span segment boundaries
- Each window includes speaker labels and enough preceding context for
  reference resolution
- Deduplicate claims across windows (same assertion from same speaker)

### LLM Prompt Design

The extraction prompt should:
1. Receive a transcript window with speaker labels
2. Identify all verifiable factual assertions
3. Apply contextual bracketing to resolve references
4. Output structured JSON with claim text, speaker, and the original
   unmodified quote for reference
5. Categorize claim type (quantitative, historical, attribution, etc.)
   to help downstream prioritization

## Stage 3: Claim Prioritization

Not all extracted claims are equally important to verify. Prioritize by:

1. **Specificity**: Quantitative claims ("7,000 targets", "$3,000 loss")
   are more verifiable than vague claims ("things are going well")
2. **Impact**: Claims about policy, casualties, spending, rights
3. **Novelty**: Claims not already verified in our database
4. **Controversy**: Claims that contradict other public statements or
   established facts

This could be a lightweight LLM pass or rule-based scoring.

## Integration with Existing Pipeline

Extracted and contextualized claims feed directly into the existing
submission API:

```python
POST /claims
{
    "text": "[The United States has] damaged or sunk over 120 of [Iran's] Navy ships",
    "speaker": "Pete Hegseth",
    "source": "https://www.rev.com/transcripts/pentagon-press-briefing-for-3-19-26"
}
```

The existing pipeline handles everything from there:
- Speaker added as interested party + Wikidata expanded
- Normalizer preserves bracketed context (may need minor adjustment)
- Research agent searches for independent verification
- Judge evaluates with speaker independence awareness

### Normalizer Interaction

The normalizer currently rewrites claims into neutral language. With
bracketed claims, it should:
- Preserve or remove brackets (the context is already resolved)
- NOT re-attribute to the speaker ("Pete Hegseth stated that...")
- Focus on neutralizing framing language only

Example:
- Input: `"[The United States has] damaged or sunk over 120 of [Iran's] Navy ships"`
- Normalized: `"The United States has damaged or sunk over 120 of Iran's Navy ships"`
- The normalizer just strips brackets and proceeds normally

## Database Schema

New tables/columns needed:

### `transcripts` table
- `id` (UUID, PK)
- `source_url` (VARCHAR) — Rev.com URL
- `title` (VARCHAR)
- `date` (DATE) — date of the event
- `speakers` (JSONB) — list of speakers in the transcript
- `raw_text` (TEXT) — full transcript text
- `status` (ENUM: pending, processing, extracted, failed)
- `created_at`, `updated_at`

### `transcript_claims` table (junction)
- `transcript_id` (FK → transcripts)
- `claim_id` (FK → claims)
- `timestamp` (VARCHAR) — timestamp within transcript
- `original_quote` (TEXT) — unmodified transcript text
- `context_window` (TEXT) — surrounding transcript text for reference

### `claims` table additions
- `transcript_id` (FK → transcripts, nullable) — link back to source transcript

## Architecture

### New Components

```
src/
  transcript/
    fetcher.py        — Rev.com HTML parsing, transcript fetching
    extractor.py      — LLM claim extraction with bracketing
    prioritizer.py    — Claim importance scoring
    poller.py         — Daily polling for new transcripts
  prompts/
    extraction.py     — LLM prompts for claim extraction
```

### Scheduling

- **Polling**: Daily cron job or Temporal scheduled workflow
- **Extraction**: Temporal workflow per transcript
  (fetch → extract → prioritize → submit claims)
- **Verification**: Existing claim queue handles the rest

## Open Questions

1. **Rev.com access**: Do we need API access or is HTML scraping sufficient?
   Are there rate limits or terms of service considerations?
2. **Other sources**: Should we support other transcript sources beyond
   Rev.com? (C-SPAN, congressional record, White House press briefings)
3. **Real-time vs. batch**: Daily polling is the starting point, but could
   we eventually process transcripts as they're published?
4. **Claim deduplication**: Same claim may appear in multiple transcripts
   (talking points). How do we detect and handle repeats?
5. **Transcript language**: Some transcripts may be partially in other
   languages (diplomatic events). Handle or skip?

## Implementation Order

1. Transcript fetcher — parse Rev.com HTML into structured segments
2. Claim extraction prompt — LLM extraction with bracketing
3. End-to-end test — fetch a known transcript, extract claims, submit
4. Database schema — transcripts table, junction table
5. Polling/scheduling — daily discovery of new transcripts
6. Prioritization — scoring and filtering extracted claims
