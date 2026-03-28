-- Migration: Thesis-level transcript extraction
-- Additive changes only — no column drops, no renames.

-- transcript_claims: support thesis-level data
ALTER TABLE transcript_claims ADD COLUMN IF NOT EXISTS supporting_references JSONB;
ALTER TABLE transcript_claims ADD COLUMN IF NOT EXISTS topic VARCHAR(64);
ALTER TABLE transcript_claims ADD COLUMN IF NOT EXISTS thesis_version INTEGER DEFAULT 1;

-- transcripts: store parsed segments + format info
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS segments_data JSONB;
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS source_format VARCHAR(32) DEFAULT 'revcom';
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS speaker_aliases JSONB;

-- Tag existing data as version 1
UPDATE transcript_claims SET thesis_version = 1 WHERE thesis_version IS NULL;
