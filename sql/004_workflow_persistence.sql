-- 004_workflow_persistence.sql
-- Persist data that was previously only passed through workflow arguments,
-- enabling each workflow to be called independently with just transcript_id.

-- Store enriched speaker data (Wikidata roles/descriptions) on the transcript.
-- Previously only lived in workflow memory between fetch and downstream workflows.
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS enriched_speakers JSONB;

-- Source-specific blurb/description (rev.com description, editor's note, og:description).
-- Provides downstream context for extraction and verification.
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS description TEXT;

-- Track which dedup group each transcript_claim belongs to.
-- Previously only the binary is_duplicate flag was stored; the full group
-- structure was transient workflow data.
ALTER TABLE transcript_claims ADD COLUMN IF NOT EXISTS dedup_group_id VARCHAR(64);

-- Store original transcript quotes on the claim record for verification context.
-- Previously passed through workflow args but never persisted on the claim.
ALTER TABLE claims ADD COLUMN IF NOT EXISTS supporting_quotes JSONB;
