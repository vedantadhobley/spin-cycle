-- 003_pipeline_refactor.sql
-- Drop dead columns, add new ones, rename claim_type → classification.

ALTER TABLE transcript_claims DROP COLUMN IF EXISTS segment_gist;
ALTER TABLE transcript_claims DROP COLUMN IF EXISTS supporting_references;
ALTER TABLE transcript_claims DROP COLUMN IF EXISTS is_restatement;
ALTER TABLE transcript_claims DROP COLUMN IF EXISTS thesis_version;
ALTER TABLE transcript_claims DROP COLUMN IF EXISTS skip_reason;
ALTER TABLE transcript_claims ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE;
ALTER TABLE transcript_claims ADD COLUMN IF NOT EXISTS factual_anchor TEXT;
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_name='transcript_claims' AND column_name='claim_type')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_name='transcript_claims' AND column_name='classification')
  THEN
    ALTER TABLE transcript_claims RENAME COLUMN claim_type TO classification;
  END IF;
END $$;
