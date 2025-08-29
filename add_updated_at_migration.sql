-- Migration to add updated_at column to scans table
-- This adds a timestamp for when scan processing completes

ALTER TABLE scans ADD COLUMN updated_at DATETIME NULL;

-- For existing completed scans, set updated_at to created_at as an approximation
-- This is just to have some value for existing data
UPDATE scans SET updated_at = created_at WHERE status = 'completed' AND updated_at IS NULL;
