-- Migration: add wants_beta column to waitlist_submissions
-- Date: 2026-04-06
-- Root cause: table was created without wants_beta column,
--             causing all form INSERT operations to fail silently.

ALTER TABLE waitlist_submissions
  ADD COLUMN IF NOT EXISTS wants_beta BOOLEAN DEFAULT FALSE;
