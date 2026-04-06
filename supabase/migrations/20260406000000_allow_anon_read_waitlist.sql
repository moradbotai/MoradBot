-- Migration: allow anon read on waitlist_submissions for internal stats page
-- Date: 2026-04-06
-- Note: /stats is an internal-only page, not publicly linked

DROP POLICY IF EXISTS "service_read_waitlist" ON waitlist_submissions;

CREATE POLICY "anon_read_waitlist" ON waitlist_submissions
  FOR SELECT USING (true);
