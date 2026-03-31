-- Migration: create waitlist_submissions table
-- Purpose: Store landing page form submissions (waitlist + beta tester forms)
-- Date: 2026-04-01

CREATE TABLE IF NOT EXISTS waitlist_submissions (
  id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  form_type    TEXT        NOT NULL CHECK (form_type IN ('waitlist', 'beta_tester')),
  name         TEXT        NOT NULL,
  email        TEXT        NOT NULL,
  store_url    TEXT,
  platform     TEXT,
  store_size   TEXT,
  phone        TEXT,
  wants_beta   BOOLEAN     DEFAULT FALSE,
  submitted_at TIMESTAMPTZ DEFAULT NOW(),
  ip_hash      TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Index for quick email lookups
CREATE INDEX IF NOT EXISTS idx_waitlist_email        ON waitlist_submissions(email);
CREATE INDEX IF NOT EXISTS idx_waitlist_form_type    ON waitlist_submissions(form_type);
CREATE INDEX IF NOT EXISTS idx_waitlist_submitted_at ON waitlist_submissions(submitted_at);

-- RLS: table is publicly insertable but only service role can read
ALTER TABLE waitlist_submissions ENABLE ROW LEVEL SECURITY;

-- Allow public insert (landing page submissions — no auth required)
CREATE POLICY "public_insert_waitlist" ON waitlist_submissions
  FOR INSERT WITH CHECK (true);

-- Only service role can read (admin/analytics access only)
CREATE POLICY "service_read_waitlist" ON waitlist_submissions
  FOR SELECT USING (auth.role() = 'service_role');
