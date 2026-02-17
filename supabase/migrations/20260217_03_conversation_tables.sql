-- Migration 3: Conversation Tables
-- Created: 2026-02-17
-- Description: Visitor sessions, tickets, messages, and escalations

-- ============================================
-- 1. VISITOR_SESSIONS TABLE
-- ============================================

CREATE TABLE visitor_sessions (
  -- Primary Key
  visitor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Key
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- Visitor Identification
  session_cookie VARCHAR(255) NOT NULL,

  -- Consent
  consent_given BOOLEAN NOT NULL DEFAULT false,
  consent_given_at TIMESTAMPTZ,

  -- Personal Data (Encrypted at application level)
  email_encrypted TEXT,
  phone_encrypted TEXT,
  name_encrypted TEXT,

  -- Session Info
  first_visit_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_visit_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  visit_count INTEGER NOT NULL DEFAULT 1 CHECK (visit_count > 0),

  -- User Agent Info
  user_agent TEXT,
  ip_address INET,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Constraints
  UNIQUE(store_id, session_cookie)
);

COMMENT ON TABLE visitor_sessions IS 'Unique visitors tracked by session cookie';
COMMENT ON COLUMN visitor_sessions.session_cookie IS 'Browser session/cookie identifier';
COMMENT ON COLUMN visitor_sessions.email_encrypted IS 'Encrypted email (app-level encryption)';

-- Indexes
CREATE INDEX idx_visitors_store ON visitor_sessions(store_id);
CREATE INDEX idx_visitors_cookie ON visitor_sessions(session_cookie);
CREATE INDEX idx_visitors_last_visit ON visitor_sessions(store_id, last_visit_at DESC);
CREATE INDEX idx_visitors_consent ON visitor_sessions(store_id, consent_given) WHERE consent_given = true;

-- ============================================
-- 2. TICKETS TABLE
-- ============================================

CREATE TABLE tickets (
  -- Primary Key
  ticket_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  visitor_id UUID NOT NULL REFERENCES visitor_sessions(visitor_id) ON DELETE CASCADE,

  -- Ticket Status
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'escalated', 'closed')),
  resolution_type VARCHAR(50) CHECK (resolution_type IN ('bot_answered', 'escalated', 'auto_closed', 'merchant_closed')),

  -- Context
  initial_question TEXT,
  category VARCHAR(50),

  -- Clarification Tracking
  clarification_count INTEGER NOT NULL DEFAULT 0 CHECK (clarification_count >= 0 AND clarification_count <= 3),

  -- Timing
  opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMPTZ,
  escalated_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE tickets IS 'Conversation threads/tickets';
COMMENT ON COLUMN tickets.clarification_count IS 'Number of clarification attempts (max 3 before escalation)';

-- Indexes
CREATE INDEX idx_tickets_store ON tickets(store_id, status);
CREATE INDEX idx_tickets_visitor ON tickets(visitor_id, created_at DESC);
CREATE INDEX idx_tickets_status ON tickets(store_id, status, updated_at DESC);
CREATE INDEX idx_tickets_escalated ON tickets(store_id, escalated_at DESC) WHERE status = 'escalated';
CREATE INDEX idx_tickets_open ON tickets(store_id, opened_at DESC) WHERE status = 'open';

-- ============================================
-- 3. MESSAGES TABLE
-- ============================================

CREATE TABLE messages (
  -- Primary Key
  message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  ticket_id UUID NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- Message Content
  sender_type VARCHAR(20) NOT NULL CHECK (sender_type IN ('visitor', 'bot', 'merchant')),
  content_ar TEXT NOT NULL CHECK (char_length(content_ar) > 0),

  -- Metadata
  is_clarification_request BOOLEAN NOT NULL DEFAULT false,
  includes_dynamic_data BOOLEAN NOT NULL DEFAULT false,

  -- AI Metadata (for bot messages only)
  model_used VARCHAR(50),
  tokens_used INTEGER CHECK (tokens_used >= 0),
  response_time_ms INTEGER CHECK (response_time_ms >= 0),

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE messages IS 'Individual messages within tickets';
COMMENT ON COLUMN messages.includes_dynamic_data IS 'True if message contains dynamic data (price/stock) requiring disclaimer';
COMMENT ON COLUMN messages.tokens_used IS 'AI tokens consumed (for cost tracking)';

-- Indexes
CREATE INDEX idx_messages_ticket ON messages(ticket_id, created_at ASC);
CREATE INDEX idx_messages_store ON messages(store_id, created_at DESC);
CREATE INDEX idx_messages_bot ON messages(store_id, sender_type, created_at DESC)
  WHERE sender_type = 'bot';
CREATE INDEX idx_messages_created ON messages(created_at DESC);

-- Partitioning recommendation (future): Partition by created_at monthly

-- ============================================
-- 4. ESCALATIONS TABLE
-- ============================================

CREATE TABLE escalations (
  -- Primary Key
  escalation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  ticket_id UUID NOT NULL UNIQUE REFERENCES tickets(ticket_id) ON DELETE CASCADE,
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  visitor_id UUID NOT NULL REFERENCES visitor_sessions(visitor_id) ON DELETE CASCADE,

  -- Escalation Details
  reason VARCHAR(100) NOT NULL CHECK (reason IN ('failed_clarification', 'unsupported_request', 'manual_request', 'error')),
  problem_description TEXT NOT NULL CHECK (char_length(problem_description) > 0),

  -- Contact Info (Collected during escalation)
  contact_method VARCHAR(20) NOT NULL CHECK (contact_method IN ('email', 'phone')),
  contact_value_encrypted TEXT NOT NULL,

  -- Order Info (Optional)
  order_number VARCHAR(100),

  -- Status
  status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'resolved', 'closed')),

  -- Resolution
  resolved_by VARCHAR(20) CHECK (resolved_by IN ('merchant', 'system')),
  resolution_notes TEXT,

  -- Timing
  escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMPTZ,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE escalations IS 'Escalated tickets requiring merchant attention';
COMMENT ON COLUMN escalations.contact_value_encrypted IS 'Encrypted email or phone (app-level encryption)';

-- Indexes
CREATE INDEX idx_escalations_store_status ON escalations(store_id, status, escalated_at DESC);
CREATE INDEX idx_escalations_pending ON escalations(store_id, escalated_at DESC)
  WHERE status = 'pending';
CREATE INDEX idx_escalations_ticket ON escalations(ticket_id);

-- ============================================
-- 5. TRIGGERS
-- ============================================

CREATE TRIGGER update_visitor_sessions_updated_at
  BEFORE UPDATE ON visitor_sessions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tickets_updated_at
  BEFORE UPDATE ON tickets
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_escalations_updated_at
  BEFORE UPDATE ON escalations
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update visitor last_visit_at when new ticket created
CREATE OR REPLACE FUNCTION update_visitor_last_visit()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE visitor_sessions
  SET
    last_visit_at = NOW(),
    visit_count = visit_count + 1,
    updated_at = NOW()
  WHERE visitor_id = NEW.visitor_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_visitor_on_ticket
  AFTER INSERT ON tickets
  FOR EACH ROW
  EXECUTE FUNCTION update_visitor_last_visit();

-- Trigger: Auto-set escalated_at on tickets when escalation created
CREATE OR REPLACE FUNCTION set_ticket_escalated()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE tickets
  SET
    status = 'escalated',
    escalated_at = NEW.escalated_at,
    updated_at = NOW()
  WHERE ticket_id = NEW.ticket_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_ticket_escalated
  AFTER INSERT ON escalations
  FOR EACH ROW
  EXECUTE FUNCTION set_ticket_escalated();

COMMENT ON FUNCTION update_visitor_last_visit IS 'Updates visitor last_visit_at and increments visit_count when new ticket created';
COMMENT ON FUNCTION set_ticket_escalated IS 'Sets ticket status to escalated when escalation record created';
