-- Migration 4: Tracking and Compliance Tables
-- Created: 2026-02-17
-- Description: Usage events, consent logs, and audit logs

-- ============================================
-- 1. USAGE_EVENTS TABLE
-- ============================================

CREATE TABLE usage_events (
  -- Primary Key
  event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  subscription_id UUID REFERENCES subscriptions(subscription_id) ON DELETE SET NULL,
  ticket_id UUID REFERENCES tickets(ticket_id) ON DELETE SET NULL,
  message_id UUID REFERENCES messages(message_id) ON DELETE SET NULL,

  -- Event Type
  event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('bot_reply', 'clarification', 'escalation')),

  -- Billing Cycle
  billing_cycle_start TIMESTAMPTZ NOT NULL,
  billing_cycle_end TIMESTAMPTZ NOT NULL,

  -- Cost Tracking (Internal)
  tokens_used INTEGER CHECK (tokens_used >= 0),
  estimated_cost_usd DECIMAL(10, 6) CHECK (estimated_cost_usd >= 0),
  model_used VARCHAR(50),

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE usage_events IS 'Bot reply usage events for billing and analytics';
COMMENT ON COLUMN usage_events.event_type IS 'Type of billable event';
COMMENT ON COLUMN usage_events.estimated_cost_usd IS 'Internal cost tracking for AI model usage';

-- Indexes
CREATE INDEX idx_usage_store_cycle ON usage_events(store_id, billing_cycle_start, billing_cycle_end);
CREATE INDEX idx_usage_event_type ON usage_events(store_id, event_type, created_at DESC);
CREATE INDEX idx_usage_created ON usage_events(created_at DESC);
CREATE INDEX idx_usage_subscription ON usage_events(subscription_id) WHERE subscription_id IS NOT NULL;

-- Partitioning recommendation (future): Partition by created_at monthly

-- ============================================
-- 2. CONSENT_LOGS TABLE
-- ============================================

CREATE TABLE consent_logs (
  -- Primary Key
  consent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  visitor_id UUID NOT NULL REFERENCES visitor_sessions(visitor_id) ON DELETE CASCADE,
  ticket_id UUID REFERENCES tickets(ticket_id) ON DELETE SET NULL,

  -- Consent Details
  consent_type VARCHAR(50) NOT NULL CHECK (consent_type IN ('personal_data_storage', 'persistent_memory')),
  consent_given BOOLEAN NOT NULL,
  consent_method VARCHAR(50) NOT NULL CHECK (consent_method IN ('chat_checkbox', 'explicit_message', 'system_default')),

  -- Context
  ip_address INET,
  user_agent TEXT,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE consent_logs IS 'Immutable log of user consent events for PDPL compliance';
COMMENT ON COLUMN consent_logs.consent_type IS 'Type of consent: personal data or persistent memory';
COMMENT ON COLUMN consent_logs.consent_method IS 'How consent was captured';

-- Indexes
CREATE INDEX idx_consent_visitor ON consent_logs(visitor_id, created_at DESC);
CREATE INDEX idx_consent_store ON consent_logs(store_id, created_at DESC);
CREATE INDEX idx_consent_type ON consent_logs(consent_type, created_at DESC);

-- No UPDATE or DELETE allowed on consent_logs (append-only for compliance)
CREATE RULE consent_logs_no_update AS ON UPDATE TO consent_logs DO INSTEAD NOTHING;
CREATE RULE consent_logs_no_delete AS ON DELETE TO consent_logs DO INSTEAD NOTHING;

COMMENT ON TABLE consent_logs IS 'Append-only consent log - no updates or deletes allowed';

-- ============================================
-- 3. AUDIT_LOGS TABLE
-- ============================================

CREATE TABLE audit_logs (
  -- Primary Key
  audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Key (nullable for system-level events)
  store_id UUID REFERENCES stores(store_id) ON DELETE CASCADE,

  -- Actor Info
  actor_type VARCHAR(20) NOT NULL CHECK (actor_type IN ('merchant', 'staff', 'system', 'api')),
  actor_id VARCHAR(255),

  -- Action
  action VARCHAR(100) NOT NULL,

  -- Resource
  resource_type VARCHAR(50),
  resource_id UUID,

  -- Context
  ip_address INET,
  user_agent TEXT,
  metadata JSONB DEFAULT '{}'::jsonb,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for security and compliance';
COMMENT ON COLUMN audit_logs.actor_type IS 'Who performed the action';
COMMENT ON COLUMN audit_logs.action IS 'What action was performed (e.g., login, access_conversation, toggle_bot)';
COMMENT ON COLUMN audit_logs.metadata IS 'Additional context as JSON';

-- Indexes
CREATE INDEX idx_audit_store ON audit_logs(store_id, created_at DESC) WHERE store_id IS NOT NULL;
CREATE INDEX idx_audit_actor ON audit_logs(actor_id, created_at DESC) WHERE actor_id IS NOT NULL;
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id, created_at DESC);
CREATE INDEX idx_audit_action ON audit_logs(action, created_at DESC);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);

-- No UPDATE or DELETE allowed on audit_logs (append-only for compliance)
CREATE RULE audit_logs_no_update AS ON UPDATE TO audit_logs DO INSTEAD NOTHING;
CREATE RULE audit_logs_no_delete AS ON DELETE TO audit_logs DO INSTEAD NOTHING;

COMMENT ON TABLE audit_logs IS 'Append-only audit log - no updates or deletes allowed. Retention >= 90 days per NFR';

-- ============================================
-- 4. TRIGGERS
-- ============================================

-- Trigger: Increment subscription usage when bot_reply event created
CREATE OR REPLACE FUNCTION increment_subscription_usage()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.event_type = 'bot_reply' AND NEW.subscription_id IS NOT NULL THEN
    UPDATE subscriptions
    SET
      current_cycle_usage = current_cycle_usage + 1,
      updated_at = NOW()
    WHERE subscription_id = NEW.subscription_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_increment_subscription_usage
  AFTER INSERT ON usage_events
  FOR EACH ROW
  EXECUTE FUNCTION increment_subscription_usage();

COMMENT ON FUNCTION increment_subscription_usage IS 'Increments subscription usage counter when bot_reply event is logged';

-- ============================================
-- 5. HELPER FUNCTIONS
-- ============================================

-- Function: Log audit event
CREATE OR REPLACE FUNCTION log_audit_event(
  p_store_id UUID,
  p_actor_type VARCHAR,
  p_actor_id VARCHAR,
  p_action VARCHAR,
  p_resource_type VARCHAR DEFAULT NULL,
  p_resource_id UUID DEFAULT NULL,
  p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
  v_audit_id UUID;
BEGIN
  INSERT INTO audit_logs (
    store_id,
    actor_type,
    actor_id,
    action,
    resource_type,
    resource_id,
    metadata
  ) VALUES (
    p_store_id,
    p_actor_type,
    p_actor_id,
    p_action,
    p_resource_type,
    p_resource_id,
    p_metadata
  ) RETURNING audit_id INTO v_audit_id;

  RETURN v_audit_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION log_audit_event IS 'Helper function to insert audit log entries';

-- Function: Log consent event
CREATE OR REPLACE FUNCTION log_consent_event(
  p_store_id UUID,
  p_visitor_id UUID,
  p_ticket_id UUID,
  p_consent_type VARCHAR,
  p_consent_given BOOLEAN,
  p_consent_method VARCHAR,
  p_ip_address INET DEFAULT NULL,
  p_user_agent TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
  v_consent_id UUID;
BEGIN
  INSERT INTO consent_logs (
    store_id,
    visitor_id,
    ticket_id,
    consent_type,
    consent_given,
    consent_method,
    ip_address,
    user_agent
  ) VALUES (
    p_store_id,
    p_visitor_id,
    p_ticket_id,
    p_consent_type,
    p_consent_given,
    p_consent_method,
    p_ip_address,
    p_user_agent
  ) RETURNING consent_id INTO v_consent_id;

  -- Also update visitor_sessions if consent given
  IF p_consent_given THEN
    UPDATE visitor_sessions
    SET
      consent_given = true,
      consent_given_at = NOW(),
      updated_at = NOW()
    WHERE visitor_id = p_visitor_id;
  END IF;

  RETURN v_consent_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION log_consent_event IS 'Helper function to log consent and update visitor session';
