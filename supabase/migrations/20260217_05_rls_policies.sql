-- Migration 5: Row Level Security (RLS) Policies
-- Created: 2026-02-17
-- Description: Strict RLS policies for multi-tenant isolation

-- ============================================
-- IMPORTANT: Multi-Tenant Security
-- ============================================
-- All policies enforce store_id isolation
-- Zero-tolerance for cross-store data access
-- Merchants can only access their own store data

-- ============================================
-- 1. PLANS TABLE - Read-only for all
-- ============================================

ALTER TABLE plans ENABLE ROW LEVEL SECURITY;

-- Anyone can view plans
CREATE POLICY "plans_select_all"
  ON plans FOR SELECT
  USING (true);

-- Only system can modify plans (via service role)
CREATE POLICY "plans_modify_service_role_only"
  ON plans FOR ALL
  USING (false)
  WITH CHECK (false);

COMMENT ON POLICY "plans_select_all" ON plans IS 'All users can view available plans';

-- ============================================
-- 2. STORES TABLE
-- ============================================

ALTER TABLE stores ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own store
CREATE POLICY "stores_select_own"
  ON stores FOR SELECT
  USING (store_id = auth.uid());

-- Merchants can update their own store
CREATE POLICY "stores_update_own"
  ON stores FOR UPDATE
  USING (store_id = auth.uid())
  WITH CHECK (store_id = auth.uid());

-- Only system can insert/delete stores (via service role)
CREATE POLICY "stores_insert_service_role_only"
  ON stores FOR INSERT
  WITH CHECK (false);

CREATE POLICY "stores_delete_service_role_only"
  ON stores FOR DELETE
  USING (false);

COMMENT ON POLICY "stores_select_own" ON stores IS 'Merchants can only view their own store';

-- ============================================
-- 3. SUBSCRIPTIONS TABLE
-- ============================================

ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own subscription
CREATE POLICY "subscriptions_select_own"
  ON subscriptions FOR SELECT
  USING (store_id = auth.uid());

-- Only system can modify subscriptions
CREATE POLICY "subscriptions_modify_service_role_only"
  ON subscriptions FOR ALL
  USING (false)
  WITH CHECK (false);

-- ============================================
-- 4. FAQ_ENTRIES TABLE
-- ============================================

ALTER TABLE faq_entries ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own FAQs
CREATE POLICY "faq_entries_select_own"
  ON faq_entries FOR SELECT
  USING (store_id = auth.uid());

-- Merchants can insert their own FAQs
CREATE POLICY "faq_entries_insert_own"
  ON faq_entries FOR INSERT
  WITH CHECK (store_id = auth.uid());

-- Merchants can update their own FAQs
CREATE POLICY "faq_entries_update_own"
  ON faq_entries FOR UPDATE
  USING (store_id = auth.uid())
  WITH CHECK (store_id = auth.uid());

-- Merchants can delete their own FAQs
CREATE POLICY "faq_entries_delete_own"
  ON faq_entries FOR DELETE
  USING (store_id = auth.uid());

-- ============================================
-- 5. PRODUCT_SNAPSHOTS TABLE
-- ============================================

ALTER TABLE product_snapshots ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own product snapshots
CREATE POLICY "product_snapshots_select_own"
  ON product_snapshots FOR SELECT
  USING (store_id = auth.uid());

-- Only system can insert/update product snapshots (sync service)
CREATE POLICY "product_snapshots_modify_service_role_only"
  ON product_snapshots FOR ALL
  USING (false)
  WITH CHECK (false);

-- ============================================
-- 6. VISITOR_SESSIONS TABLE
-- ============================================

ALTER TABLE visitor_sessions ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own visitors
CREATE POLICY "visitor_sessions_select_own"
  ON visitor_sessions FOR SELECT
  USING (store_id = auth.uid());

-- Only system can modify visitor sessions
CREATE POLICY "visitor_sessions_modify_service_role_only"
  ON visitor_sessions FOR ALL
  USING (false)
  WITH CHECK (false);

-- ============================================
-- 7. TICKETS TABLE
-- ============================================

ALTER TABLE tickets ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own tickets
CREATE POLICY "tickets_select_own"
  ON tickets FOR SELECT
  USING (store_id = auth.uid());

-- Merchants can update their own tickets (for manual closure)
CREATE POLICY "tickets_update_own"
  ON tickets FOR UPDATE
  USING (store_id = auth.uid())
  WITH CHECK (store_id = auth.uid());

-- Only system can insert/delete tickets
CREATE POLICY "tickets_insert_service_role_only"
  ON tickets FOR INSERT
  WITH CHECK (false);

CREATE POLICY "tickets_delete_service_role_only"
  ON tickets FOR DELETE
  USING (false);

-- ============================================
-- 8. MESSAGES TABLE
-- ============================================

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own messages
CREATE POLICY "messages_select_own"
  ON messages FOR SELECT
  USING (store_id = auth.uid());

-- Only system can insert messages
CREATE POLICY "messages_insert_service_role_only"
  ON messages FOR INSERT
  WITH CHECK (false);

-- No updates or deletes on messages
CREATE POLICY "messages_no_update"
  ON messages FOR UPDATE
  USING (false);

CREATE POLICY "messages_no_delete"
  ON messages FOR DELETE
  USING (false);

COMMENT ON POLICY "messages_no_update" ON messages IS 'Messages are immutable once created';

-- ============================================
-- 9. ESCALATIONS TABLE
-- ============================================

ALTER TABLE escalations ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own escalations
CREATE POLICY "escalations_select_own"
  ON escalations FOR SELECT
  USING (store_id = auth.uid());

-- Merchants can update their own escalations (for resolution)
CREATE POLICY "escalations_update_own"
  ON escalations FOR UPDATE
  USING (store_id = auth.uid())
  WITH CHECK (store_id = auth.uid());

-- Only system can insert escalations
CREATE POLICY "escalations_insert_service_role_only"
  ON escalations FOR INSERT
  WITH CHECK (false);

-- No deletes on escalations
CREATE POLICY "escalations_no_delete"
  ON escalations FOR DELETE
  USING (false);

-- ============================================
-- 10. USAGE_EVENTS TABLE
-- ============================================

ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own usage events
CREATE POLICY "usage_events_select_own"
  ON usage_events FOR SELECT
  USING (store_id = auth.uid());

-- Only system can insert usage events
CREATE POLICY "usage_events_insert_service_role_only"
  ON usage_events FOR INSERT
  WITH CHECK (false);

-- No updates or deletes on usage events
CREATE POLICY "usage_events_no_update"
  ON usage_events FOR UPDATE
  USING (false);

CREATE POLICY "usage_events_no_delete"
  ON usage_events FOR DELETE
  USING (false);

COMMENT ON POLICY "usage_events_no_update" ON usage_events IS 'Usage events are immutable for billing integrity';

-- ============================================
-- 11. CONSENT_LOGS TABLE
-- ============================================

ALTER TABLE consent_logs ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own consent logs
CREATE POLICY "consent_logs_select_own"
  ON consent_logs FOR SELECT
  USING (store_id = auth.uid());

-- Only system can insert consent logs
CREATE POLICY "consent_logs_insert_service_role_only"
  ON consent_logs FOR INSERT
  WITH CHECK (false);

-- No updates or deletes (enforced by RULEs already, but adding for clarity)
CREATE POLICY "consent_logs_no_update"
  ON consent_logs FOR UPDATE
  USING (false);

CREATE POLICY "consent_logs_no_delete"
  ON consent_logs FOR DELETE
  USING (false);

COMMENT ON POLICY "consent_logs_no_update" ON consent_logs IS 'Consent logs are immutable for PDPL compliance';

-- ============================================
-- 12. AUDIT_LOGS TABLE
-- ============================================

ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Merchants can view their own audit logs
CREATE POLICY "audit_logs_select_own"
  ON audit_logs FOR SELECT
  USING (store_id = auth.uid() OR store_id IS NULL);

-- Only system can insert audit logs
CREATE POLICY "audit_logs_insert_service_role_only"
  ON audit_logs FOR INSERT
  WITH CHECK (false);

-- No updates or deletes (enforced by RULEs already, but adding for clarity)
CREATE POLICY "audit_logs_no_update"
  ON audit_logs FOR UPDATE
  USING (false);

CREATE POLICY "audit_logs_no_delete"
  ON audit_logs FOR DELETE
  USING (false);

COMMENT ON POLICY "audit_logs_no_update" ON audit_logs IS 'Audit logs are immutable for compliance';

-- ============================================
-- 13. VIEWS - No RLS needed (inherit from tables)
-- ============================================

-- Views automatically inherit RLS from underlying tables
-- v_active_subscriptions
-- v_pending_escalations

-- ============================================
-- SECURITY NOTES
-- ============================================

-- 1. All policies use auth.uid() which should be set to store_id
-- 2. Service role bypasses RLS for system operations
-- 3. Append-only tables (consent_logs, audit_logs, messages, usage_events)
--    have no UPDATE/DELETE policies
-- 4. Cross-store data access is IMPOSSIBLE with these policies
-- 5. Any RLS policy failure = immediate access denial

-- ============================================
-- TESTING RLS POLICIES
-- ============================================

-- Test queries (run as authenticated merchant):
-- Should succeed:
--   SELECT * FROM stores WHERE store_id = auth.uid();
--   SELECT * FROM tickets WHERE store_id = auth.uid();
--
-- Should fail (no rows returned):
--   SELECT * FROM stores WHERE store_id != auth.uid();
--   SELECT * FROM tickets WHERE store_id != auth.uid();

COMMENT ON SCHEMA public IS 'All tables have RLS enabled with strict multi-tenant isolation';
