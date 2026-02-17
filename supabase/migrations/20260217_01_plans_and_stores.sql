-- Migration 1: Plans, Stores, and Subscriptions
-- Created: 2026-02-17
-- Description: Core subscription and billing tables

-- ============================================
-- 1. PLANS TABLE
-- ============================================

CREATE TABLE plans (
  -- Primary Key
  plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Plan Details
  plan_name VARCHAR(20) NOT NULL UNIQUE CHECK (plan_name IN ('basic', 'mid', 'premium')),
  plan_name_ar VARCHAR(50) NOT NULL,
  bot_reply_limit INTEGER NOT NULL CHECK (bot_reply_limit > 0),
  sync_frequency_hours INTEGER NOT NULL CHECK (sync_frequency_hours IN (1, 6, 24)),
  price_monthly_sar DECIMAL(10, 2) NOT NULL CHECK (price_monthly_sar >= 0),

  -- Features
  features JSONB DEFAULT '{}'::jsonb,

  -- Status
  is_active BOOLEAN NOT NULL DEFAULT true,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE plans IS 'Available subscription plans with limits and pricing';
COMMENT ON COLUMN plans.bot_reply_limit IS 'Maximum bot replies per billing cycle';
COMMENT ON COLUMN plans.sync_frequency_hours IS 'Product sync frequency in hours (1, 6, or 24)';

-- Index
CREATE INDEX idx_plans_active ON plans(is_active) WHERE is_active = true;

-- ============================================
-- 2. STORES TABLE
-- ============================================

CREATE TABLE stores (
  -- Primary Key
  store_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Salla Integration
  salla_merchant_id VARCHAR(255) UNIQUE NOT NULL,
  salla_access_token TEXT,  -- Will be encrypted at application level
  salla_refresh_token TEXT, -- Will be encrypted at application level
  salla_token_expires_at TIMESTAMPTZ,

  -- Store Info
  store_name_ar VARCHAR(255) NOT NULL,
  store_url VARCHAR(500) NOT NULL,

  -- Contact Info (for escalations)
  contact_email VARCHAR(255),
  contact_phone VARCHAR(20),

  -- Widget Settings
  bot_enabled BOOLEAN NOT NULL DEFAULT true,
  widget_settings JSONB DEFAULT '{
    "color": "#2563eb",
    "branding_enabled": true,
    "position": "bottom-right"
  }'::jsonb,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at TIMESTAMPTZ
);

COMMENT ON TABLE stores IS 'Merchant stores connected via Salla OAuth';
COMMENT ON COLUMN stores.salla_access_token IS 'Salla OAuth access token (encrypted at app level)';
COMMENT ON COLUMN stores.widget_settings IS 'JSON configuration for chat widget appearance';

-- Indexes
CREATE INDEX idx_stores_salla_merchant ON stores(salla_merchant_id);
CREATE INDEX idx_stores_active ON stores(bot_enabled) WHERE bot_enabled = true AND deleted_at IS NULL;
CREATE INDEX idx_stores_deleted ON stores(deleted_at) WHERE deleted_at IS NOT NULL;

-- ============================================
-- 3. SUBSCRIPTIONS TABLE
-- ============================================

CREATE TABLE subscriptions (
  -- Primary Key
  subscription_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  plan_id UUID NOT NULL REFERENCES plans(plan_id) ON DELETE RESTRICT,

  -- Subscription Status
  status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled', 'suspended')),

  -- Billing Period
  started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ends_at TIMESTAMPTZ,

  -- Current Billing Cycle
  current_cycle_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  current_cycle_end TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 month'),
  current_cycle_usage INTEGER NOT NULL DEFAULT 0 CHECK (current_cycle_usage >= 0),

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Constraints
  CONSTRAINT one_active_subscription_per_store UNIQUE (store_id) WHERE (status = 'active')
);

COMMENT ON TABLE subscriptions IS 'Store subscriptions to MoradBot plans';
COMMENT ON COLUMN subscriptions.current_cycle_usage IS 'Bot replies count in current billing cycle';

-- Indexes
CREATE INDEX idx_subscriptions_store ON subscriptions(store_id, status);
CREATE INDEX idx_subscriptions_status ON subscriptions(status, current_cycle_end) WHERE status = 'active';
CREATE INDEX idx_subscriptions_cycle ON subscriptions(current_cycle_start, current_cycle_end) WHERE status = 'active';

-- ============================================
-- 4. TRIGGERS
-- ============================================

-- Trigger: Update updated_at on plans
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_plans_updated_at
  BEFORE UPDATE ON plans
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stores_updated_at
  BEFORE UPDATE ON stores
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at
  BEFORE UPDATE ON subscriptions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 5. SEED DATA FOR PLANS
-- ============================================

INSERT INTO plans (plan_name, plan_name_ar, bot_reply_limit, sync_frequency_hours, price_monthly_sar, features) VALUES
('basic', 'الخطة الأساسية', 500, 24, 99.00, '{
  "support": "email",
  "branding": false,
  "analytics": "basic"
}'::jsonb),
('mid', 'الخطة المتوسطة', 2000, 6, 299.00, '{
  "support": "priority_email",
  "branding": true,
  "analytics": "advanced"
}'::jsonb),
('premium', 'الخطة المتقدمة', 10000, 1, 799.00, '{
  "support": "phone_email",
  "branding": true,
  "analytics": "premium",
  "dedicated_support": true
}'::jsonb);

COMMENT ON TABLE plans IS 'Seeded with 3 default plans: basic, mid, premium';
