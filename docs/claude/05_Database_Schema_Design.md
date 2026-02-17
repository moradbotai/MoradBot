# MoradBot Database Schema Design

**Database:** Supabase (PostgreSQL)
**Date:** February 17, 2026
**Version:** 1.0
**Phase:** MVP - FAQ Automation Only

---

## Design Principles

1. **Zero-Tolerance Multi-Tenant Isolation**
   - Every table has `store_id` as partition key
   - RLS policies enforce strict data boundaries
   - No cross-store queries allowed

2. **Data Retention Policy**
   - Active subscription: Full data retention
   - Post-cancellation: 30-90 days retention
   - Personal data encrypted at rest

3. **Performance Optimization**
   - Composite indexes on frequently queried columns
   - Partitioning by store_id where applicable
   - Efficient FK constraints

4. **Audit Trail**
   - `created_at`, `updated_at` on all tables
   - Soft deletes where needed (`deleted_at`)

---

## Entity Relationship Diagram

```
┌─────────────┐
│   stores    │◄──────┐
└─────────────┘       │
       │              │
       ├──────────────┼────────────┐
       │              │            │
       ▼              ▼            ▼
┌────────────┐  ┌──────────────┐  ┌────────────────┐
│    faqs    │  │   products   │  │ usage_tracking │
└────────────┘  │  _snapshots  │  └────────────────┘
                └──────────────┘
                       │
       ┌───────────────┴────────────┐
       │                            │
       ▼                            ▼
┌────────────┐              ┌────────────┐
│  visitors  │              │  tickets   │
└────────────┘              └────────────┘
       │                            │
       │                    ┌───────┴──────┐
       │                    │              │
       │                    ▼              ▼
       │            ┌────────────┐  ┌──────────────┐
       └───────────►│  messages  │  │ escalations  │
                    └────────────┘  └──────────────┘
```

---

## Table Definitions

### 1. stores
**Purpose:** Store merchant accounts and settings

```sql
CREATE TABLE stores (
  -- Primary Key
  store_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Salla Integration
  salla_merchant_id VARCHAR(255) UNIQUE NOT NULL,
  salla_access_token TEXT,
  salla_refresh_token TEXT,
  salla_token_expires_at TIMESTAMPTZ,

  -- Store Info
  store_name_ar VARCHAR(255) NOT NULL,
  store_url VARCHAR(500) NOT NULL,

  -- Subscription
  plan_tier VARCHAR(20) NOT NULL DEFAULT 'basic' CHECK (plan_tier IN ('basic', 'mid', 'premium')),
  subscription_status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (subscription_status IN ('active', 'cancelled', 'suspended')),
  subscription_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  subscription_ends_at TIMESTAMPTZ,

  -- Usage Limits
  usage_limit_per_cycle INTEGER NOT NULL DEFAULT 500,
  current_cycle_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  current_cycle_end TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 month'),

  -- Product Sync Schedule
  sync_frequency_hours INTEGER NOT NULL DEFAULT 24 CHECK (sync_frequency_hours IN (1, 6, 24)),
  last_sync_at TIMESTAMPTZ,
  next_sync_at TIMESTAMPTZ,

  -- Widget Settings
  bot_enabled BOOLEAN NOT NULL DEFAULT true,
  widget_color VARCHAR(7) DEFAULT '#2563eb',
  branding_enabled BOOLEAN NOT NULL DEFAULT true,

  -- Contact Info (for escalations)
  contact_email VARCHAR(255),
  contact_phone VARCHAR(20),

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_stores_salla_merchant ON stores(salla_merchant_id);
CREATE INDEX idx_stores_subscription ON stores(subscription_status) WHERE deleted_at IS NULL;
CREATE INDEX idx_stores_next_sync ON stores(next_sync_at) WHERE bot_enabled = true AND deleted_at IS NULL;
```

---

### 2. faqs
**Purpose:** Store FAQ data provided by merchants

```sql
CREATE TABLE faqs (
  -- Primary Key
  faq_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Key
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- FAQ Data
  category VARCHAR(50) NOT NULL CHECK (category IN ('shipping', 'payment', 'returns', 'products', 'general')),
  question_ar TEXT NOT NULL,
  answer_ar TEXT NOT NULL,

  -- Metadata
  is_active BOOLEAN NOT NULL DEFAULT true,
  usage_count INTEGER NOT NULL DEFAULT 0,
  last_used_at TIMESTAMPTZ,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_faqs_store ON faqs(store_id, is_active);
CREATE INDEX idx_faqs_category ON faqs(store_id, category) WHERE is_active = true;
CREATE INDEX idx_faqs_search ON faqs USING GIN (to_tsvector('arabic', question_ar || ' ' || answer_ar));
```

---

### 3. products_snapshots
**Purpose:** Periodic snapshots of Salla products

```sql
CREATE TABLE products_snapshots (
  -- Primary Key
  snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Key
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- Product Data (from Salla API)
  salla_product_id VARCHAR(255) NOT NULL,
  name_ar VARCHAR(500) NOT NULL,
  description_ar TEXT,
  price DECIMAL(10, 2) NOT NULL,
  currency VARCHAR(3) NOT NULL DEFAULT 'SAR',
  available BOOLEAN NOT NULL DEFAULT true,
  stock_quantity INTEGER,
  image_url TEXT,
  category_ar VARCHAR(255),
  sku VARCHAR(100),

  -- Snapshot Metadata
  snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  is_latest BOOLEAN NOT NULL DEFAULT true,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Constraints
  UNIQUE(store_id, salla_product_id, snapshot_timestamp)
);

-- Indexes
CREATE INDEX idx_products_store_latest ON products_snapshots(store_id, is_latest) WHERE is_latest = true;
CREATE INDEX idx_products_salla_id ON products_snapshots(store_id, salla_product_id) WHERE is_latest = true;
CREATE INDEX idx_products_search ON products_snapshots USING GIN (to_tsvector('arabic', name_ar || ' ' || COALESCE(description_ar, ''))) WHERE is_latest = true;
CREATE INDEX idx_products_available ON products_snapshots(store_id, available) WHERE is_latest = true AND available = true;
```

**Note:** When a new sync happens:
1. Set `is_latest = false` for all previous snapshots of the store
2. Insert new products with `is_latest = true`

---

### 4. visitors
**Purpose:** Track unique visitors (cookie/session-based)

```sql
CREATE TABLE visitors (
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
  visit_count INTEGER NOT NULL DEFAULT 1,

  -- User Agent
  user_agent TEXT,
  ip_address INET,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Constraints
  UNIQUE(store_id, session_cookie)
);

-- Indexes
CREATE INDEX idx_visitors_store ON visitors(store_id);
CREATE INDEX idx_visitors_cookie ON visitors(session_cookie);
CREATE INDEX idx_visitors_last_visit ON visitors(store_id, last_visit_at DESC);
```

---

### 5. tickets
**Purpose:** Conversation threads

```sql
CREATE TABLE tickets (
  -- Primary Key
  ticket_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  visitor_id UUID NOT NULL REFERENCES visitors(visitor_id) ON DELETE CASCADE,

  -- Ticket Status
  status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'escalated', 'closed')),
  resolution_type VARCHAR(50) CHECK (resolution_type IN ('bot_answered', 'escalated', 'auto_closed', 'merchant_closed')),

  -- Context
  initial_question TEXT,
  category VARCHAR(50),

  -- Clarification Tracking
  clarification_count INTEGER NOT NULL DEFAULT 0,

  -- Timing
  opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMPTZ,
  escalated_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_tickets_store ON tickets(store_id, status);
CREATE INDEX idx_tickets_visitor ON tickets(visitor_id, created_at DESC);
CREATE INDEX idx_tickets_status ON tickets(store_id, status, updated_at DESC);
CREATE INDEX idx_tickets_escalated ON tickets(store_id, escalated_at) WHERE status = 'escalated';
```

---

### 6. messages
**Purpose:** Individual messages within tickets

```sql
CREATE TABLE messages (
  -- Primary Key
  message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  ticket_id UUID NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- Message Content
  sender_type VARCHAR(20) NOT NULL CHECK (sender_type IN ('visitor', 'bot', 'merchant')),
  content_ar TEXT NOT NULL,

  -- Metadata
  is_clarification_request BOOLEAN NOT NULL DEFAULT false,
  includes_dynamic_data BOOLEAN NOT NULL DEFAULT false,

  -- AI Metadata (for bot messages)
  model_used VARCHAR(50),
  tokens_used INTEGER,
  response_time_ms INTEGER,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_messages_ticket ON messages(ticket_id, created_at ASC);
CREATE INDEX idx_messages_store ON messages(store_id, created_at DESC);
CREATE INDEX idx_messages_bot ON messages(store_id, sender_type, created_at DESC) WHERE sender_type = 'bot';
```

---

### 7. escalations
**Purpose:** Escalated tickets requiring merchant attention

```sql
CREATE TABLE escalations (
  -- Primary Key
  escalation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  ticket_id UUID NOT NULL UNIQUE REFERENCES tickets(ticket_id) ON DELETE CASCADE,
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  visitor_id UUID NOT NULL REFERENCES visitors(visitor_id) ON DELETE CASCADE,

  -- Escalation Details
  reason VARCHAR(100) NOT NULL CHECK (reason IN ('failed_clarification', 'unsupported_request', 'manual_request', 'error')),
  problem_description TEXT NOT NULL,

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

-- Indexes
CREATE INDEX idx_escalations_store_status ON escalations(store_id, status, escalated_at DESC);
CREATE INDEX idx_escalations_pending ON escalations(store_id, escalated_at DESC) WHERE status = 'pending';
```

---

### 8. usage_tracking
**Purpose:** Track bot reply usage for billing

```sql
CREATE TABLE usage_tracking (
  -- Primary Key
  usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Keys
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,
  ticket_id UUID REFERENCES tickets(ticket_id) ON DELETE SET NULL,
  message_id UUID REFERENCES messages(message_id) ON DELETE SET NULL,

  -- Usage Type
  event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('bot_reply', 'clarification', 'escalation')),

  -- Billing Cycle
  billing_cycle_start TIMESTAMPTZ NOT NULL,
  billing_cycle_end TIMESTAMPTZ NOT NULL,

  -- Cost Tracking (Internal)
  tokens_used INTEGER,
  estimated_cost_usd DECIMAL(10, 6),
  model_used VARCHAR(50),

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_usage_store_cycle ON usage_tracking(store_id, billing_cycle_start, billing_cycle_end);
CREATE INDEX idx_usage_event_type ON usage_tracking(store_id, event_type, created_at DESC);
CREATE INDEX idx_usage_created ON usage_tracking(created_at) WHERE event_type = 'bot_reply';
```

---

## Triggers & Functions

### 1. Update `updated_at` timestamp

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_stores_updated_at BEFORE UPDATE ON stores
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_faqs_updated_at BEFORE UPDATE ON faqs
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_visitors_updated_at BEFORE UPDATE ON visitors
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tickets_updated_at BEFORE UPDATE ON tickets
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_escalations_updated_at BEFORE UPDATE ON escalations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### 2. Auto-update visitor last_visit_at

```sql
CREATE OR REPLACE FUNCTION update_visitor_last_visit()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE visitors
  SET
    last_visit_at = NOW(),
    visit_count = visit_count + 1
  WHERE visitor_id = NEW.visitor_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ticket_updates_visitor AFTER INSERT ON tickets
  FOR EACH ROW EXECUTE FUNCTION update_visitor_last_visit();
```

### 3. Increment FAQ usage count

```sql
CREATE OR REPLACE FUNCTION increment_faq_usage()
RETURNS TRIGGER AS $$
BEGIN
  -- Logic to be implemented in application layer
  -- This is a placeholder for future enhancement
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## Views for Common Queries

### 1. Active Stores with Current Usage

```sql
CREATE VIEW v_active_stores_usage AS
SELECT
  s.store_id,
  s.store_name_ar,
  s.plan_tier,
  s.usage_limit_per_cycle,
  s.current_cycle_start,
  s.current_cycle_end,
  COUNT(ut.usage_id) FILTER (WHERE ut.event_type = 'bot_reply') AS current_cycle_usage,
  (s.usage_limit_per_cycle - COUNT(ut.usage_id) FILTER (WHERE ut.event_type = 'bot_reply')) AS remaining_usage
FROM stores s
LEFT JOIN usage_tracking ut ON s.store_id = ut.store_id
  AND ut.billing_cycle_start = s.current_cycle_start
WHERE s.subscription_status = 'active'
  AND s.bot_enabled = true
  AND s.deleted_at IS NULL
GROUP BY s.store_id;
```

### 2. Pending Escalations Dashboard

```sql
CREATE VIEW v_pending_escalations AS
SELECT
  e.escalation_id,
  e.store_id,
  s.store_name_ar,
  e.ticket_id,
  e.problem_description,
  e.contact_method,
  e.order_number,
  e.escalated_at,
  v.session_cookie,
  COUNT(m.message_id) AS message_count
FROM escalations e
JOIN stores s ON e.store_id = s.store_id
JOIN visitors v ON e.visitor_id = v.visitor_id
JOIN tickets t ON e.ticket_id = t.ticket_id
LEFT JOIN messages m ON t.ticket_id = m.ticket_id
WHERE e.status = 'pending'
GROUP BY e.escalation_id, s.store_name_ar, v.session_cookie
ORDER BY e.escalated_at DESC;
```

---

## Data Retention Policy Implementation

### Anonymization after cancellation

```sql
CREATE OR REPLACE FUNCTION anonymize_store_data(target_store_id UUID)
RETURNS VOID AS $$
BEGIN
  -- Anonymize visitor personal data
  UPDATE visitors
  SET
    email_encrypted = NULL,
    phone_encrypted = NULL,
    name_encrypted = NULL,
    ip_address = NULL,
    user_agent = NULL
  WHERE store_id = target_store_id;

  -- Anonymize escalation contact info
  UPDATE escalations
  SET contact_value_encrypted = NULL
  WHERE store_id = target_store_id;

  -- Optionally delete old tickets (retain only aggregated stats)
  -- Implementation depends on retention period
END;
$$ LANGUAGE plpgsql;
```

---

## Summary

### Tables: 8
1. `stores` - Merchant accounts
2. `faqs` - FAQ data
3. `products_snapshots` - Product sync data
4. `visitors` - Unique visitors
5. `tickets` - Conversation threads
6. `messages` - Individual messages
7. `escalations` - Escalated cases
8. `usage_tracking` - Billing/usage metrics

### Key Features
- ✅ Multi-tenant isolation via `store_id`
- ✅ Encrypted personal data
- ✅ Soft deletes
- ✅ Audit trails
- ✅ Performance indexes
- ✅ Triggers for automation
- ✅ Views for common queries

**Next Steps:**
1. Define RLS policies (Task #7)
2. Create migration files (Task #9)
3. Test schema with sample data
