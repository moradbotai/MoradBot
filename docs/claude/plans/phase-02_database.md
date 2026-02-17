# خطة المرحلة 2: تصميم قاعدة البيانات

**التاريخ:** 17 فبراير 2026
**الحالة:** قيد التنفيذ
**المدة المتوقعة:** يوم واحد

---

## 1. الهدف

تصميم وتنفيذ قاعدة بيانات Supabase (PostgreSQL) كاملة لـ MoradBot MVP مع:
- 12 جدول رئيسي
- RLS policies لعزل البيانات
- TypeScript types للنوع الآمن
- Migration files قابلة للتنفيذ

---

## 2. المبادئ الأساسية

### 2.1 عزل البيانات (Zero-Tolerance)
- كل جدول يحتوي على `store_id` (FK)
- RLS policies صارمة على جميع الجداول
- لا استعلامات عابرة للمتاجر

### 2.2 الأمان والخصوصية
- تشفير البيانات الحساسة (email, phone)
- consent_logs لتتبع الموافقات
- audit_logs لجميع الأحداث الهامة

### 2.3 الأداء
- Indexes على الأعمدة المستخدمة بكثرة
- Composite indexes حيث مناسب
- Partitioning للجداول الكبيرة (messages, usage_events)

### 2.4 قابلية التدقيق
- `created_at`, `updated_at` في كل جدول
- soft deletes حيث مناسب (`deleted_at`)
- timestamps على كل حدث

---

## 3. الجداول المطلوبة

### 3.1 Subscription & Billing Layer
1. **plans** - الخطط المتاحة (Basic, Mid, Premium)
2. **stores** - بيانات المتاجر
3. **subscriptions** - اشتراكات المتاجر الفعلية

### 3.2 Content Layer
4. **faq_entries** - إدخالات FAQ للمتاجر
5. **product_snapshots** - نسخ المنتجات من سلة

### 3.3 Conversation Layer
6. **visitor_sessions** - جلسات الزوار (session/cookie)
7. **tickets** - محادثات/threads
8. **messages** - الرسائل الفردية
9. **escalations** - التصعيدات

### 3.4 Tracking & Compliance Layer
10. **usage_events** - أحداث الاستخدام للفوترة
11. **consent_logs** - سجلات موافقات الخصوصية
12. **audit_logs** - سجلات تدقيق عامة

---

## 4. خطة التنفيذ

### المرحلة 4.1: إنشاء Migrations (5 ملفات)
```
supabase/migrations/
├── 20260217_01_plans_and_stores.sql
├── 20260217_02_content_tables.sql
├── 20260217_03_conversation_tables.sql
├── 20260217_04_tracking_tables.sql
└── 20260217_05_rls_policies.sql
```

**المحتوى:**
- Migration 1: plans, stores, subscriptions + triggers
- Migration 2: faq_entries, product_snapshots + indexes
- Migration 3: visitor_sessions, tickets, messages, escalations + indexes
- Migration 4: usage_events, consent_logs, audit_logs + indexes
- Migration 5: جميع RLS policies

### المرحلة 4.2: RLS Policies
كل جدول يحتاج:
- **SELECT**: `WHERE store_id = auth.uid()` (أو equivalent)
- **INSERT**: نفس الشرط
- **UPDATE**: نفس الشرط
- **DELETE**: نفس الشرط أو منع حسب الجدول

**استثناءات:**
- `plans`: read-only للجميع
- `audit_logs`: append-only, no updates/deletes

### المرحلة 4.3: TypeScript Types
ملف واحد شامل:
```typescript
// packages/shared/src/types/database.ts
export interface Database {
  public: {
    Tables: {
      plans: { ... }
      stores: { ... }
      subscriptions: { ... }
      // ... باقي الجداول
    }
  }
}
```

---

## 5. تفاصيل الجداول

### 5.1 plans
```sql
- plan_id: UUID PRIMARY KEY
- plan_name: VARCHAR (basic, mid, premium)
- bot_reply_limit: INTEGER
- sync_frequency_hours: INTEGER
- price_monthly_sar: DECIMAL
- is_active: BOOLEAN
```

**ملاحظات:**
- جدول reference فقط
- يمكن تعبئته بـ seed data
- RLS: read-only للجميع

---

### 5.2 stores
```sql
- store_id: UUID PRIMARY KEY
- salla_merchant_id: VARCHAR UNIQUE
- salla_access_token: TEXT (encrypted)
- salla_refresh_token: TEXT (encrypted)
- salla_token_expires_at: TIMESTAMPTZ
- store_name_ar: VARCHAR
- store_url: VARCHAR
- contact_email: VARCHAR
- contact_phone: VARCHAR
- bot_enabled: BOOLEAN DEFAULT true
- widget_settings: JSONB
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ
- deleted_at: TIMESTAMPTZ
```

**ملاحظات:**
- tokens مشفرة على مستوى التطبيق
- widget_settings: {color, branding_enabled, etc}
- soft delete

---

### 5.3 subscriptions
```sql
- subscription_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- plan_id: UUID FK plans
- status: VARCHAR (active, cancelled, suspended)
- started_at: TIMESTAMPTZ
- ends_at: TIMESTAMPTZ
- current_cycle_start: TIMESTAMPTZ
- current_cycle_end: TIMESTAMPTZ
- current_cycle_usage: INTEGER DEFAULT 0
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ
```

**ملاحظات:**
- متجر واحد = اشتراك واحد نشط
- usage يُحسب من usage_events

---

### 5.4 faq_entries
```sql
- faq_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- category: VARCHAR (shipping, payment, returns, products, general)
- question_ar: TEXT
- answer_ar: TEXT
- is_active: BOOLEAN DEFAULT true
- usage_count: INTEGER DEFAULT 0
- last_used_at: TIMESTAMPTZ
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ
```

**Indexes:**
- `(store_id, is_active)`
- `(store_id, category)` WHERE is_active
- GIN on `to_tsvector('arabic', question_ar || answer_ar)`

---

### 5.5 product_snapshots
```sql
- snapshot_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- salla_product_id: VARCHAR
- name_ar: VARCHAR
- description_ar: TEXT
- price: DECIMAL
- currency: VARCHAR DEFAULT 'SAR'
- available: BOOLEAN
- stock_quantity: INTEGER
- image_url: TEXT
- category_ar: VARCHAR
- sku: VARCHAR
- snapshot_timestamp: TIMESTAMPTZ
- is_latest: BOOLEAN DEFAULT true
- created_at: TIMESTAMPTZ
```

**ملاحظات:**
- UNIQUE(store_id, salla_product_id, snapshot_timestamp)
- عند sync جديد: is_latest = false للقديم، true للجديد

**Indexes:**
- `(store_id, is_latest)` WHERE is_latest
- `(store_id, salla_product_id)` WHERE is_latest
- GIN on `to_tsvector('arabic', name_ar || description_ar)` WHERE is_latest

---

### 5.6 visitor_sessions
```sql
- visitor_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- session_cookie: VARCHAR UNIQUE
- consent_given: BOOLEAN DEFAULT false
- consent_given_at: TIMESTAMPTZ
- email_encrypted: TEXT
- phone_encrypted: TEXT
- name_encrypted: TEXT
- first_visit_at: TIMESTAMPTZ
- last_visit_at: TIMESTAMPTZ
- visit_count: INTEGER DEFAULT 1
- user_agent: TEXT
- ip_address: INET
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ
```

**ملاحظات:**
- UNIQUE(store_id, session_cookie)
- بيانات شخصية مشفرة على مستوى التطبيق

---

### 5.7 tickets
```sql
- ticket_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- visitor_id: UUID FK visitor_sessions
- status: VARCHAR (open, resolved, escalated, closed)
- resolution_type: VARCHAR
- initial_question: TEXT
- category: VARCHAR
- clarification_count: INTEGER DEFAULT 0
- opened_at: TIMESTAMPTZ
- resolved_at: TIMESTAMPTZ
- escalated_at: TIMESTAMPTZ
- closed_at: TIMESTAMPTZ
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ
```

**Indexes:**
- `(store_id, status, updated_at)`
- `(visitor_id, created_at DESC)`
- `(store_id, escalated_at)` WHERE status = 'escalated'

---

### 5.8 messages
```sql
- message_id: UUID PRIMARY KEY
- ticket_id: UUID FK tickets
- store_id: UUID FK stores
- sender_type: VARCHAR (visitor, bot, merchant)
- content_ar: TEXT
- is_clarification_request: BOOLEAN DEFAULT false
- includes_dynamic_data: BOOLEAN DEFAULT false
- model_used: VARCHAR
- tokens_used: INTEGER
- response_time_ms: INTEGER
- created_at: TIMESTAMPTZ
```

**Indexes:**
- `(ticket_id, created_at ASC)`
- `(store_id, created_at DESC)`
- `(store_id, sender_type, created_at DESC)` WHERE sender_type = 'bot'

---

### 5.9 escalations
```sql
- escalation_id: UUID PRIMARY KEY
- ticket_id: UUID FK tickets UNIQUE
- store_id: UUID FK stores
- visitor_id: UUID FK visitor_sessions
- reason: VARCHAR (failed_clarification, unsupported_request, manual_request, error)
- problem_description: TEXT
- contact_method: VARCHAR (email, phone)
- contact_value_encrypted: TEXT
- order_number: VARCHAR
- status: VARCHAR (pending, in_progress, resolved, closed)
- resolved_by: VARCHAR (merchant, system)
- resolution_notes: TEXT
- escalated_at: TIMESTAMPTZ
- resolved_at: TIMESTAMPTZ
- created_at: TIMESTAMPTZ
- updated_at: TIMESTAMPTZ
```

**Indexes:**
- `(store_id, status, escalated_at DESC)`
- `(store_id, escalated_at DESC)` WHERE status = 'pending'

---

### 5.10 usage_events
```sql
- event_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- subscription_id: UUID FK subscriptions
- ticket_id: UUID FK tickets
- message_id: UUID FK messages
- event_type: VARCHAR (bot_reply, clarification, escalation)
- billing_cycle_start: TIMESTAMPTZ
- billing_cycle_end: TIMESTAMPTZ
- tokens_used: INTEGER
- estimated_cost_usd: DECIMAL
- model_used: VARCHAR
- created_at: TIMESTAMPTZ
```

**ملاحظات:**
- append-only table
- partitioned by created_at (monthly) في المستقبل

**Indexes:**
- `(store_id, billing_cycle_start, billing_cycle_end)`
- `(store_id, event_type, created_at DESC)`
- `(created_at)` WHERE event_type = 'bot_reply'

---

### 5.11 consent_logs
```sql
- consent_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- visitor_id: UUID FK visitor_sessions
- ticket_id: UUID FK tickets
- consent_type: VARCHAR (personal_data_storage, persistent_memory)
- consent_given: BOOLEAN
- consent_method: VARCHAR (chat_checkbox, explicit_message)
- ip_address: INET
- user_agent: TEXT
- created_at: TIMESTAMPTZ
```

**ملاحظات:**
- append-only
- immutable للامتثال القانوني

**Indexes:**
- `(visitor_id, created_at DESC)`
- `(store_id, created_at DESC)`

---

### 5.12 audit_logs
```sql
- audit_id: UUID PRIMARY KEY
- store_id: UUID FK stores
- actor_type: VARCHAR (merchant, staff, system, api)
- actor_id: VARCHAR
- action: VARCHAR (login, access_conversation, toggle_bot, modify_faq, etc)
- resource_type: VARCHAR (store, ticket, escalation, etc)
- resource_id: UUID
- ip_address: INET
- user_agent: TEXT
- metadata: JSONB
- created_at: TIMESTAMPTZ
```

**ملاحظات:**
- append-only
- retention >= 90 days (حسب NFR)

**Indexes:**
- `(store_id, created_at DESC)`
- `(actor_id, created_at DESC)`
- `(resource_type, resource_id, created_at DESC)`

---

## 6. Triggers & Functions

### 6.1 update_updated_at
```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

تُطبق على: stores, subscriptions, faq_entries, visitor_sessions, tickets, escalations

---

### 6.2 update_visitor_last_visit
```sql
-- عند إنشاء ticket جديد، تحديث last_visit_at
CREATE OR REPLACE FUNCTION update_visitor_last_visit()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE visitor_sessions
  SET
    last_visit_at = NOW(),
    visit_count = visit_count + 1
  WHERE visitor_id = NEW.visitor_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

### 6.3 increment_cycle_usage
```sql
-- عند إضافة usage_event، تحديث subscription usage
CREATE OR REPLACE FUNCTION increment_subscription_usage()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.event_type = 'bot_reply' THEN
    UPDATE subscriptions
    SET current_cycle_usage = current_cycle_usage + 1
    WHERE subscription_id = NEW.subscription_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## 7. Views للاستعلامات الشائعة

### 7.1 v_active_subscriptions
```sql
CREATE VIEW v_active_subscriptions AS
SELECT
  s.subscription_id,
  s.store_id,
  st.store_name_ar,
  p.plan_name,
  p.bot_reply_limit,
  s.current_cycle_usage,
  (p.bot_reply_limit - s.current_cycle_usage) AS remaining_usage,
  ROUND((s.current_cycle_usage::DECIMAL / p.bot_reply_limit) * 100, 2) AS usage_percentage
FROM subscriptions s
JOIN stores st ON s.store_id = st.store_id
JOIN plans p ON s.plan_id = p.plan_id
WHERE s.status = 'active'
  AND st.deleted_at IS NULL;
```

---

### 7.2 v_pending_escalations
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
  COUNT(m.message_id) AS message_count
FROM escalations e
JOIN stores s ON e.store_id = s.store_id
JOIN tickets t ON e.ticket_id = t.ticket_id
LEFT JOIN messages m ON t.ticket_id = m.ticket_id
WHERE e.status = 'pending'
GROUP BY e.escalation_id, s.store_name_ar
ORDER BY e.escalated_at DESC;
```

---

## 8. RLS Policies - الملخص

### نمط عام
```sql
-- Enable RLS
ALTER TABLE table_name ENABLE ROW LEVEL SECURITY;

-- SELECT policy
CREATE POLICY "Users can view own store data"
ON table_name FOR SELECT
USING (store_id = auth.uid());

-- INSERT policy
CREATE POLICY "Users can insert own store data"
ON table_name FOR INSERT
WITH CHECK (store_id = auth.uid());

-- UPDATE policy
CREATE POLICY "Users can update own store data"
ON table_name FOR UPDATE
USING (store_id = auth.uid());

-- DELETE policy (حسب الجدول)
CREATE POLICY "Users can delete own store data"
ON table_name FOR DELETE
USING (store_id = auth.uid());
```

**استثناءات:**
- `plans`: read-only للجميع (SELECT فقط)
- `audit_logs`: append-only (INSERT فقط، no UPDATE/DELETE)
- `consent_logs`: append-only (INSERT فقط، no UPDATE/DELETE)

---

## 9. TypeScript Types Structure

```typescript
// packages/shared/src/types/database.ts

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      plans: {
        Row: { ... }
        Insert: { ... }
        Update: { ... }
      }
      stores: { ... }
      subscriptions: { ... }
      // ... باقي الجداول
    }
    Views: {
      v_active_subscriptions: { Row: { ... } }
      v_pending_escalations: { Row: { ... } }
    }
    Functions: {
      // في المستقبل
    }
  }
}
```

---

## 10. معايير النجاح

- ✅ جميع الجداول (12) موجودة ومختبرة
- ✅ RLS policies على كل جدول
- ✅ Indexes محسنة للأداء
- ✅ Triggers تعمل بشكل صحيح
- ✅ TypeScript types مولدة ومحدثة
- ✅ Migration files قابلة للتنفيذ والرجوع (up/down)
- ✅ Views تعمل وتعطي نتائج صحيحة
- ✅ اختبار عزل البيانات يمر بنجاح

---

## 11. الخطوات التالية (بعد المرحلة 2)

- المرحلة 3: API Foundation (Cloudflare Workers + Hono)
- المرحلة 4: Salla OAuth Integration
- المرحلة 5: AI Orchestrator Implementation
- المرحلة 6: Chat Widget UI
- المرحلة 7: Merchant Dashboard

---

**تاريخ الإنشاء:** 17 فبراير 2026
**آخر تحديث:** 17 فبراير 2026
**الحالة:** قيد التنفيذ
