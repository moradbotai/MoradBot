# MoradBot — System Requirements Document (SRD)

Scope: Derived from BRD + PRD + NFR. Requirements only (no code).

---

## 1) System Overview

MoradBot is a B2B SaaS application delivered as a Salla App that provides an Arabic, text-only, in-store chat assistant focused on automating repetitive FAQ responses.

### Core system outcomes
- Provide instant responses for repetitive FAQ using merchant-provided FAQ answers + periodically synced Salla product data.
- Escalate unresolved questions after up to 3 clarification attempts.
- Track bot reply usage per store and notify merchants at 80% and 100% consumption.
- Ensure strict data isolation between stores (zero leakage tolerance).

### Operating constraints (from product decisions)
- MVP is narrow: FAQ responder only.
- Salla integration is read-only and limited to Products scope.
- No WhatsApp, no order tracking, no refunds/cancellation.
- Two environments only: Development and Production.
- Manual deployments.

---

## 2) Main Components

### 2.1 Salla App (Merchant Installation & Authorization)
**Purpose:** Provide merchant installation and OAuth authorization flow.

**Responsibilities:**
- Allow merchant to install MoradBot from Salla App Store.
- Establish OAuth authorization to access Salla Products (read-only).
- Persist store-level authorization and store identity.

**Key system requirements:**
- Installation must result in automatic widget presence in the store without manual theme changes.
- Support a single merchant identity connecting multiple stores; billing per store.

---

### 2.2 Chat Widget (Customer-Facing UI)
**Purpose:** Customer interface inside the Salla store.

**Responsibilities:**
- Render on all store pages except Checkout.
- Display AI disclosure only in the first message.
- Provide Arabic-only text chat input/output.
- Capture explicit consent before storing personal data / enabling persistent memory.

**Key system requirements:**
- Must support ticket/thread separation by topic.
- Must support session memory per ticket and persistent memory per visitor (cookie/session ID), gated by consent.
- Must enforce rate limits (visitor-level) and fail gracefully with a clear message.

---

### 2.3 Conversation Service (Ticketing + State Management)
**Purpose:** Owns conversation lifecycle, ticket/thread model, and message persistence.

**Responsibilities:**
- Create and manage tickets (threads) per visitor.
- Maintain per-ticket session context.
- Link visitor identity across sessions via cookie/session ID (with consent).
- Mark tickets resolved automatically when bot answers successfully.
- Require manual closure when escalated.

**Key system requirements:**
- Must store conversations as a core strategic asset.
- Must ensure strict store-level isolation for all records.
- Must provide analytics inputs (counts, resolution rates, escalation rates).

---

### 2.4 FAQ & Policy Content Service (Merchant-Provided Answers)
**Purpose:** Store and serve the merchant’s FAQ answers.

**Responsibilities:**
- Capture initial FAQ setup during onboarding.
- Serve FAQs as authoritative source for shipping/payment/returns/contact.
- Allow optional later edits via dashboard (future/optional depending on product scope decisions).

**Key system requirements:**
- Must ensure each store has its own FAQ set.
- Must ensure answers are treated as “verified” content.

---

### 2.5 Salla Product Sync Service
**Purpose:** Periodically synchronize Salla Products (read-only) into local store data.

**Responsibilities:**
- Perform scheduled sync per store according to plan tier:
  - Basic: every 24 hours
  - Higher: every 6 hours
  - Highest: every 1 hour
- Store product snapshots locally to power bot responses.
- Track last sync timestamp.

**Key system requirements:**
- Must complete sync for up to 1,000 products per store within 60 seconds.
- Must retry failed sync within 5 minutes.
- Must support plan-based scheduling enforcement.

---

### 2.6 AI Response Orchestrator (Single-Agent MVP)
**Purpose:** Generate bot replies based on verified sources.

**Responsibilities:**
- Accept customer message + ticket context + store FAQ + product snapshot.
- Return Arabic-only responses.
- Ask for clarification when not understood (up to 3 attempts).
- Escalate after 3 unsuccessful clarifications.
- Add “حسب آخر تحديث…” disclaimer only for dynamic data (price/stock) and not on every reply.

**Key system requirements:**
- Must never perform restricted actions (refunds, cancellations, product edits).
- Must never request contact info unless escalation flow is triggered.
- Must minimize hallucination risk by grounding on store data and FAQ.

---

### 2.7 Escalation Workflow
**Purpose:** Collect necessary details and hand off to merchant.

**Responsibilities:**
- Trigger escalation after 3 clarification failures.
- Collect:
  - Problem description
  - Order number (optional)
  - Contact method (email/phone)
- Surface escalation item in dashboard.
- Require merchant/manual staff closure.

**Key system requirements:**
- Must store contact details encrypted at rest.
- Must log the consent event and escalation event.

---

### 2.8 Usage & Billing Metering
**Purpose:** Track plan usage and notify merchants.

**Responsibilities:**
- Track usage as bot replies count per store per billing cycle.
- Notify merchant at 80% and 100% usage.
- At 100%: customer-facing chat shows a message instructing to contact the store via merchant-defined contact method.

**Key system requirements:**
- Must support per-store subscription and plan tier.
- Must support non-automatic upgrade/overage (merchant decides).
- Must be auditable and tamper-resistant.

---

### 2.9 Merchant Dashboard (Admin UI)
**Purpose:** Minimal operational view for merchant.

**Responsibilities:**
- View conversations/tickets (read-only for MVP).
- View escalation items.
- Manual close escalated tickets.
- Toggle bot on/off.
- View basic stats: bot replies per day, usage vs plan.

**Key system requirements:**
- Dashboard initial load P95 ≤ 2.5s.
- Must enforce store-level access control.

---

### 2.10 Internal Operations & Support Access
**Purpose:** Enable MoradBot staff to assist with support and improvement.

**Responsibilities:**
- Allow staff access to conversations for support.
- Prevent staff from viewing sensitive customer fields in plaintext.
- Log all staff access.

**Key system requirements:**
- Sensitive fields (email/phone) encrypted; no plaintext display.
- Staff access events logged with retention ≥ 90 days.

---

### 2.11 Monitoring & Alerting
**Purpose:** Maintain reliability, cost control, and incident detection.

**Responsibilities:**
- Monitor latency, error rate, AI provider failures, usage spikes.
- Provide dashboard monitoring.
- Send alerts via Email and Telegram.

**Key system requirements (from NFR):**
- Alerts when:
  - Error rate > 5% over 5 minutes
  - Average latency > 4 seconds over 5 minutes
  - Token usage anomaly > 200% baseline

---

## 3) Data Flows (End-to-End)

### 3.1 Merchant Onboarding
1) Merchant installs MoradBot in Salla.
2) OAuth authorization established (Products read-only).
3) Merchant completes FAQ setup.
4) Store plan is selected/activated and billing is created externally.
5) Bot is enabled; widget becomes active.

### 3.2 Customer Chat (Normal Path)
1) Customer opens store page (non-checkout) and sees chat widget.
2) Customer sends message.
3) Conversation Service creates/assigns ticket.
4) AI Orchestrator generates grounded reply using FAQ + product snapshot.
5) Bot returns Arabic response.
6) Ticket auto-resolves when answered successfully.

### 3.3 Clarification & Escalation Path
1) If bot cannot understand, it requests clarification.
2) Repeat up to 3 attempts.
3) On third failure, escalation triggers.
4) Chat collects issue + optional order + contact.
5) Escalation item appears in dashboard.
6) Merchant contacts customer externally.
7) Merchant closes ticket manually.

### 3.4 Product Sync Flow
1) Scheduler triggers sync per store based on plan.
2) Sync service fetches products and updates local snapshot.
3) Store last_sync timestamp updated.
4) Bot uses latest snapshot and adds “حسب آخر تحديث…” when answering dynamic fields.

### 3.5 Usage Limit Flow
1) Usage meter increments per bot reply.
2) At 80% threshold: notify merchant.
3) At 100% threshold: notify merchant + customer widget shows alternative contact message.

---

## 4) Technical Requirements per Component (System-Level)

### 4.1 Hosting/Runtime
- Must run on Cloudflare-managed hosting.
- Must support cron/scheduled triggers for product sync.

### 4.2 Data Store
- Must use Supabase for primary database and authentication.
- Must support strict tenant isolation.
- Must store conversation data as primary asset.

### 4.3 Performance
- Chat reply latency targets:
  - P50 ≤ 1.5s
  - P95 ≤ 3.0s
  - Timeout 8s

### 4.4 Security
- TLS 1.2+ for all connections.
- Sensitive fields encrypted at rest.
- Secrets stored only in Cloudflare Secrets.
- Rate limiting:
  - Visitor: max 20 messages/min
  - Store: max 3,000 bot replies/hour
- Audit logging:
  - Auth events
  - Staff access events
  - Retention ≥ 90 days

### 4.5 Reliability & Recovery
- Internal availability target: ≥ 99% monthly.
- RPO ≤ 24h (daily backups via Supabase).
- RTO ≤ 4h.
- Incident policy: immediate shutdown for critical incidents.

### 4.6 Privacy & Consent
- Explicit consent required before storing personal data and enabling persistent memory.
- Consent events logged with timestamps.
- Post-cancellation retention:
  - Personal data deleted within 30–90 days
  - Anonymized data may be retained

---

## 5) Constraints & Assumptions

### Constraints
- Single-person development capacity using AI coding assistants.
- MVP must remain narrowly focused on FAQ automation.
- Two environments only; no staging.
- Manual deployments only.

### Assumptions
- Salla App Store installation can inject/enable chat widget automatically (no merchant theme edits).
- Salla OAuth with Products read-only is sufficient for MVP value.
- External billing provider handles payments per store.
- Merchant willingness to provide initial FAQ answers during onboarding.

---

## 6) Future Integration Requirements

These are not MVP requirements but must be supported by the architecture direction.

### 6.1 Salla Expansions
- Inventory/stock APIs if separated from Products.
- Orders API for tracking (Phase 2).
- Webhooks for real-time updates (optional future).

### 6.2 Channels
- WhatsApp Business integration (Phase 2+).
- Additional channels (email, social) later.

### 6.3 Product Capabilities
- Add-on reply packages (billing feature).
- Advanced analytics & reporting.
- Brand voice configuration.
- English language support.

### 6.4 Multi-Agent Platform
- Agent routing/orchestration layer.
- Dedicated Sales Agent.
- Dedicated Analytics/Insights Agent.
- Governance policies per agent to prevent prohibited actions.

---

End of SRD

