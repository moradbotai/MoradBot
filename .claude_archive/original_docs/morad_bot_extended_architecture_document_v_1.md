# MoradBot — Extended Architecture & Operational Decisions (Version 1.0)

This document contains all additional architectural, operational, and security decisions taken after the core product definition.

---

## 1. Chat Widget Deployment Inside Salla

### Decision
- Merchant installs MoradBot from Salla App Store.
- Chat widget appears automatically.
- No manual script injection.
- No theme modification required.

### Display Rules
- Chat appears on all store pages.
- Chat does NOT appear on Checkout page.

---

## 2. Product Data Synchronization Strategy

### Decision: Periodic Sync (Not On-Demand)

Data from Salla (Products API - Read Only) will be synchronized and stored locally.

### Sync Frequency by Plan
- Basic Plan: Every 24 hours
- Mid Plan: Every 6 hours
- Higher Plan: Every 1 hour

### Response Behavior for Dynamic Data
- Bot answers based on last synchronized data.
- If question relates to dynamic data (price / stock):
  - Bot adds disclaimer: "حسب آخر تحديث..."
- Disclaimer appears only for dynamic fields.
- Not repeated continuously inside same ticket.

---

## 3. Rate Limiting & Abuse Protection

Protection is applied on two levels:

### Visitor-Level Limits
- Prevent spam from single user.
- Limit messages per minute/session.

### Store-Level Limits
- Prevent distributed abuse.
- Protect total system load.

Both limits are active simultaneously.

---

## 4. Monitoring & Alerting

### Monitoring Dashboard Includes
- Bot reply usage per store
- Token cost tracking
- Error rate monitoring
- Response latency
- Model failure rate

### Alerts
- Email alerts
- Telegram alerts

Alerts triggered for:
- Sudden token spikes
- High error rate
- Latency issues
- OpenRouter failure

---

## 5. Backup Strategy

### Current Phase (MVP)
- Rely only on Supabase automatic backups.

### Future Phase
- Add external backup on VPS.

---

## 6. Incident Response Policy

If any critical issue occurs:
- Data leakage
- Severe logic error
- Resource abuse
- Security breach

### Action
- Immediate service shutdown.
- Investigate and resolve before resuming.
- No graceful degradation for critical incidents.

Security and trust are prioritized over availability.

---

## 7. SLA Policy

### Decision (First 6 Months)
- No public SLA declared.
- No formal uptime guarantee during Beta phase.

Future SLA may be introduced after stabilization.

---

## 8. Development Environments

### Structure
- Development environment
- Production environment

No staging environment.

### Deployment Strategy
- Manual deployment only.
- No automatic production deploy.
- Developer triggers release manually.

---

## 9. API Key & Secret Management

### Decision
All sensitive credentials are stored ONLY in Cloudflare Secrets.

Includes:
- OpenRouter API key
- Supabase service role key
- Internal secret keys

Secrets are NOT stored in database.
Secrets are NOT stored in code.

---

## 10. Product Positioning (Phase 1)

### Philosophy
MVP is extremely focused.
Single-purpose tool:
- FAQ intelligent responder only.

No customization complexity.
No multi-agent orchestration in MVP.

Architecture is designed to evolve later.

---

## 11. Multi-Agent Future Direction

Current version:
- Single FAQ Agent.

Future vision:
- Expandable architecture.
- Sales Agent.
- Support Agent.
- Analytics Agent.

Without requiring full system rewrite.

---

## 12. Conversation Data as Strategic Asset

Conversations are considered:
- Core long-term asset.
- Valuable for analytics.
- Valuable for model improvement.

With strict:
- Encryption of personal data.
- Isolation between stores.
- Personal data deletion after cancellation.

---

End of Document

