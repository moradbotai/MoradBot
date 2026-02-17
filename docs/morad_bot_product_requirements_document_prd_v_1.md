# MoradBot — Product Requirements Document (PRD)

Based on BRD v1
Scope: Product definition only. No business repetition. No technical implementation details.

---

## 1. Product Description (From User Perspective)

MoradBot is an in-store AI chat assistant for Salla merchants.

From the merchant’s perspective:
- I install the app.
- I answer 5 basic setup questions.
- The assistant starts replying automatically to repetitive customer questions.
- If it cannot answer, it escalates to me.

From the customer’s perspective:
- I open a store.
- I see a chat.
- I ask a question.
- I get an immediate answer.

MoradBot is not a full support system.
It is not a sales engine.
It is not CRM software.

It is a focused FAQ automation tool.

---

## 2. Primary Users (Personas)

### Persona 1 — The Solo Merchant
- 30–150 orders/month
- Handles marketing, operations, and customer support
- Overwhelmed by repetitive questions
- Price sensitive but values time savings

### Persona 2 — Micro-Team Operator
- 2–3 team members
- No dedicated support agent
- Growth-focused
- Wants automation without complexity

### Secondary User — End Customer (Indirect User)
- Visits store
- Asks product or policy questions
- Expects instant answers

The primary product focus is on Persona 1.

---

## 3. Must-Have Features (MVP Critical)

1. Salla App Installation
   - One-click install
   - OAuth connection

2. Basic FAQ Setup
   - Merchant defines answers for:
     - Shipping time
     - Payment methods
     - Return policy
     - Contact info
     - Custom FAQ field

3. Product Data Sync
   - Periodic sync from Salla
   - Read-only

4. Chat Widget
   - Visible on all pages except checkout
   - Arabic only
   - Text only

5. Automatic FAQ Response
   - Answers based on:
     - Stored FAQ
     - Synced product data

6. Escalation Logic
   - 3 failed clarification attempts
   - Collect issue description + order number + contact
   - Show in dashboard

7. Basic Merchant Dashboard
   - View conversations
   - View bot reply count
   - On/Off toggle

8. Usage Tracking
   - Count bot replies
   - 80% and 100% usage notification

---

## 4. Nice to Have (Deferred)

- Order tracking integration
- WhatsApp integration
- English language support
- Advanced analytics
- Brand voice customization
- File uploads
- Proactive messaging
- Add-on reply packages

None of these belong in MVP.

---

## 5. Out of Scope (Explicitly Excluded)

- Refund processing
- Order cancellation
- Price modification
- Payment data access
- CRM functionality
- Marketing automation
- Multi-agent orchestration
- Deep personalization

If a feature risks expanding scope beyond FAQ automation, it is excluded.

---

## 6. Core User Stories

1. As a merchant, I want to install the app and activate it in under 10 minutes.

2. As a merchant, I want repetitive customer questions answered automatically without my intervention.

3. As a merchant, I want to be notified when the bot cannot handle a question.

4. As a customer, I want immediate answers about price, availability, and policies.

5. As a merchant, I want to see how many bot replies were used this month.

6. As a merchant, I want to disable the bot instantly if needed.

7. As a merchant, I want reassurance that the bot does not perform financial actions.

---

## 7. Clear MVP Definition

MVP =

A Salla-installed AI chat assistant that:
- Automatically answers repetitive FAQ questions in Arabic
- Uses synced product data
- Escalates unresolved questions after 3 attempts
- Tracks bot reply usage
- Provides a minimal merchant dashboard

Nothing more.

No sales engine.
No automation flows.
No integrations beyond Salla products.

Success of MVP is measured by:
- Merchants keeping it active
- High rate of FAQ auto-resolution
- Low manual intervention rate

---

End of PRD

