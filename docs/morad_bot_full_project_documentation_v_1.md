# MoradBot — Full Project Documentation (Version 1.0)

---

## 1. Project Overview

### Project Name
MoradBot

### Project Type
B2B SaaS Platform for AI-Powered Customer Support

### Target Market
Saudi e-commerce merchants using Salla

### Target Merchant Size
10–500 orders per month

### Initial Launch Mode
Closed Beta (selected merchants only)

---

## 2. Core Objective (MVP)

MoradBot Version 1 focuses on one problem only:

> Automating FAQ responses inside Salla stores using an AI assistant.

### Primary Goals
- Save 5 hours per day of manual replies
- Provide 24/7 automated responses
- Deliver instant answers based on verified store data

No sales logic. No upselling. No proactive messaging.
Only FAQ support automation.

---

## 3. MVP Scope

### 3.1 Included in MVP

1. FAQ Automation (Top 5 Questions)
   - Product availability (via Salla Products API)
   - Product price (via Salla Products API)
   - Shipping duration (merchant-provided FAQ)
   - Payment methods (merchant-provided FAQ)
   - Return policy (merchant-provided FAQ)

2. Salla Integration
   - OAuth connection
   - Products Read-only access
   - No Orders access
   - No financial permissions

3. Chat Widget
   - Embedded inside Salla store
   - Arabic only
   - Text only (no files, images, voice)
   - First message indicates it is an AI assistant

4. Merchant Dashboard (Basic)
   - View conversations (read-only)
   - Basic stats (bot replies count)
   - On/Off toggle
   - Escalation view

5. Escalation System
   - If bot fails to understand after 3 clarification attempts
   - Ask for:
     - Problem description
     - Order number (if any)
     - Contact method (email/phone)
   - Escalation appears in dashboard
   - Merchant contacts customer externally
   - Merchant closes ticket manually

6. Memory Model
   - Session memory (within ticket)
   - Persistent visitor memory (via cookie)
   - Personal data stored encrypted

7. Consent & Compliance
   - Explicit consent before storing personal data
   - PDPL-aware data handling

---

### 3.2 Excluded from MVP

- Order tracking
- Advanced analytics
- Brand voice customization
- English language
- WhatsApp integration
- File uploads
- Proactive bot messages
- Multi-language support

---

### 3.3 Strictly Forbidden

- Cancel orders
- Issue refunds
- Modify prices
- Access payment data
- Modify product data
- Change store policies

MoradBot is read-only with respect to store data.

---

## 4. Business Model

### Subscription Structure
- Fixed monthly plans
- Based on number of bot replies per month

Example:
- 500 bot replies per month

### Usage Logic
- Only bot replies count toward usage
- Customer messages do not count
- Internal cost calculated via tokens (hidden from merchant)

### Threshold Notifications
- 80% usage → notify merchant
- 100% usage → notify merchant
- No automatic overcharge
- No automatic upgrade
- Add-ons planned for future versions

### Cancellation Policy
- Service continues until end of billing cycle
- No immediate termination

---

## 5. Data Policy

### During Subscription
- Conversations stored
- Personal data encrypted
- Persistent visitor tracking via cookie

### After Cancellation
- 30–90 day retention period
- Then:
  - Delete all personal data
  - Retain anonymized data for AI improvement

---

## 6. Multi-Tenant Isolation

- One merchant login
- Each store has separate subscription
- Zero tolerance for data leakage
- Any cross-store data exposure = launch blocker

Employees may access conversations for support purposes
But:
- Sensitive fields encrypted
- No plaintext access to personal data

---

## 7. Infrastructure

### Hosting
- Cloudflare

### Database & Auth
- Supabase

### Language
- TypeScript

### Architecture Philosophy
- Single agent in MVP
- Designed to evolve into Multi-Agent system later
- Minimal complexity for first 6 months

---

## 8. AI Model

### Provider
OpenRouter

### Primary Model
Gemini 2.0 Flash

### Fallback Models
GPT-4 Mini
Claude 3.5 Sonnet

---

## 9. Branding

Basic plan:
- "Powered by MoradBot" (clickable link)

Higher plans:
- Fully white-label

---

## 10. Product Philosophy

MVP = Focused, Sharp, Single-purpose tool.

Future = Multi-agent AI platform.

---

End of Document

