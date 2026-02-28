# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**مراد بوت** — B2B SaaS, AI-powered FAQ automation for Saudi Salla merchants.
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**Audience:** Merchants with 30–300 orders/month.
**Language:** Arabic only in the Chat Widget (Rule 5).
**Date:** February 2026.
**GitHub:** `https://github.com/moradbotai/Morad_Bot` (private)

---

## Tech Stack

| Component | Technology | Version |
| ----------- | ------------ | ------- |
| API Runtime | Cloudflare Workers (TypeScript) + Hono | Hono 4.11.9 / Wrangler 3.114.17 |
| Database/Auth | Supabase (PostgreSQL + RLS) | supabase-js 2.48.x |
| AI Provider | OpenRouter → Gemini 2.0 Flash (fallbacks: GPT-4 Mini, Claude 3.5 Sonnet) | — |
| Chat Widget | Preact + Vite → bundled JS (<50KB gzipped) | Preact 10.28.3 / Vite 6.4.1 |
| Dashboard | Next.js 15 (App Router) + Cloudflare Pages | Next.js 15.5.12 / React 19.2.4 |
| Package Manager | pnpm + Turborepo | pnpm 8.15.0 / Turborepo 2.8.9 |
| Linting/Formatting | Biome (tabs, double quotes, 100 char line width) | Biome 1.9.4 |
| Validation | Zod (shared across all packages) | Zod 3.24.x |
| Language | TypeScript | 5.9.3 |
| Testing | Vitest + Playwright (Phase 8 — not yet installed) | — |
| Rate Limiting | Cloudflare KV | — |
| Encryption | AES-256-GCM via Web Crypto API (built-in Workers) | — |
| Email | Resend (Phase 5) | — |

> Full compatibility review: `docs/claude/technology_compatibility_review.md`

---

## Commands

### Root (Turborepo — runs all apps/packages)

```bash
pnpm build          # Build all
pnpm dev            # Dev all (persistent)
pnpm lint           # Lint all
pnpm format         # Format all files (writes)
pnpm format:check   # Format check (CI)
pnpm type-check     # TypeScript check all
pnpm test           # Test all
pnpm clean          # Clean dist + node_modules
```

### Per-app (use `--filter`)

```bash
pnpm --filter @moradbot/api dev          # Wrangler dev server
pnpm --filter @moradbot/widget dev       # Vite dev server
pnpm --filter @moradbot/dashboard dev    # Next.js dev server

pnpm --filter @moradbot/api lint
pnpm --filter @moradbot/api type-check
```

### API Deployment (manual only — Rule 6)

```bash
cd apps/api
pnpm deploy                              # wrangler deploy (production)
```

### Wrangler Secrets (must be set before first deploy)

```bash
# Phase 1–3 (required now)
wrangler secret put SUPABASE_URL
wrangler secret put SUPABASE_ANON_KEY
wrangler secret put SUPABASE_SERVICE_ROLE_KEY
wrangler secret put SALLA_CLIENT_ID
wrangler secret put SALLA_CLIENT_SECRET
wrangler secret put SALLA_REDIRECT_URI

# Phase 4 (before Salla OAuth deploy)
wrangler secret put ENCRYPTION_KEY          # openssl rand -hex 32

# Phase 5 (before AI + email features)
wrangler secret put OPENROUTER_API_KEY
wrangler secret put RESEND_API_KEY
wrangler secret put RESEND_FROM_EMAIL
```

### KV Namespace Setup (Phase 4 — Rate Limiting)

```bash
# Create KV namespaces
wrangler kv namespace create RATE_LIMIT_KV
wrangler kv namespace create RATE_LIMIT_KV --preview

# Then replace placeholder IDs in apps/api/wrangler.toml
```

### Supabase Migrations

```bash
supabase db push                         # Apply migrations to remote
supabase db reset                        # Reset local DB and re-apply
```

---

## Architecture

### Monorepo Layout

```
moradbot/
├── apps/
│   ├── api/          # Cloudflare Worker — Hono backend
│   ├── widget/       # Preact chat widget → bundled JS
│   └── dashboard/    # Next.js 15 merchant admin panel (placeholder)
├── packages/
│   ├── shared/       # Database TypeScript types (Database interface)
│   ├── ai-orchestrator/  # OpenRouter integration (Phase 5 — scaffold)
│   └── salla-client/    # Salla API client + OAuth (Phase 4 — scaffold)
├── supabase/
│   └── migrations/   # 5 migration files (schema + RLS)
├── docs_v2/          # الوثائق الرسمية المعتمدة للمشروع (المرجع الأول دائماً)
├── .claude_archive/   # Archived content (NOT deleted — move back when needed)
│   ├── original_docs/ # Original v1 docs (protected, read-only)
│   ├── future/        # agents/, commands/, skills/, mcp_servers.json
│   └── not_useful/    # agents/, commands/, skills/, mcp_servers.json
└── docs/
    ├── claude/        # All Claude-generated docs go HERE
    │   ├── plans/     # Phase plans: phase-XX_name.md
    │   ├── docs-audit/ # Documentation audit reports (Feb 2026)
    │   └── tools_report_v2.md  # Full ecosystem audit (Feb 2026)
    └── (empty)        # Original docs archived to .claude_archive/original_docs/
```

### API Worker (`apps/api`)

**Entry:** `src/index.ts` → `createApp()` in `src/app.ts`

**Middleware chain (applied globally):**
`cors` → `errorHandler` → `auditLog` → route handlers

**Route structure:**
```
GET  /health
GET  /auth/salla/start       # Start OAuth, redirect to Salla
GET  /auth/salla/callback    # Exchange code for tokens
POST /auth/salla/refresh     # Refresh access token
POST /api/chat               # Handle visitor message (rate-limited)
GET  /api/faq                # List FAQ entries
POST /api/faq                # Create FAQ entry
PUT  /api/faq/:id            # Update FAQ entry
DEL  /api/faq/:id            # Soft delete FAQ entry
GET  /api/stats              # Dashboard analytics
GET  /api/stats/usage        # Usage metrics
GET  /api/tickets            # List conversations
GET  /api/tickets/:id        # Get conversation details
GET  /api/escalations        # List escalations
PATCH /api/escalations/:id   # Update escalation status
```

**Authentication flow:**
1. Request carries `Authorization: Bearer <jwt>`
2. `requireAuth` middleware calls `supabase.auth.getUser(token)`
3. `user.id` is used as `store_id` (Supabase user = merchant store)
4. `storeId` and `userId` are set in Hono context via `c.set()`

**OAuth Security Principles (Phase 4):**
- Salla `access_token` and `refresh_token` are stored encrypted (AES-256-GCM) — never as plaintext in the database.
- Tokens are never written to logs, error messages, or audit records.
- `SALLA_REDIRECT_URI` is validated strictly: the callback rejects any `redirect_uri` that does not exactly match the registered value.
- OAuth `state` parameter is required and verified on every callback to prevent CSRF.
- Refresh token rotation: on each refresh, the old token is invalidated immediately. Reusing a consumed refresh token triggers revocation of all tokens for that store and requires re-authorization.

**Supabase dual-client pattern (`src/lib/supabase.ts`):**
- `createSupabaseClient(env, storeId)` — uses anon key + sends `x-store-id` header; RLS enforces `auth.uid() = store_id`. **Default for all route handlers.**
- `createSupabaseAdmin(env)` — uses service role key; bypasses RLS entirely.

**`createSupabaseAdmin()` usage policy:**
- Permitted only in: OAuth callback (token storage), cron jobs (product sync), migration scripts.
- Forbidden inside: any route handler that processes a merchant or visitor request.
- Every usage must have an inline comment: `// Admin: <reason> — bypasses RLS intentionally`.
- Never use admin client to query data that belongs to a specific store — use `createSupabaseClient` with that store's ID instead.

**Error hierarchy (`src/lib/errors.ts`):**
`AppError` → `ValidationError` | `AuthenticationError` | `AuthorizationError` | `NotFoundError` | `RateLimitError` | `DatabaseError`

All errors are caught by `errorHandler` middleware and returned as structured JSON.

**Env types (`src/env.ts`):** The `Env` interface defines all Cloudflare secret bindings. Never add secrets to code or `.env`.

**Worker constraints:**
- Every request must carry a valid `store_id`
- Rate limiting applies to every endpoint
- Audit log required for all sensitive operations
- Worker timeout: 30 seconds maximum

### Database (Supabase — 5 migrations)

12 tables: `plans`, `stores`, `subscriptions`, `faq_entries`, `product_snapshots`, `visitor_sessions`, `tickets`, `messages`, `escalations`, `audit_logs`, `usage_tracking`, `bot_configurations`.

**RLS Policy rule:** All tenant queries use `auth.uid() = store_id`. Every query in route handlers must include `.eq("store_id", storeId)`. Omitting this filter is a Rule 3 violation — see enforcement behavior under Rule 3.

### Widget (`apps/widget`)

Preact + Vite, bundled to a single JS file. Loaded on merchant storefronts. Target: <50KB gzipped. Must not render on `/checkout/*` paths.

**7 Widget states:** closed → open → typing → response → escalation → error → limit-reached

**First message:** AI disclosure required. No cookies/storage without explicit visitor consent.

### Dashboard (`apps/dashboard`)

Next.js 15 App Router. Currently: placeholder page at `app/page.tsx`. All API calls must carry a valid JWT. Real-time subscriptions for escalations only.

**4 Dashboard sections:**
1. **الرئيسية (Home)** — Bot usage stats + on/off toggle
2. **المحادثات (Conversations)** — Read-only conversation list
3. **التصعيدات (Escalations)** — View + manual close
4. **الإعدادات (Settings)** — FAQ management + subscription info

**Dashboard rules:**
- Never display data from other stores
- Every API call must carry a valid JWT
- Real-time subscriptions for escalations only

**Cloudflare Pages deployment note:**
Next.js 15 on Cloudflare Pages requires `@cloudflare/next-on-pages` adapter. Add it in Phase 6 before building any real UI:
```bash
pnpm add -D @cloudflare/next-on-pages --filter @moradbot/dashboard
```

---

## Non-Negotiable Rules

### Rule 1 — MVP Scope
Any feature not in `docs_v2/product_requirements.md` (or `.claude_archive/original_docs/morad_bot_product_requirements_document_prd_v_1.md`) is rejected immediately.
**Banned from MVP:** order tracking, WhatsApp, English language, advanced analytics, file uploads, proactive messages.

### Rule 2 — Salla Read-Only
MoradBot only calls `GET /products` on Salla. No write, delete, or update operations on Salla data.

### Rule 3 — Zero-Tolerance Data Isolation

Every query in route handlers must include `.eq("store_id", storeId)`. Queries that bypass RLS are not permitted.

**Enforcement behavior:**
- Missing `store_id` in a query → reject request, return `AuthorizationError (403)`, write to `audit_logs` with `action: "rls_violation"` and `severity: "critical"`.
- Using `createSupabaseAdmin()` inside a route handler without documented justification → treated as Rule 3 violation.
- On confirmed cross-tenant data access: terminate the request, log with `severity: "critical"`, and halt all feature deployments until root cause is identified and patched.

### Rule 4 — Secrets in Cloudflare Secrets Only
No secrets in source code, database, or `.env` files in production. Use `wrangler secret put`.

### Rule 5 — Arabic Only in Widget
Bot always responds in Arabic, even if the visitor writes in English.

### Rule 6 — Manual Deployment Always
No CI/CD auto-deploy to production. Every deploy goes through a manual Pre-Deploy Checklist.

### Rule 7 — Docs Organization
All Claude-generated docs → `docs/claude/` only. Original v1 docs are in `.claude_archive/original_docs/` (protected, read-only). `docs_v2/` هو المجلد الرسمي الأول للوثائق المعتمدة — يُرجع إليه دائماً قبل أي مرجع آخر.

### Rule 8 — Phase Plans
Each development phase gets a plan file in `docs/claude/plans/phase-XX_name.md`.

---

## Decisions Requiring `/ultra-think` First

- Schema changes
- Bot system prompt changes
- Any new feature (even small)
- Security decisions
- Project structure changes
- OpenRouter integration changes
- RLS policy changes
- PDPL-related decisions

---

## Session Workflow

```
Start: /prime  →  /resume
Work:  ultra-think → code → test → security-scan → document → commit
End:   /session-learning-capture  →  /update-docs
```

---

## Workflow Orchestration

### Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### Verification Before Done

- Never mark a task complete without proving it works
- Diff your behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan:** Check in before starting implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections

### Core Principles

- **Simplicity First:** Make every change as simple as possible. Impact minimal code.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Changes should only touch what's necessary. Avoid introducing bugs.

---

## Current Implementation Status

| Phase | Status | What Was Built |
| ------- | -------- | ---------------- |
| Phase 1 | ✅ Complete | Dev environment, Turborepo, Biome, base packages |
| Phase 2 | ✅ Complete | 12-table DB schema, 5 migrations, RLS policies, TypeScript types |
| Phase 3 | ✅ Complete | Hono API: 16 endpoints, middleware stack, error hierarchy, Supabase clients |
| Phase 4 | 🔜 Next | Salla Client Package (OAuth + `GET /products`) |
| Phase 5+ | ⏳ Pending | AI Orchestrator, Widget UI, Dashboard UI |

---

## Claude Ecosystem (Active Tools)

Reorganized Feb 18, 2026. Full audit: `docs/claude/tools_report_v2.md`.

- **Agents (9 active):** backend-architect, typescript-pro, database-optimizer, security-auditor, ai-engineer, error-detective, api-documenter, prompt-engineer, technical-writer — archived: `.claude_archive/future/agents/` (11) + `not_useful/agents/` (12)
- **Commands (16 active):** prime, resume, ultra-think, session-learning-capture, update-docs, code-review, debug-error, write-tests, test-coverage, security-audit, optimize-api-performance, optimize-database-performance, + 4 supabase — archived: `.claude_archive/future/commands/` (16) + `not_useful/commands/` (3)
- **Skills (4 active):** prompt-architect, vercel-react-best-practices, vercel-composition-patterns, skill-creator — archived: `.claude_archive/future/skills/` (3) + `not_useful/skills/` (2)
- **MCP Servers (6 active):** context7, supabase, memory, firecrawl-mcp, fetch, **github** (added Feb 28 2026) — archived: `.claude_archive/future/mcp_servers.json` (6) + `not_useful/mcp_servers.json` (11)

**Skills Note:** `.claude/skills/` entries are symlinks → `.agents/skills/`. Moving a skill dir from `.agents/skills/` requires removing the broken symlink from `.claude/skills/`.

---

## Knowledge Sources (Priority Order)

> **`docs_v2/`** هو المجلد الرسمي الوحيد للوثائق المعتمدة — يُرجع إليه أولاً في أي قرار.
> **Original documents** (محفوظة للأرشيف فقط) → `.claude_archive/original_docs/`

1. `docs_v2/system_requirements.md` — متطلبات النظام (AI provider، rate limiting، multi-store)
2. `docs_v2/product_requirements.md` — متطلبات المنتج (widget states، consent flow، AI disclosure)
3. `docs_v2/architecture.md` — قرارات المعمارية (CORS، Worker timeout، naming)
4. `docs_v2/marketing_strategy.md` — الباقات (97/197/449 SAR)، التجربة المجانية، الاستراتيجية التسويقية (v3.0، فبراير 2026)
5. `docs_v2/infrastructure_and_costs.md` — تكاليف البنية التحتية، التقنيات، نقطة التعادل (فبراير 2026)
6. `docs_v2/implementation_plan.md` — خطة التنفيذ والمراحل
7. `docs_v2/business_requirements.md` — متطلبات الأعمال، KPIs، نموذج الإيرادات
8. `docs_v2/market_requirements.md` — تحليل السوق، ICP، محفزات الشراء
9. `docs/claude/ai-orchestrator-reference/` — نسخة مُكيَّفة من Google ADK Customer Service (original + adaptation README)
10. `docs/claude/salla_api_reference.md` — Salla API: OAuth, Products, Errors, Rate Limits
11. `docs/claude/github_integration.md` — GitHub workflow, branch strategy, commit convention, PR process

---

## Performance Targets

| Metric | Target |
| -------- | -------- |
| Chat Reply P50 | ≤ 1.5s |
| Chat Reply P95 | ≤ 3.0s |
| Chat Timeout | 8s |
| Dashboard Load P95 | ≤ 2.5s |
| Product Sync (1,000) | < 60s |
| Monthly Uptime | ≥ 99% |
| Test Coverage | ≥ 80% |

### Latency Budget (P50 / P95)

How the 1.5s P50 and 3.0s P95 targets for `/api/chat` are allocated:

| Component | P50 Budget | P95 Budget |
| --------- | ---------- | ---------- |
| Worker overhead (routing, middleware, auth) | ≤ 50ms | ≤ 100ms |
| Supabase read (FAQ + product snapshot) | ≤ 150ms | ≤ 300ms |
| AI provider (OpenRouter → Gemini 2.0 Flash) | ≤ 1,200ms | ≤ 2,500ms |
| Supabase write (message + usage log) | ≤ 100ms | ≤ 100ms |
| **Total** | **≤ 1,500ms** | **≤ 3,000ms** |

AI provider dominates the budget. Supabase queries must stay under 150ms P50 or the AI budget is squeezed. Hard timeout for the full request: 8s.

---

## LLM Security Policy

Applies to all code in `packages/ai-orchestrator/` and `apps/api/src/routes/chat.ts`.

- **System prompt confidentiality:** The system prompt is never included in API responses, logs, or error messages. It is injected server-side only.
- **Instruction override prevention:** Visitor input is treated as untrusted data. Any message attempting to override, reveal, or modify system behavior is passed to the model as user content only — the system prompt is never modified at runtime.
- **Tool call restriction:** The AI model may only invoke tools explicitly defined in the orchestrator. Any tool call not in the allow-list is rejected before execution.
- **Response schema validation:** All AI responses are validated against a defined output schema before being sent to the visitor. Malformed responses are treated as errors, not passed through.
- **Rate limiting:** `/api/chat` applies both visitor-level (20 msg/min) and store-level (3,000 replies/hour) rate limits. Both limits run on Cloudflare KV for consistency across Worker instances.
- **Prompt injection awareness:** Visitor input is structurally separated from system instructions at the API level. The model is not instructed to ignore injections — the architecture prevents them from reaching the instruction layer.

---

## Observability & Monitoring

Minimum requirements before production scaling.

**Error logging:**

- All `4xx` errors: logged at `warn` level with `store_id`, `visitor_id`, error `code`, and `path`.
- All `5xx` errors: logged at `error` level with full context including `stack`, `store_id`, and `alert: true`.
- `audit_logs` table captures all auth events, RLS violation attempts, and escalation actions.

**Latency measurement:**

- Every request logs `duration_ms` from entry to response.
- `/api/chat` logs AI provider latency separately as `ai_duration_ms`.
- Supabase query time is tracked per query using wrapper timing.

**Usage tracking:**

- `usage_tracking` table records every bot reply with timestamp and store ID.
- Spike detection threshold: store exceeding 200% of their plan's hourly average triggers a `warn` log and a dashboard notification.
- At 80% of monthly quota: notification to merchant (dashboard badge + Resend email).
- At 100% of monthly quota: widget displays alternative contact message; bot replies are blocked.

**Pre-production monitoring checklist:**

- [ ] Error rate for `/api/chat` < 1% over 24h in dev
- [ ] P95 latency for `/api/chat` < 3s under simulated load
- [ ] RLS violation detection confirmed working via test
- [ ] Usage quota enforcement confirmed at 80% and 100% thresholds
- [ ] Audit log writes confirmed non-blocking (async)

---

## Important Note

After major changes, update this file. Keep implementation status and rules current.
