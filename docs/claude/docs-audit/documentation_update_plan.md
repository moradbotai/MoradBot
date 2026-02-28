# Documentation Update Plan

**Date:** February 22, 2026 (updated — added original docs audit)
**Audit Scope:** All files in `/docs/` and `/docs/claude/` (27 files total)
**Status at Time of Audit:** Phase 3 complete, Phase 4 (Salla Client) next

---

## Original Docs Audit — February 22, 2026

تم مراجعة جميع الوثائق الـ10 في `/docs/` بواسطة 3 وكلاء متوازيين.
التفاصيل الكاملة في:
- `docs/claude/current_documentation_inventory.md` — جرد شامل لكل وثيقة
- `docs/claude/documentation_analysis.md` — تحليل تفصيلي بـ5 محاور
- `docs/claude/documentation_classification.md` — التصنيف مع المبررات والاكتشافات المشتركة
- `docs/claude/documentation_action_plan.md` — خطة الإجراءات مرتبة حسب الأولوية

### ملخص تصنيف الوثائق الأصلية

| الوثيقة | التصنيف | الإجراء المطلوب |
|---------|---------|----------------|
| MRD | 🟢 GREEN | لا إجراء |
| BRD | 🟡 YELLOW | لاحظ تعارض ICP + غياب الأسعار في ملحق |
| Extended Architecture | 🟡 YELLOW | لاحظ "Higher"→"Premium" + قيم rate limiting في ملحق |
| PRD | 🟡 YELLOW | أنشئ phase-05_widget-supplement.md |
| SRD | 🟡 YELLOW | أنشئ srd-corrections.md + phase-05_ai-orchestrator-spec.md |
| README | 🟡 YELLOW | تحديثات بسيطة (تاريخ، مؤشر docs/claude/) |
| Full Project Doc | 🟠 ORANGE | وثّق تصحيح ICP (10-500 → 30-300) في ملحق |
| 03_CLAUDE_md_Working_Standards | 🟠 ORANGE | انقل إلى docs/claude/ ثم حدّث Sections 2 و4 |
| 02_Implementation_Plan | 🟠 ORANGE | انقل إلى docs/claude/plans/ + أضف ملاحظات تصحيح |
| 01_Tools_Report | ⚫ BLACK | **أرشف فوراً** — أمر تثبيت خطير (975 مكوناً) |

### الاكتشافات المشتركة الحرجة

1. **ICP متعارض في وثيقتين:** 10-500 (Full Project Doc ❌) مقابل 30-300 (القيمة الصحيحة — CLAUDE.md)
2. **أسعار الخطط يتيمة:** 99/299/799 SAR موجودة في DB ولكن غائبة من MRD + PRD + SRD + BRD
3. **تسمية "Premium" متضاربة:** Extended Architecture وSRD يستخدمان "Higher"/"Highest"
4. **انتهاكات Rule 7:** 3 وثائق Claude-generated في `docs/` بدلاً من `docs/claude/`
5. **Multi-store غير مدعوم بالكود:** SRD يدّعي multi-store؛ UNIQUE constraint لا يدعمه

### الإجراءات الفورية (قبل Phase 4)

```bash
# 1. أرشف 01_Tools_Report.md (خطر)
mv docs/01_Tools_Report.md .claude_archive/not_useful/

# 2. انقل وثائق Claude-generated إلى موقعها الصحيح
mv docs/02_Implementation_Plan.md docs/claude/plans/implementation-plan-original.md
mv docs/03_CLAUDE_md_Working_Standards.md docs/claude/working-standards.md
```

---

## Audit Summary

| Category | Count |
|----------|-------|
| Protected original docs (do not modify) | 7 |
| Claude-generated docs — CURRENT | 9 |
| Claude-generated docs — NEEDS UPDATE | 3 |
| Claude-generated docs — HISTORICAL (archive candidates) | 1 |
| Files MISSING (must create) | 9 |
| Junk files at project root (delete) | 8 |

**Overall scores:** Completeness 75% · Accuracy 95% · Currency 80%

---

## Protected Files — Do NOT Modify (Rule 7)

| File | Status |
|------|--------|
| `docs/morad_bot_product_requirements_document_prd_v_1.md` | ✅ Authoritative |
| `docs/morad_bot_system_requirements_document_srd_v_1.md` | ✅ Authoritative |
| `docs/morad_bot_extended_architecture_document_v_1.md` | ✅ Authoritative |
| `docs/morad_bot_business_requirements_document_brd_v_1.md` | ✅ Authoritative |
| `docs/morad_bot_market_requirements_document_mrd_v_1.md` | ✅ Authoritative |
| `docs/morad_bot_full_project_documentation_v_1.md` | ✅ Authoritative |
| `docs/readme_morad_bot_documentation.md` | ✅ Authoritative |

---

## Priority 1 — Critical (Blocking Development Now)

These files must exist before Phase 4 work begins.

### 1.1 CREATE: `docs/claude/plans/phase-04_salla-client.md`

**Why critical:** Rule 8 requires a plan file before every phase. Phase 4 (Salla Client) is the next immediate phase. Without this file, Rule 8 is violated the moment Phase 4 coding starts.

**Contents to cover:**
- OAuth complete flow: start → callback → token storage in DB
- `packages/salla-client` full implementation (currently just exports a version string)
- `GET /products` client with pagination + retry logic
- Scheduled cron for product sync (Basic: 24h, Mid: 6h, Premium: 1h)
- Token refresh mechanism (single-use refresh token — critical edge case)
- Scope requirement: `products.read` + `offline_access` (currently missing from OAuth start)
- Fix: In-memory rate limiter → Cloudflare KV or Durable Objects

**Status:** ❌ Missing

---

### 1.2 CREATE: `docs/claude/pre-deploy_checklist.md`

**Why critical:** Rule 6 requires every deployment go through a manual Pre-Deploy Checklist. The rule exists but the checklist document does not.

**Contents to cover:**
- Pre-flight checks (secrets set, migrations applied, types generated)
- Security scan (no hardcoded secrets, RLS verified)
- Smoke test sequence (health endpoint, auth flow, chat endpoint)
- Rollback procedure
- Post-deploy verification

**Status:** ❌ Missing

---

### 1.3 CREATE: `docs/claude/pdpl_compliance_checklist.md`

**Why critical:** PDPL (Saudi Personal Data Protection Law) is referenced across PRD, SRD, and Architecture docs. Decisions require `/ultra-think` per CLAUDE.md. No compliance checklist exists for verification.

**Contents to cover:**
- Consent capture requirements (first message disclosure + storage consent)
- Consent logging requirements (`consent_logs` table usage)
- Data retention rules (personal data deleted within 30–90 days post-cancellation)
- Encryption requirements (AES-256 for escalation contacts)
- Audit log retention (≥ 90 days)
- What cannot be stored without consent
- Pre-launch PDPL verification checklist

**Status:** ❌ Missing

---

## Priority 2 — Important (Improve Development Clarity)

These files reduce friction and prevent mistakes during active development.

### 2.1 CREATE: `docs/claude/testing_guide.md`

**Why important:** Test coverage target is ≥ 80% (CLAUDE.md). No testing guide exists. Vitest + Playwright are configured but no strategy is documented.

**Contents to cover:**
- Unit test strategy for Hono route handlers (mock Supabase client)
- Integration test approach (Wrangler miniflare for CF Workers)
- E2E test setup with Playwright (widget + dashboard)
- Coverage reporting and CI enforcement
- What to test at each layer (routes, middleware, RLS, widget states)

**Status:** ❌ Missing

---

### 2.2 CREATE: `docs/claude/security_audit_checklist.md`

**Why important:** Security audit is a defined step in the session workflow (`security-scan`). The `security-audit` command exists but has no corresponding checklist document.

**Contents to cover:**
- RLS policy verification per table
- Injection vulnerability checklist (SQL, prompt injection)
- Auth bypass scenarios
- Rate limiter effectiveness (especially the in-memory → KV migration)
- Secrets audit (wrangler secrets vs code scan)
- OWASP Top 10 mapping for this stack

**Status:** ❌ Missing

---

### 2.3 UPDATE: `docs/claude/00_Development_Environment_Setup.md`

**Why important:** File exists but was written before Phase 3. Missing Phase 3 dependencies, wrangler secret setup, and Biome configuration details.

**What to add:**
- Wrangler secrets setup (all 7 required secrets)
- Supabase local dev setup (`supabase start`)
- Biome configuration (tabs, double quotes, 100-char width)
- `pnpm --filter` usage for per-app commands
- Known issues: in-memory rate limiter won't work in deployed Workers (uses KV)

**Status:** ⚠️ Outdated

---

### 2.4 UPDATE: `CLAUDE.md` — Implementation Status Table

**Why important:** After Phase 4 completes, the status table must be updated. Flag this as a reminder.

**What to update when Phase 4 is done:**
- Phase 4 row: `🔜 Next` → `✅ Complete`
- Phase 5 row: `⏳ Pending` → `🔜 Next`
- Add Phase 4 summary: Salla OAuth + Products client + sync cron

**Status:** ⚠️ Will need update after Phase 4 completion

---

### 2.5 ARCHIVE: `docs/claude/session-2026-02-18_summary.md`

**Why:** Session log from Feb 18. Historical record only. Not referenced by any active workflow. Keeping in place clutters active docs.

**Action:** Move to `.claude_archive/` or add a `[HISTORICAL]` header and leave in place.

**Status:** ⚠️ Historical — low priority, not urgent

---

## Priority 3 — Supplementary (Nice to Have)

These docs improve long-term maintainability but don't block current work.

### 3.1 CREATE: `docs/claude/onboarding_guide.md`

**Contents:** 5-question merchant setup flow (shipping policy, payment methods, returns, contact info, 1 custom FAQ), widget embed behavior in Salla, first-time dashboard walkthrough.

**Status:** ❌ Missing

---

### 3.2 CREATE: `docs/claude/openrouter_model_selection.md`

**Contents:** Why Gemini 2.0 Flash is primary (cost + Arabic quality), fallback criteria (timeout/error conditions), token cost estimates, how to evaluate new models, model swap procedure.

**Status:** ❌ Missing

---

### 3.3 CREATE: `docs/claude/escalation_notification_spec.md`

**Contents:** Email notification format for merchant when escalation occurs, real-time Supabase subscription setup in dashboard, escalation close flow and audit log entry.

**Status:** ❌ Missing

---

## Files to DELETE (Junk Shell Artifacts)

The following files at the project root are artifacts from failed shell commands. They are not code, not docs, and serve no purpose.

```
/moradbot/-p
/moradbot/-type
/moradbot/cp
/moradbot/echo
/moradbot/f
/moradbot/find
/moradbot/mkdir
/moradbot/✓ Done. Files copied:
```

**Action:** `rm` all 8 files. No archiving needed.

---

## Current Files — Status Reference

| File | Status | Notes |
|------|--------|-------|
| `docs/claude/plans/phase-02_database.md` | ✅ Current | 605 lines, accurate |
| `docs/claude/plans/phase-03_api-foundation.md` | ✅ Current | 1,193 lines, accurate |
| `docs/claude/phase-02_summary.md` | ✅ Current | Phase 2 completion record |
| `docs/claude/phase-03_summary.md` | ✅ Current | Phase 3 completion record |
| `docs/claude/salla_api_reference.md` | ✅ Current | 607 lines, OAuth + Products API |
| `docs/claude/environment_variables.md` | ✅ Current | All secrets + dev vars |
| `docs/claude/tools_report_v2.md` | ✅ Current | Feb 18 ecosystem audit |
| `docs/claude/exploration_report.md` | ✅ Current | Full project state (Feb 21) |
| `docs/claude/diagnostics_issues.md` | ✅ Current | Plugin/MCP fixes (Feb 21) |
| `docs/claude/claude_md_unification.md` | ✅ Current | CLAUDE.md merge record |
| `docs/claude/00_Development_Environment_Setup.md` | ⚠️ Outdated | Pre-Phase-3, needs update |
| `docs/claude/04_Reference_Project_Analysis.md` | ✅ Current | Google ADK analysis |
| `docs/claude/05_Database_Schema_Design.md` | ✅ Current | Schema details |
| `docs/claude/diagrams/01_widget_state_machine.md` | ✅ Current | 7 states documented |
| `docs/claude/diagrams/02_conversation_flow.md` | ✅ Current | Full flow |
| `docs/claude/diagrams/03_escalation_flow.md` | ✅ Current | Escalation process |
| `docs/claude/diagrams/04_usage_metering.md` | ✅ Current | Usage + limits |
| `docs/claude/diagrams/README.md` | ✅ Current | Index |
| `docs/claude/session-2026-02-18_summary.md` | ⚠️ Historical | Archive candidate |
| `docs/claude/plans/phase-04_salla-client.md` | ❌ Missing | **CRITICAL — create now** |
| `docs/claude/pre-deploy_checklist.md` | ❌ Missing | **CRITICAL — create before Phase 4** |
| `docs/claude/pdpl_compliance_checklist.md` | ❌ Missing | **CRITICAL — create before Phase 4** |
| `docs/claude/testing_guide.md` | ❌ Missing | HIGH |
| `docs/claude/security_audit_checklist.md` | ❌ Missing | HIGH |
| `docs/claude/onboarding_guide.md` | ❌ Missing | MEDIUM |
| `docs/claude/openrouter_model_selection.md` | ❌ Missing | MEDIUM |
| `docs/claude/escalation_notification_spec.md` | ❌ Missing | MEDIUM |

---

## Known Technical Gaps (Not Documentation — Code Issues)

Found during audit. Not documentation tasks but should be tracked:

| Issue | Location | Severity | Phase |
|-------|----------|----------|-------|
| In-memory rate limiter | `apps/api/src/middleware/rate-limit.ts` | **CRITICAL** — won't work in distributed CF Workers | Phase 4 |
| OAuth tokens not stored in DB | `apps/api/src/routes/auth.ts:31` | HIGH — callback writes TODO, tokens lost on Worker restart | Phase 4 |
| `products.read` scope missing | `apps/api/src/routes/auth.ts` OAuth start | HIGH — Salla won't grant product access | Phase 4 |
| `/api/chat` returns hardcoded mock | `apps/api/src/routes/chat.ts` | MEDIUM — stub, expected for Phase 3 | Phase 5 |
| `packages/salla-client` is a stub | `packages/salla-client/src/index.ts` | MEDIUM — only exports version string | Phase 4 |
| `packages/ai-orchestrator` is a stub | `packages/ai-orchestrator/src/index.ts` | MEDIUM — only exports version string | Phase 5 |
| Supabase MCP placeholder credentials | `.mcp.json` | LOW — MCP server won't connect | Anytime |

---

## Execution Order

```
1. DELETE junk root files (5 min, no planning needed)
2. CREATE phase-04_salla-client.md (required before Phase 4 coding)
3. CREATE pre-deploy_checklist.md (required before first production deploy)
4. CREATE pdpl_compliance_checklist.md (required before Phase 4 — stores real data)
5. CREATE testing_guide.md (before writing Phase 4 tests)
6. CREATE security_audit_checklist.md (before Phase 4 security review)
7. UPDATE 00_Development_Environment_Setup.md (onboarding aid)
8. CREATE onboarding_guide.md, openrouter_model_selection.md, escalation_notification_spec.md (Phase 5 prep)
9. UPDATE CLAUDE.md status table (after Phase 4 completes)
10. ARCHIVE session-2026-02-18_summary.md (cleanup, low priority)
```
