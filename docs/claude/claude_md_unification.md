# CLAUDE.md Unification Report

**Date:** February 21, 2026
**Action:** Merged 4 CLAUDE.md files into one canonical file at project root

---

## 1. Files Found

| Path | Lines | Size |
|------|-------|------|
| `/CLAUDE.md` | 333 | Root — master file |
| `apps/api/CLAUDE.md` | 22 | API-specific rules |
| `apps/dashboard/CLAUDE.md` | 17 | Dashboard-specific rules |
| `apps/widget/CLAUDE.md` | 22 | Widget-specific rules |
| **Total** | **394** | |

---

## 2. Content Analysis

### Root `CLAUDE.md` (333 lines) — Kept as base

Contained: project identity, full tech stack, all commands, complete architecture (API, DB, Widget, Dashboard), all 8 non-negotiable rules, session workflow, implementation status, Claude ecosystem, knowledge sources, performance targets, workflow orchestration.

### `apps/api/CLAUDE.md` (22 lines)

| Section | In Root? | Action |
|---------|----------|--------|
| Tech: Cloudflare Worker + Hono | ✅ Yes | Skip |
| Endpoints table (Arabic) | ✅ Partial (English in root) | Skip — root version is more detailed |
| "Rate Limiting على كل endpoint" | ✅ Implied | Added explicitly to Worker constraints |
| "Audit Log لكل عملية حساسة" | ✅ In middleware section | Skip |
| **"Timeout: 30 ثانية max"** | ❌ **Missing** | **Merged into root** |

### `apps/dashboard/CLAUDE.md` (17 lines)

| Section | In Root? | Action |
|---------|----------|--------|
| "React + Vite + TypeScript" | ⚠️ **Incorrect** — root says Next.js 15 | Discarded (root is correct) |
| **4 Dashboard sections UI layout** | ❌ **Missing** | **Merged into root** |
| "Real-time للتصعيدات فقط" | ✅ In root | Skip |
| "لا يعرض بيانات متاجر أخرى" | ✅ Rule 3 + Dashboard section | Skip |
| "كل نداء API يحمل JWT صحيح" | ✅ In Dashboard section | Skip |

### `apps/widget/CLAUDE.md` (22 lines)

| Section | In Root? | Action |
|---------|----------|--------|
| "Vanilla TypeScript فقط (بدون Framework)" | ⚠️ **Incorrect** — root says Preact + Vite | Discarded (root is correct) |
| 7 widget states (Arabic names) | ✅ In root (English names) | Skip — root version sufficient |
| "لا يظهر على /checkout/*" | ✅ In root | Skip |
| "أول رسالة دائماً: إفصاح AI" | ✅ In root | Skip |
| "موافقة قبل أي تخزين" | ✅ In root | Skip |
| "لا cookies بدون موافقة" | ✅ In root | Skip |
| "20 رسالة/دقيقة max" | ✅ Rate limit in root | Skip |

---

## 3. Contradictions Found

| Sub-file | Claim | Root Truth | Resolution |
|----------|-------|------------|------------|
| `widget/CLAUDE.md` | "Vanilla TypeScript فقط (بدون Framework)" | Preact + Vite (confirmed by package.json) | Root is correct — Preact is used |
| `dashboard/CLAUDE.md` | "React + Vite + TypeScript" | Next.js 15 App Router (confirmed by package.json + next.config.ts) | Root is correct — Next.js 15 is used |

These contradictions appear to be outdated notes written before the tech stack was finalized.

---

## 4. What Was Merged into Root

### Added under `### API Worker` section — Worker constraints block:

```
- Every request must carry a valid store_id
- Rate limiting applies to every endpoint
- Audit log required for all sensitive operations
- Worker timeout: 30 seconds maximum
```

### Added under `### Dashboard` section — 4 sections + rules block:

```
4 Dashboard sections:
1. الرئيسية (Home) — Bot usage stats + on/off toggle
2. المحادثات (Conversations) — Read-only conversation list
3. التصعيدات (Escalations) — View + manual close
4. الإعدادات (Settings) — FAQ management + subscription info

Dashboard rules:
- Never display data from other stores
- Every API call must carry a valid JWT
- Real-time subscriptions for escalations only
```

---

## 5. Archive

Old files moved to: `.claude_archive/old_claude_md_files/`

| Archived File | Original Path |
|---------------|---------------|
| `api_CLAUDE.md` | `apps/api/CLAUDE.md` |
| `dashboard_CLAUDE.md` | `apps/dashboard/CLAUDE.md` |
| `widget_CLAUDE.md` | `apps/widget/CLAUDE.md` |

---

## 6. Final State

**Single canonical CLAUDE.md:** `/CLAUDE.md` (~345 lines)

This file is now the **only** CLAUDE.md in the project. It contains all rules from the sub-files plus the complete project-wide guidance. No information was lost.

To restore a sub-file: copy from `.claude_archive/old_claude_md_files/`
