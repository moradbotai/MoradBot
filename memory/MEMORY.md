# MoradBot Project Memory

> Concise cross-session knowledge. Details in topic files linked below.
> Keep under 200 lines. Last updated: 2026-03-03.

---

## Project Identity

- **MoradBot** — B2B SaaS, AI FAQ bot for Saudi Salla merchants
- **Repo:** `https://github.com/moradbotai/Morad_Bot.git` (private)
- **Branch:** `main` — only production-ready code
- **Current Phase:** Phase 4 (Salla OAuth) — Phases 1-3 complete

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 | ✅ Done | Turborepo, Biome, base packages |
| 2 | ✅ Done | 12-table DB, 5 migrations, RLS, TS types |
| 3 | ✅ Done | Hono API: 16 endpoints, middleware, errors |
| 4 | 🔜 Next | Salla Client Package (OAuth + GET /products) |
| 5+ | ⏳ Wait | AI, Widget UI, Dashboard UI |

---

## Critical Security Rules

- **`.mcp.json` is in `.gitignore`** — contains real API keys, NEVER commit
- Use `.mcp.json.example` as template for new devs
- Firecrawl key `fc-609db51c9ec24d4d9d1f00cc7dbe3dd5` exists in old git history — rotate if repo goes public
- All production secrets → `wrangler secret put` only (Rule 4)
- Every DB query in route handlers must include `.eq("store_id", storeId)` (Rule 3)

---

## Active MCP Servers

All configured in `.mcp.json` (local only, gitignored):
- `context7` — library docs
- `supabase` — DB queries (read-only, project: `qvujnhkfqwqfzkkweylk`)
- `memory` — persistent memory
- `firecrawl-mcp` — web scraping/search
- `github` — GitHub repo management (PAT configured Feb 28 2026)

---

## Key File Locations

| What | Where |
|------|-------|
| API entry | `apps/api/src/index.ts` → `src/app.ts` |
| Env types | `apps/api/src/env.ts` |
| Error hierarchy | `apps/api/src/lib/errors.ts` |
| Supabase clients | `apps/api/src/lib/supabase.ts` |
| DB types | `packages/shared/` |
| v2 docs | `docs_v2/` (authoritative) |
| Plans | `docs/claude/plans/` |
| GitHub workflow | `docs/claude/github_integration.md` |

---

## GitHub Workflow (established Feb 28 2026)

```
Branch: phase/XX-name → feat/name or fix/name
Commits: <type>(<scope>): <description>
Types: feat | fix | docs | chore | refactor | security | test
Scopes: api | widget | dashboard | db | auth | ai | docs
```

- No direct commits to `main`
- No auto-deploy (manual wrangler deploy only — Rule 6)
- No force push on `main`

→ Details: `docs/claude/github_integration.md`

---

## Session Closure Workflow (established Mar 3 2026)

**Session-closer agent** — Automated 4-step end-of-session closure:
1. `/end1-3-session-learning-capture` — Extract learnings to memory
2. `/end2-3-update-docs` — Sync CLAUDE.md, docs_v2/
3. `/end3-3-claude-sync-manager` — Verify claude.md ↔ gemini.md ↔ agents.md identical
4. `/github-setup-workflow` — Stage, commit, push to origin/main

**Key patterns:**
- Sequential execution only (no parallelization)
- All steps must succeed or workflow stops
- User confirmation required before beginning
- Real-time status reporting during execution
- Error at any step halts workflow immediately

**Pre-closure checklist:**
- [ ] git status is clean (all code committed)
- [ ] Tests passing (if applicable)
- [ ] CLAUDE.md updated with decisions
- [ ] GitHub remote configured
- [ ] No secrets in staged files

---

## Pricing Tiers (current)

| Tier | Price | Responses |
|------|-------|-----------|
| Free Trial | 0 SAR / 30 days | 30 total |
| Starter | 97 SAR/mo | 1,000/mo |
| Growth | 197 SAR/mo | 3,000/mo |
| Scale | 449 SAR/mo | 8,000/mo |

Setup fee: 450 SAR (one-time). Payment: Moyasar (150 SAR/yr + 2.75%).

→ Details: `docs_v2/marketing_strategy_moradbot.md`, `docs_v2/tools_and_costs.md`

---

## Session Notes

→ Detailed session learnings: `memory/sessions.md`
