# MoradBot Claude Tools Report v2

**Date:** February 18, 2026
**Purpose:** Comprehensive audit and classification of all `.claude` ecosystem components after reorganization.

---

## 1. ملخص تنفيذي (Executive Summary)

| Category | Total | Active | Future | Not Useful |
|----------|-------|--------|--------|------------|
| Agents | 32 | 9 | 11 | 12 |
| Commands | 35 | 16 | 16 | 3 |
| Skills | 9 | 4 | 3 | 2 |
| MCP Servers | 22 | 5 | 6 | 11 |
| Hooks (PostToolUse) | 10 | 3 | — | 7 removed |
| Hooks (PreToolUse) | 5 | 2 | — | 3 removed |
| Hooks (SessionStart) | 3 | 1 | — | 2 removed |
| Hooks (Stop/Notification) | 2 | 0 | — | 2 removed |

**Before:** 22 MCP servers creating startup overhead, 15+ hooks running on every file edit, 32 agents loaded into context
**After:** 5 MCP servers, 5 essential hooks, 9 active agents

---

## 2. Agents — Full Classification

### 2.1 Active (9 agents) — `.claude/agents/`

| Agent | Description | Relevance to MoradBot |
|-------|-------------|----------------------|
| `backend-architect` | RESTful APIs, microservice boundaries, DB schemas | Core — Hono API architecture decisions |
| `typescript-pro` | Advanced TypeScript patterns, generics, e2e type safety | Core — entire stack is TypeScript |
| `database-optimizer` | Query tuning, indexing, execution plans | Core — Supabase/PostgreSQL optimization |
| `security-auditor` | Comprehensive security audits, compliance, RLS | Core — PDPL compliance, RLS policies |
| `ai-engineer` | AI system architecture, model selection, pipelines | Core — OpenRouter + Phase 4+ AI Orchestrator |
| `error-detective` | Root cause analysis, error correlation, incident diagnosis | Core — debugging API/AI errors |
| `api-documenter` | OpenAPI specs, interactive docs, code examples | Phase 3 deliverable — API documentation |
| `prompt-engineer` | LLM prompt design, optimization, A/B testing | Phase 4+ — Bot system prompt engineering |
| `technical-writer` | API references, SDK docs, getting-started guides | Phase 3+ — developer documentation |

### 2.2 Future (11 agents) — `.claude_archive/future/agents/`

| Agent | When Needed | Phase |
|-------|-------------|-------|
| `frontend-developer` | Chat widget (Preact) + Dashboard (Next.js) | Phase 5 |
| `ui-designer` | Widget UI design system | Phase 5 |
| `ui-ux-designer` | UX research and design critique | Phase 5 |
| `data-engineer` | Usage analytics pipeline, reporting | Phase 6+ |
| `incident-responder` | Production incidents after launch | Post-launch |
| `legal-advisor` | PDPL privacy policy, terms of service | Pre-launch |
| `sql-pro` | Complex analytical queries, reporting | Phase 6+ |
| `load-testing-specialist` | Performance validation before launch | Pre-launch |
| `dx-optimizer` | Dev workflow optimization for team growth | Phase 5+ |
| `payment-integration` | Stripe subscription billing | Phase 4+ |
| `api-security-audit` | REST API security audit (complements security-auditor) | Pre-launch |

### 2.3 Not Useful (12 agents) — `.claude_archive/not_useful/agents/`

| Agent | Reason |
|-------|--------|
| `python-pro` | Stack is TypeScript/Cloudflare Workers — no Python |
| `javascript-pro` | typescript-pro covers JS/TS; redundant |
| `seo-specialist` | B2B SaaS — no SEO requirements in PRD |
| `web-vitals-optimizer` | Core Web Vitals — widget is <50KB, not a content site |
| `content-marketer` | B2B product, not a content marketing platform |
| `content-curator` | Obsidian-specific tool — no Obsidian usage |
| `competitive-intelligence-analyst` | Strategy research, not engineering |
| `gemini-researcher` | Redundant with firecrawl-mcp + claude-mem |
| `connection-agent` | Obsidian vault linking — no Obsidian usage |
| `context7` | Redundant — context7 MCP server serves same purpose |
| `database-optimization` | Redundant with database-optimizer (duplicate agent) |
| `task-decomposition-expert` | Superseded by superpowers skills ecosystem |

---

## 3. Commands — Full Classification

### 3.1 Active (16 commands) — `.claude/commands/`

| Command | Purpose | Phase |
|---------|---------|-------|
| `prime` | Enhanced mode for complex tasks | All |
| `resume` | Session orchestration resume | All |
| `ultra-think` | Deep analysis — required before schema/security changes | All |
| `session-learning-capture` | End-of-session knowledge integration | All |
| `update-docs` | Sync documentation with implementation | All |
| `code-review` | Comprehensive code quality review | All |
| `debug-error` | Systematic debugging workflow | All |
| `write-tests` | Unit/integration test writing | Phase 3+ |
| `test-coverage` | Coverage analysis and gap identification | Phase 3+ |
| `security-audit` | Security vulnerability assessment | All |
| `optimize-api-performance` | API response time optimization | Phase 3+ |
| `optimize-database-performance` | Query/index optimization | Phase 3+ |
| `supabase-migration-assistant` | Generate and validate DB migrations | Phase 2+ |
| `supabase-security-audit` | RLS policy and security analysis | Phase 2+ |
| `supabase-type-generator` | TypeScript types from Supabase schema | Phase 2+ |
| `supabase-schema-sync` | Schema sync via MCP | Phase 2+ |

### 3.2 Future (16 commands) — `.claude_archive/future/commands/`

| Command | When Needed |
|---------|-------------|
| `generate-api-documentation` | After API stabilizes (Phase 4+) |
| `doc-api` | Same as above (alternative approach) |
| `performance-audit` | Pre-launch performance validation |
| `setup-development-environment` | Onboarding new team members |
| `dependency-audit` | Regular security maintenance |
| `generate-test-cases` | Phase 4+ test expansion |
| `troubleshooting-guide` | Support documentation after launch |
| `supabase-performance-optimizer` | Phase 4+ query optimization at scale |
| `supabase-realtime-monitor` | Phase 5+ escalation real-time monitoring |
| `supabase-data-explorer` | Phase 4+ data analysis |
| `architecture-scenario-explorer` | Phase 4+ scaling decisions |
| `nextjs-component-generator` | Phase 5 dashboard components |
| `refactor-code` | Phase 5+ technical debt cleanup |
| `project-health-check` | Post Phase 4 health checks |
| `memory-spring-cleaning` | Periodic memory maintenance |
| `design-database-schema` | Future schema additions |

### 3.3 Not Useful (3 commands) — `.claude_archive/not_useful/commands/`

| Command | Reason |
|---------|--------|
| `setup-docker-containers` | Stack uses Cloudflare Workers (serverless) + Supabase — no Docker |
| `all-tools` | Meta-command to display tools — covered by `/help` |
| `directory-deep-dive` | Generic exploration command — superseded by Explore agent + memory system |

---

## 4. Skills — Full Classification

### 4.1 Active (4 skills) — `.claude/skills/` + `.agents/skills/`

| Skill | Location | Purpose |
|-------|----------|---------|
| `prompt-architect` | `.agents/skills/` (symlinked) | System prompt engineering for the bot |
| `vercel-react-best-practices` | `.agents/skills/` (symlinked) | Phase 5 Dashboard (Next.js) development |
| `vercel-composition-patterns` | `.agents/skills/` (symlinked) | Phase 5 component architecture |
| `skill-creator` | `.claude/skills/` (real dir) | Creating new project-specific skills |

### 4.2 Future (3 skills) — `.claude_archive/future/skills/`

| Skill | When Needed | Reason |
|-------|-------------|--------|
| `mckinsey-research` | Pre-Series A / investor relations | Market sizing, competitive analysis |
| `web-design-guidelines` | Phase 5 — widget + dashboard UI review | Accessibility and UX compliance |
| `theme-factory` | Phase 5 — widget theming system | Multi-theme support for merchants |

### 4.3 Not Useful (2 skills) — `.claude_archive/not_useful/skills/`

| Skill | Reason |
|-------|--------|
| `vercel-react-native-skills` | MoradBot has no mobile app — stack is Web only |
| `file-organizer` | Meta-skill for file organization — this reorganization is a one-time task |

---

## 5. MCP Servers — Full Classification

### 5.1 Active (5 servers) — `.mcp.json`

| Server | Package | Purpose |
|--------|---------|---------|
| `context7` | `@upstash/context7-mcp` | Latest library docs (Hono, Supabase, etc.) |
| `supabase` | `@supabase/mcp-server-supabase` | Schema inspection, migration assistance |
| `memory` | `@modelcontextprotocol/server-memory` | Cross-session knowledge graph |
| `firecrawl-mcp` | `firecrawl-mcp` | Web research, Salla docs, API docs |
| `fetch` | `@modelcontextprotocol/server-fetch` | Lightweight URL fetching |

### 5.2 Future (6 servers) — `.claude_archive/future/mcp_servers.json`

| Server | When Needed | Notes |
|--------|-------------|-------|
| `playwright-server` | Phase 5 E2E testing | Official `@playwright/mcp` |
| `github` | Phase 5+ CI/CD | Needs real `GITHUB_PERSONAL_ACCESS_TOKEN` |
| `postgresql` | Phase 2+ reference | Needs real connection string; prefer supabase MCP |
| `TestSprite` | Phase 5 visual testing | Needs real `API_KEY` |
| `mcp-mermaid` | Architecture diagrams | Available now but not critical |
| `postgres-documentation` | PostgreSQL reference | URL-based MCP |

### 5.3 Not Useful (11 servers) — `.claude_archive/not_useful/mcp_servers.json`

| Server | Reason |
|--------|--------|
| `executeautomation-playwright-server` | DUPLICATE — official playwright-server is preferred |
| `puppeteer` | DUPLICATE — firecrawl-mcp handles all web automation |
| `browser-server` | BLOCKED — requires `OPENAI_API_KEY` (conflicts with our setup) |
| `mcp-server-box` | PLACEHOLDER — `/path/to/mcp-server-box` never configured |
| `filesystem` | PLACEHOLDER — path never configured; Claude Code has native file access |
| `markitdown` | BLOCKED — requires Docker; firecrawl handles document conversion |
| `chrome-devtools` | NO USE CASE — serverless API has no Chrome DevTools need |
| `imagesorcery-mcp` | NO USE CASE — no image manipulation in PRD |
| `DeepGraph Next.js MCP` | NICHE — deep graph of Next.js source; not needed for using Next.js |
| `DeepGraph React MCP` | NICHE — deep graph of React source; not needed for using React |
| `DeepGraph TypeScript MCP` | NICHE — deep graph of TypeScript source; not needed for writing TypeScript |

---

## 6. Hooks — Full Classification

### 6.1 Active Hooks (5 total)

| Event | Matcher | Purpose | Kept |
|-------|---------|---------|------|
| `PostToolUse` | `Edit\|MultiEdit` | Change log → `~/.claude/changes.log` | ✅ |
| `PostToolUse` | `Write` | Creation log → `~/.claude/changes.log` | ✅ |
| `PostToolUse` | `Edit\|Write` | Secret detection (semgrep, gitleaks, regex) | ✅ |
| `PreToolUse` | `Edit\|MultiEdit` | File backup to `.backups/` directory | ✅ |
| `PreToolUse` | `Edit\|MultiEdit\|Write` | Protected files guard | ✅ |
| `SessionStart` | `startup\|resume` | AGENTS.md loader for context | ✅ |

### 6.2 Removed Hooks (12 removed)

| Event | Matcher | Hook | Reason for Removal |
|-------|---------|------|-------------------|
| `PostToolUse` | `Edit` | dependency audit (package.json) | Duplicate — also was in settings.local.json |
| `PostToolUse` | `Edit` (2nd copy) | dependency audit | Exact duplicate in settings.json |
| `PostToolUse` | `Write\|Edit\|MultiEdit` | Next.js code quality enforcer | Exits with code 2, blocking edits; Next.js not the primary stack |
| `PostToolUse` | `Edit` | `npm test` auto-run | Slows every edit; 2-min Turborepo test suite is too heavy |
| `PostToolUse` | `Edit` | `npm build` auto-run | Slows every edit; not appropriate for continuous file edits |
| `PostToolUse` | `Bash` | Vercel deployment health check | Vercel not configured; adds latency to every Bash call |
| `PostToolUse` | `Bash` | Telegram notification (long bash) | Telegram not configured; no-op with overhead |
| `PostToolUse` | `*` | Desktop notification (osascript) | Excessive noise — fires on EVERY tool use |
| `PostToolUse` | `*` | Performance CSV tracking (end) | Unnecessary overhead on every tool |
| `PreToolUse` | `Edit` | Inline `.backup.timestamp` file | Creates files alongside source; `.backups/` dir approach is cleaner |
| `PreToolUse` | `Bash` | Telegram bash start timer | No-op without TELEGRAM_BOT_TOKEN |
| `PreToolUse` | `*` | Performance CSV tracking (start) | Unnecessary overhead on every tool |
| `SessionStart` | `startup` | Telegram session notification | No-op without TELEGRAM_BOT_TOKEN |
| `SessionStart` | `startup` | Vercel health check | No-op without VERCEL_TOKEN |
| `Stop` | — | Telegram session summary | No-op without TELEGRAM_BOT_TOKEN |
| `Notification` | — | Telegram input wait notification | No-op without TELEGRAM_BOT_TOKEN |

---

## 7. Settings Files Analysis

### 7.1 `settings.json` (project-level)

**Before:** 228 lines, 12 hooks, heavy per-tool overhead
**After:** 71 lines, 5 essential hooks, ~80% reduction

**Retained settings:**
- `enableAllProjectMcpServers: true`
- `env.DISABLE_TELEMETRY: "1"`
- `env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: "1"` — required for TeamCreate tool

### 7.2 `settings.local.json` (user-level, gitignored)

**Before:** 53 lines with a duplicate dependency audit hook in `hooks.PostToolUse`
**After:** 40 lines, hooks block removed entirely

**Retained:**
- `permissions.allow` — 27 pre-approved commands for smooth development
- `statusLine` — `context-monitor.py` for model name + context % display

---

## 8. Scripts

### 8.1 `.claude/scripts/context-monitor.py`

**Status:** Active — used by `statusLine` in `settings.local.json`
**Purpose:** Displays current model name and context window usage percentage in the Claude Code status bar
**Dependencies:** Python 3 (system)
**Notes:** Working correctly. No changes needed.

---

## 9. External Plugins

These are installed globally (not in this repo) and appear in the session's skill list.

| Plugin | Tools | Status | Relevance |
|--------|-------|--------|-----------|
| `superpowers` | 12 skills (brainstorming, TDD, debugging, etc.) | Active | High — core workflow skills |
| `pr-review-toolkit` | review-pr | Active | Medium — code review workflows |
| `ralph-loop` | ralph-loop, cancel-ralph, help | Active | Medium — iterative refinement |
| `claude-mem` | mem-search, do, make-plan | Active | High — cross-session memory |
| `firecrawl` | firecrawl-cli | Active | High — web research (replaces WebFetch/WebSearch) |
| `frontend-design` | frontend-design | Active | Medium — Phase 5 widget/dashboard |

---

## 10. Archive Structure Verification

```
.claude_archive/
├── future/
│   ├── agents/          (11 files: frontend-developer, ui-designer, ui-ux-designer,
│   │                     data-engineer, incident-responder, legal-advisor, sql-pro,
│   │                     load-testing-specialist, dx-optimizer, payment-integration,
│   │                     api-security-audit)
│   ├── commands/        (16 files)
│   ├── skills/          (3 dirs: mckinsey-research, web-design-guidelines, theme-factory)
│   └── mcp_servers.json (6 servers: playwright-server, github, postgresql,
│                         TestSprite, mcp-mermaid, postgres-documentation)
└── not_useful/
    ├── agents/          (12 files)
    ├── commands/        (3 files: setup-docker-containers, all-tools, directory-deep-dive)
    ├── skills/          (2 dirs: vercel-react-native-skills, file-organizer)
    └── mcp_servers.json (11 servers)
```

---

## 11. Final Recommendations

### Immediate Actions (Done ✅)
- [x] Reduced agents from 32 → 9 active
- [x] Reduced commands from 35 → 16 active
- [x] Reduced MCP servers from 22 → 5 active
- [x] Reduced hooks from 15+ → 5 essential
- [x] Removed duplicate hooks from settings.local.json
- [x] Preserved all components in archive (no deletions)

### Next Session Priorities
1. **Phase 4 kickoff** — Salla Client Package (OAuth + GET /products)
2. **Activate `legal-advisor` agent** early — PDPL compliance review before data goes live
3. **Activate `playwright-server` MCP** when widget E2E tests begin (Phase 5)
4. **Review `github` MCP** — add real `GITHUB_PERSONAL_ACCESS_TOKEN` when CI/CD needed

### Maintenance Notes
- **`.backups/` directory** will grow over time — add to `.gitignore` if not already there
- **`~/.claude/changes.log`** — consider periodic rotation (already handled for performance.csv)
- **Archive is browseable** — check `.claude_archive/future/` before building new functionality to see if an archived component already exists

---

*Report generated: February 18, 2026 | MoradBot v0.3 (Phase 3 complete)*
