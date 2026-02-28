# Diagnostics Issues Log

**Date:** February 21, 2026
**Command:** `/doctor` + `/mcp`

---

## Issues Found & Resolved

### Issue 1 — Plugin Not Found: `compounding-engineering@every-marketplace`

**Error:**
```
Plugin Error
└ compounding-engineering@every-marketplace: Plugin not found
```

**Root Cause:**
The plugin was **renamed** from `compounding-engineering` to `compound-engineering` (documented in its CHANGELOG). The old key remained in `~/.claude/settings.json`.

**Fix Applied:**
- Removed `"compounding-engineering@every-marketplace": true` from `~/.claude/settings.json`

**File Modified:** `~/.claude/settings.json` (global)

**Status:** ✅ Resolved

---

### Issue 2 — MCP Connection Failed: `fetch`

**Error:**
```
Failed to reconnect to fetch
```

**Root Cause:**
The `fetch` MCP server (`@modelcontextprotocol/server-fetch`) was defined in **two places**:
1. Global: `~/.claude/settings.json` → `mcpServers.fetch`
2. Project: `/moradbot/.mcp.json` → `mcpServers.fetch`

Duplicate server names cause a reconnect conflict when Claude loads both global and project MCP configs.

**Fix Applied:**
- Removed the duplicate `fetch` entry from `/moradbot/.mcp.json`
- The global `fetch` server in `~/.claude/settings.json` remains active

**File Modified:** `moradbot/.mcp.json`

**Status:** ✅ Resolved

---

## Current MCP Configuration (Post-Fix)

### Global (`~/.claude/settings.json`)

| Server | Package | Notes |
|--------|---------|-------|
| filesystem | @modelcontextprotocol/server-filesystem | — |
| memory | @modelcontextprotocol/server-memory | — |
| sequential-thinking | @modelcontextprotocol/server-sequential-thinking | — |
| fetch | @modelcontextprotocol/server-fetch | — |
| github | @modelcontextprotocol/server-github | — |
| postgresql | @modelcontextprotocol/server-postgres | — |
| puppeteer | @modelcontextprotocol/server-puppeteer | — |
| cloudflare | @cloudflare/mcp-server-cloudflare | — |
| n8n | n8n-mcp | — |

### Project (`moradbot/.mcp.json`)

| Server | Package | Notes |
|--------|---------|-------|
| context7 | @upstash/context7-mcp | Library docs |
| supabase | @supabase/mcp-server-supabase | Needs real project-ref + token |
| memory | @modelcontextprotocol/server-memory | Duplicate of global (OK, same name) |
| firecrawl-mcp | firecrawl-mcp | Web research |

---

## Open Issues

### Supabase MCP Server — Placeholder Credentials

**Warning:** The `supabase` server in `.mcp.json` still uses placeholder values:
```json
"--project-ref=<project-ref>"
"SUPABASE_ACCESS_TOKEN": "<personal-access-token>"
```

These must be replaced with real credentials before the Supabase MCP server will work.

**Severity:** Medium — server will fail to connect until credentials are set.

**Resolution:** Replace with actual Supabase project ref and personal access token from the Supabase dashboard.

---

## Plugins Status (Post-Fix)

| Plugin | Status |
|--------|--------|
| n8n-workflow-builder@claude-code-marketplace | Active |
| pr-review-toolkit@claude-code-plugins | Active |
| frontend-design@claude-code-plugins | Active |
| ~~compounding-engineering@every-marketplace~~ | **Removed** (plugin renamed) |
| superpowers@claude-plugins-official | Active |
| frontend-design@claude-plugins-official | Active |
| ralph-loop@claude-plugins-official | Active |
| firecrawl@claude-plugins-official | Active |
| code-review@claude-plugins-official | Active |
| claude-mem@thedotmack | Active |
