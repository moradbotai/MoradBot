# GitHub Integration Documentation

## Repository Configuration

- **URL:** https://github.com/moradbotai/MoradBot
- **Branch Strategy:** main (production) + feature/phase branches
- **Setup Date:** 2026-03-03
- **Status:** Initial setup complete

---

## Workflow Rules

### Branch Strategy

- **`main`** — Production-ready code only
  - Protected branch: no direct commits
  - No force push allowed
  - Requires PR review before merge

- **`phase/XX-name`** — Phase development branch
  - Example: `phase/04-salla-oauth`
  - Created from main when starting new phase
  - Merged back to main after phase completion

- **`feat/name`** — Feature branch (from phase)
  - Example: `feat/rate-limiting`
  - Short-lived, merged via PR

- **`fix/name`** — Bug fix branch (from phase)
  - Example: `fix/widget-css-overflow`
  - Short-lived, merged via PR

---

### Commit Message Convention

**Format:**
```
<type>(<scope>): <description>
```

**Types:**
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation only
- `chore` — Setup, config, dependencies
- `refactor` — Code restructuring (no behavior change)
- `security` — Security fix
- `test` — Test additions/updates

**Scopes:**
- `api` — Cloudflare Worker / Hono API
- `widget` — Preact chat widget
- `dashboard` — Next.js merchant dashboard
- `db` — Database schema / migrations
- `auth` — Authentication (Salla OAuth, JWT)
- `ai` — AI orchestrator, model integration
- `docs` — Documentation updates

**Examples:**
```bash
feat(api): add rate limiting middleware
docs(db): update schema migration guide
fix(widget): resolve CSS overflow issue
chore(deps): upgrade Hono to 4.12.0
security(api): validate JWT expiration
```

---

### Pull Request Process

1. **Create PR** from feature/fix branch → phase branch (or main for hotfixes)
2. **Require 1 code review approval** (via GitHub branch protection)
3. **All CI checks must pass** (when enabled)
4. **Merge via "Squash and merge"** (default)
   - Keeps main history clean
   - All commits collapsed to single commit
5. **Delete branch after merge** (automatic)
6. **Phase → Main**: Requires manual review (Rule 6 — no auto-deploy)

---

### Pre-Push Checklist

Before pushing code to any branch:

- [ ] All tests passing
- [ ] No `console.log` or debug code
- [ ] TypeScript type-check passes
  ```bash
  pnpm type-check
  ```
- [ ] Biome lint passes
  ```bash
  pnpm lint
  ```
- [ ] No secrets or `.env` files staged
  ```bash
  git status
  ```
- [ ] Commit message follows convention
- [ ] Branch is up to date with base branch
  ```bash
  git pull --rebase origin main
  ```

---

### Deployment Rules

**Rule 6 — Manual Deployment Always**

- ❌ No CI/CD auto-deploy to production
- ✅ Every deploy: Manual via Wrangler
  ```bash
  cd apps/api
  pnpm deploy
  ```
- ✅ Every deploy: Manual Pre-Deploy Checklist
  - [ ] All tests passing
  - [ ] Code reviewed
  - [ ] Staging environment verified
  - [ ] Secrets configured in Wrangler
  - [ ] No breaking changes to API

---

## Protected Resources

| Resource | Protection |
|----------|-----------|
| `main` branch | No direct commits, no force push, require PR |
| `.env` files | Always in `.gitignore`, never committed |
| `.mcp.json` | Local MCP secrets, in `.gitignore` |
| `.wrangler/` | Local Cloudflare config, in `.gitignore` |
| Secrets | Cloudflare/Supabase env variables only |

---

## Security Policy

### Secrets Management

**NEVER commit:**
- API keys (OpenRouter, Supabase, Firecrawl)
- OAuth credentials (Salla client secret)
- JWT signing keys
- Encryption keys
- GitHub tokens
- MCP configuration with keys

**INSTEAD use:**
- Cloudflare Secrets: `wrangler secret put NAME`
- Supabase Environment Variables
- GitHub Actions Secrets (if using CI/CD)

### Pre-Commit Hook

Verify no secrets leak before committing:

```bash
# Check for common secret patterns
grep -r "OPENROUTER_API_KEY" . --exclude-dir=node_modules
grep -r "SUPABASE_.*KEY" . --exclude-dir=node_modules
grep -r "SALLA_CLIENT_SECRET" . --exclude-dir=node_modules
grep -r "ENCRYPTION_KEY" . --exclude-dir=node_modules
```

If found, **STOP immediately** and remove secrets.

---

## First Sync (Setup)

This documentation was created during initial GitHub setup.

**Initial Commit:**
```
chore(setup): initial commit - Morad Bot SaaS project

- Initialize Git and GitHub integration
- Include all source code, migrations, and documentation
- Set up project structure with Turborepo monorepo
- Configure TypeScript, Biome, and testing infrastructure
- Database migrations ready for Phase 2 deployment
- All secrets excluded (use Cloudflare/Supabase for production)

Co-Authored-By: MoradBot Setup <setup@moradbot.dev>
```

---

## Next Steps

After initial setup:

1. **Create Phase Branches:**
   ```bash
   git checkout -b phase/04-salla-oauth
   ```

2. **Set Up Team Access:**
   - Add collaborators to GitHub repository
   - Assign roles: Maintainer, Developer, etc.

3. **Configure Branch Protection (GitHub Settings):**
   - Require pull request reviews
   - Require status checks
   - Require branches up to date
   - Restrict who can push

4. **Enable GitHub Actions (if needed):**
   - Set up automated testing
   - Configure linting
   - Configure deployment workflows

5. **Start Development:**
   - Begin Phase 4 work on `phase/04-salla-oauth` branch
   - Follow commit convention and PR process documented above

---

## Useful Commands

```bash
# View branch list
git branch -a

# Create and switch to feature branch
git checkout -b feat/feature-name

# Push new branch
git push -u origin feat/feature-name

# Create PR (via GitHub UI or gh CLI)
gh pr create --title "Feature: ..." --body "Fixes #123"

# Sync with main
git pull --rebase origin main

# View commit history
git log --oneline -10

# View changes
git diff main
git diff --staged
```

---

## References

- **MoradBot CLAUDE.md:** Core project rules and architecture
- **CLAUDE.md Rule 6:** Manual Deployment Always (no auto-deploy)
- **CLAUDE.md Rule 4:** Secrets in Cloudflare Secrets Only
- **docs_v2/:** Official product requirements and architecture decisions
