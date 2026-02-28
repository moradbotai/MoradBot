# GitHub Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** ربط مستودع MoradBot بـ GitHub بأمان تام مع workflow واضح للمستقبل.

**Architecture:** تنظيف الـ secrets من Git، إضافة `.mcp.json` إلى `.gitignore`، commit جميع التغييرات المعلقة، ثم push إلى `https://github.com/moradbotai/Morad_Bot.git`.

**Tech Stack:** Git, GitHub, bash

---

## ⚠️ مشاكل أمنية مكتشفة (يجب حلها قبل أي push)

### المشكلة 1: `.mcp.json` يحتوي على API keys حقيقية
- `FIRECRAWL_API_KEY: fc-609db51c9ec24d4d9d1f00cc7dbe3dd5` — موجود في git history منذ commit الأول
- `SUPABASE_ACCESS_TOKEN: sbp_578f...` — موجود في working tree (لم يُكمَّت بعد)

**الحل:** إضافة `.mcp.json` إلى `.gitignore` ثم إزالته من tracking. إذا الـ repo سيكون **public** → يجب rewrite للـ history. إذا **private** → يكفي الإضافة إلى `.gitignore` مع تدوير المفاتيح.

### المشكلة 2: `.env.example` يكشف Supabase project ref
- `SUPABASE_URL` يحتوي على `qvujnhkfqwqfzkkweylk` — وهذا project reference ليس سراً بالضرورة، لكن يُفضَّل الانتباه.

---

## Task 1: تأمين `.mcp.json` قبل أي شيء

**Files:**
- Modify: `.gitignore`
- Delete from tracking: `.mcp.json`
- Create: `.mcp.json.example`

**Step 1: أضف `.mcp.json` إلى `.gitignore`**

```bash
# في نهاية .gitignore أضف:
# MCP Server config (contains API keys)
.mcp.json
```

**Step 2: أزل `.mcp.json` من git tracking**

```bash
git rm --cached .mcp.json
```

Expected output:
```
rm '.mcp.json'
```

**Step 3: أنشئ `.mcp.json.example` بقيم وهمية**

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    },
    "supabase": {
      "command": "npx",
      "args": [
        "-y",
        "@supabase/mcp-server-supabase@latest",
        "--read-only",
        "--project-ref=YOUR_PROJECT_REF"
      ],
      "env": {
        "SUPABASE_ACCESS_TOKEN": "sbp_YOUR_PERSONAL_ACCESS_TOKEN"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "firecrawl-mcp": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "fc-YOUR_FIRECRAWL_API_KEY"
      }
    }
  }
}
```

**Step 4: تحقق من الوضع**

```bash
git status
```

Expected: `.mcp.json` يظهر في "untracked files" (ليس staged). `.mcp.json.example` يظهر كـ "new file".

**Step 5: Commit الإصلاح الأمني**

```bash
git add .gitignore .mcp.json.example
git commit -m "security: remove .mcp.json from tracking (contains API keys)

- Add .mcp.json to .gitignore
- Create .mcp.json.example with placeholder values
- Existing keys should be rotated if repo will be made public"
```

---

## Task 2: مراجعة وتنظيف الملفات قبل الـ commit الكبير

**Step 1: افحص كل الملفات المعدَّلة**

```bash
git diff --stat
git status
```

**Step 2: تحقق من عدم وجود secrets في الكود**

```bash
# البحث عن patterns خطيرة
grep -r "sbp_" . --exclude-dir=.git --exclude="*.md" 2>/dev/null
grep -r "sk-" . --exclude-dir=.git --exclude="*.md" 2>/dev/null
grep -r "fc-[a-f0-9]" . --exclude-dir=.git 2>/dev/null
```

Expected: لا نتائج (أو فقط في `.mcp.json` المُستثنى الآن)

**Step 3: تحقق من الملفات الكبيرة**

```bash
find . -not -path "./.git/*" -not -path "./node_modules/*" -size +1M 2>/dev/null
```

Expected: لا ملفات كبيرة

**Step 4: تحقق من حالة `.gitignore` للـ `.claude/` folder**

```bash
git check-ignore -v .claude/
```

Expected: `.gitignore:36:.claude/`  ← يعني `.claude/` مستثنى صح ✓

---

## Task 3: إضافة GitHub remote

**Step 1: تحقق من عدم وجود remote حالي**

```bash
git remote -v
```

Expected: لا output (لا يوجد remote)

**Step 2: أضف الـ remote**

```bash
git remote add origin https://github.com/moradbotai/Morad_Bot.git
```

**Step 3: تحقق**

```bash
git remote -v
```

Expected:
```
origin  https://github.com/moradbotai/Morad_Bot.git (fetch)
origin  https://github.com/moradbotai/Morad_Bot.git (push)
```

---

## Task 4: تنظيم وcommit التغييرات المعلقة

الملفات المعدَّلة المعلقة من جلسات سابقة:

**Step 1: Stage الملفات الأساسية (modified)**

```bash
git add CLAUDE.md README.md
git add apps/api/src/env.ts apps/api/wrangler.toml
git add tsconfig.base.json turbo.json
```

**Step 2: Stage ملفات محذوفة (deleted app-level CLAUDE.md)**

```bash
git add apps/api/CLAUDE.md apps/dashboard/CLAUDE.md apps/widget/CLAUDE.md
```

**Step 3: Stage docs الجديدة**

```bash
git add docs/claude/
```

**Step 4: Stage ملفات جديدة مهمة**

```bash
git add .env.example
git add docs_v2/
git add supabase/config.toml supabase/seed.sql
git add .mcp.json.example
```

**ملاحظة:** `.claude_archive/` و `.agents/` — لا تُضف إلا إذا أردت version control عليها. عادةً تُستثنى.

**Step 5: تحقق مما سيُكمَّت**

```bash
git diff --cached --stat
```

**Step 6: Commit**

```bash
git commit -m "chore: sync all pending changes from phases 1-3 work

- Update CLAUDE.md with complete tech stack, architecture, and rules
- Update README with project overview
- Update wrangler.toml with CF Workers configuration
- Update env.ts with rate limiting environment bindings
- Add docs_v2/ with v2 documentation suite (marketing, costs, architecture)
- Add docs/claude/ research and planning documents
- Add supabase/config.toml and seed.sql
- Add .env.example for developer onboarding
- Add .mcp.json.example (safe placeholder config)
- Remove app-level CLAUDE.md files (consolidated to root)
- Clean up tsconfig.base.json and turbo.json"
```

---

## Task 5: إعداد branch وpush الأول

**Step 1: تأكد أن branch اسمه `main`**

```bash
git branch -M main
```

**Step 2: Push الأول**

```bash
git push -u origin main
```

Expected:
```
Enumerating objects: ...
To https://github.com/moradbotai/Morad_Bot.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

إذا فشل بسبب repository غير موجود أو غير مُهيَّأ، أنشئه على GitHub أولاً:
- الاسم: `Morad_Bot`
- **Private** (موصى به بشدة)
- لا تُضف README أو .gitignore أو License (الـ repo عندنا محلياً بالفعل)

---

## Task 6: إعداد Branch Protection Rules على GitHub

يُنفَّذ يدوياً من واجهة GitHub (لا يمكن عبر CLI بدون GitHub token):

**الخطوات:**
1. اذهب إلى: `https://github.com/moradbotai/Morad_Bot/settings/branches`
2. اضغط "Add branch ruleset"
3. اسم الـ ruleset: `main-protection`
4. Target: `main` branch
5. فعِّل هذه القواعد:

```
✅ Require a pull request before merging
   - Required approvals: 1 (أو 0 إذا كنت تعمل لوحدك)
✅ Require status checks to pass (عند إضافة CI لاحقاً)
✅ Do not allow bypassing the above settings
✅ Restrict deletions
✅ Block force pushes
```

---

## Task 7: توثيق GitHub Workflow

**Files:**
- Create: `docs/claude/github_integration.md`

**المحتوى (انظر القسم التالي)**

---

## المستقبل: GitHub Workflow

### Branch Strategy

```
main          ← production-ready code فقط
  └── phase/XX-name     ← branch لكل phase
       └── feat/name    ← feature داخل الـ phase
       └── fix/name     ← bugfix
       └── docs/name    ← توثيق فقط
```

**قاعدة:** لا commit مباشرة على `main` إلا في حالات الطوارئ أو التوثيق البسيط.

### Commit Message Convention

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

**Types:**
- `feat` — ميزة جديدة
- `fix` — إصلاح خطأ
- `docs` — توثيق فقط
- `chore` — تغييرات build/config
- `refactor` — إعادة هيكلة بدون تغيير السلوك
- `security` — إصلاح أمني
- `test` — إضافة/تحديث tests

**Scope:** `api`, `widget`, `dashboard`, `db`, `auth`, `ai`, `docs`

**أمثلة:**
```
feat(api): add Salla OAuth callback handler
fix(db): correct RLS policy for faq_entries table
security: rotate compromised API key in .mcp.json
docs(api): add endpoint documentation for /api/chat
```

### PR Process

```
1. أنشئ branch من main
2. اعمل على الـ feature/fix
3. افتح PR مع description واضح
4. Code Review (حتى لو أنت الوحيد — راجع الكود بعد يوم)
5. Merge بـ "Squash and merge" للـ features الصغيرة
   أو "Create a merge commit" للـ phases الكاملة
6. Delete branch بعد الـ merge
```

### Rules (Rule 6 من CLAUDE.md)

- ❌ لا Auto-deploy من CI/CD إلى production
- ✅ كل deploy يدوي عبر `wrangler deploy` بعد Pre-Deploy Checklist
- ❌ لا `git push --force` على `main`
- ✅ الـ secrets دائماً عبر `wrangler secret put` — لا في كود أو `.env`

---

## Checklist ما قبل كل Push

```
□ git status نظيف (لا unintended files)
□ لا secrets في git diff
□ .mcp.json غير مُضمَّن
□ .env غير مُضمَّن
□ Tests تمر (إذا وُجدت)
□ Commit message واضح
```
