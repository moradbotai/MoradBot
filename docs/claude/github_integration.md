# GitHub Integration — MoradBot

**التاريخ:** فبراير 28, 2026
**الـ Repo:** `https://github.com/moradbotai/Morad_Bot.git`
**الـ Branch الرئيسي:** `main`

---

## الوضع الحالي

- Git مُهيَّأ محلياً ✅
- آخر commit: `1797074` — `docs(research): add Salla API reference and session setup`
- لا remote مُضبوط بعد

---

## ⚠️ تحذير أمني — يجب قراءته أولاً

### `.mcp.json` يحتوي على keys حقيقية

هذا الملف **مُتتبَّع بـ Git** وفيه:
- `FIRECRAWL_API_KEY` حقيقي (موجود في git history)
- `SUPABASE_ACCESS_TOKEN` حقيقي (في working tree)

**قبل أي push يجب:**
1. إضافة `.mcp.json` إلى `.gitignore`
2. تشغيل `git rm --cached .mcp.json`
3. إذا الـ repo سيكون **public**: دوِّر (rotate) المفاتيح فوراً + rewrite history

---

## Branch Strategy

```
main
 └── phase/04-salla-client     ← Phase 4: Salla OAuth
      └── feat/oauth-handler   ← feature داخل phase
      └── fix/token-refresh    ← bugfix
      └── docs/api-reference   ← توثيق
 └── phase/05-ai-orchestrator  ← Phase 5: AI
 └── phase/06-widget-ui        ← Phase 6: Widget
```

**قاعدة صارمة:** لا commit مباشرة على `main`.

---

## Commit Message Convention

```
<type>(<scope>): <description في حدود 72 حرف>

[body اختياري — لماذا وليس ماذا]
[footer: Breaking Change / Closes #issue]
```

### Types المعتمدة

| Type | الاستخدام |
|------|-----------|
| `feat` | ميزة جديدة |
| `fix` | إصلاح خطأ |
| `docs` | توثيق فقط |
| `chore` | build، config، dependencies |
| `refactor` | إعادة هيكلة بدون تغيير سلوك |
| `security` | إصلاح أمني |
| `test` | إضافة أو تحديث tests |
| `perf` | تحسين أداء |

### Scopes المعتمدة

`api` | `widget` | `dashboard` | `db` | `auth` | `ai` | `docs` | `deps`

### أمثلة

```bash
feat(api): add Salla OAuth callback endpoint
fix(db): correct RLS policy missing store_id filter
security: add .mcp.json to .gitignore and remove from tracking
docs(claude): add github integration workflow documentation
chore(deps): update hono to 4.11.9
```

---

## PR Process

```
1. أنشئ branch: git checkout -b phase/04-salla-client
2. اعمل بـ TDD: tests أولاً ثم implementation
3. Commit صغير وكثير (كل خطوة منطقية = commit)
4. افتح PR مع:
   - Title واضح (يتبع convention)
   - Description: ماذا وراء الـ changes
   - Self-review: راجع diff قبل فتح الـ PR
5. Merge strategy:
   - Features صغيرة: Squash and merge
   - Phases كاملة: Create a merge commit
6. احذف الـ branch بعد الـ merge
```

---

## Checklist ما قبل كل Push

```
□ git status يُظهر فقط الـ files المقصودة
□ git diff --cached: لا secrets، لا .env، لا .mcp.json
□ .mcp.json في .gitignore ✓
□ لا ملفات > 5MB في staging
□ Commit message يتبع الـ convention
□ Tests تمر (Phase 8 فصاعداً)
□ لا console.log متروك في production code
```

---

## GitHub Settings الموصى بها

### Repository Settings
- **Visibility:** Private (حتى إشعار آخر)
- **Default branch:** main

### Branch Protection (main)
```
✅ Require pull request before merging
✅ Block force pushes
✅ Restrict deletions
✅ Do not allow bypassing the above settings
□ Require status checks (فعِّل عند إضافة CI/GitHub Actions)
```

### GitHub Actions (مستقبلاً — Phase 8)
- Lint check على كل PR
- Type check على كل PR
- Tests على كل PR
- ❌ لا auto-deploy (Rule 6 في CLAUDE.md)

---

## الـ Workflow اليومي

```bash
# بداية عمل جديد
git checkout main
git pull origin main
git checkout -b feat/my-feature

# أثناء العمل
git add specific-files
git commit -m "feat(api): add specific behavior"

# عند الانتهاء
git push origin feat/my-feature
# افتح PR على GitHub
```

---

## الـ Secrets — القاعدة الذهبية

```
🔴 لا secrets في source code
🔴 لا secrets في .env (في production)
🔴 لا secrets في .mcp.json (مُضمَّن في .gitignore)
✅ كل secrets عبر: wrangler secret put
✅ للـ local dev: .env.local (في .gitignore)
✅ للـ MCP: .mcp.json محلي فقط (في .gitignore)
```
