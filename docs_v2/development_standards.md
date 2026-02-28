# مراد بوت — دليل قواعد العمل
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.0
**الحالة:** معتمد

---

## الهدف
مرجع موحّد لبروتوكولات العمل اليومي مع Claude Code على مشروع مراد بوت.

## النطاق
✅ يغطي:
- بروتوكول الجلسة (بداية + أثناء + نهاية)
- Hooks النشطة الفعلية (5 hooks)
- خريطة الـ Agents النشطين (9 agents)
- بروتوكول الطوارئ والـ Checklists

❌ لا يغطي:
- محتوى CLAUDE.md (راجعه مباشرةً — هو المرجع الوحيد والمحدَّث دائماً)
- ملفات per-app CLAUDE.md (تم التخلي عنها)
- Agents أو Hooks أرشيفية

---

## 1. ملف CLAUDE.md الرئيسي

راجع root CLAUDE.md — هو المرجع الوحيد والمحدَّث دائماً.

المسار: `/Users/mohammedaljohani/Documents/Proj/moradbot/CLAUDE.md`

يتضمن: هوية المشروع، Tech Stack، القواعد غير القابلة للكسر (Rules 1-8)، بنية المشروع، حالة المراحل، أوامر الجلسة، وقرارات تتطلب `/ultra-think`.

---

## 2. نهج CLAUDE.md لكل تطبيق

تم التخلي عن نهج per-app CLAUDE.md. جميع القواعد في CLAUDE.md الجذر فقط.

الملفات الثلاثة السابقة (`apps/api/CLAUDE.md`، `apps/widget/CLAUDE.md`، `apps/dashboard/CLAUDE.md`) حُذفت ولن تُعاد.

---

## 3. بروتوكول جلسة العمل اليومي

### 3.1 بداية الجلسة

```bash
# الخطوة 1: تهيئة الجلسة
/prime
# Claude يُحمّل CLAUDE.md ويستوعب السياق الكامل

# الخطوة 2: استئناف من حيث توقف
/resume
# Claude يُحدّد المرحلة الحالية ويقترح الخطوة التالية
```

### 3.2 أثناء العمل — قواعد التعامل

| الموقف | الفعل الصحيح |
|--------|-------------|
| قرار معماري جديد | `/ultra-think` أولاً |
| تغيير في Schema | `/ultra-think` + أوامر supabase |
| ميزة غير موجودة في PRD | رفض مباشر (Rule 1) |
| خطأ غير مفهوم | `/debug-error` |
| كود جديد جاهز | `/code-review` |
| Performance بطيء | `/optimize-api-performance` |
| مشكلة أمنية | `/security-audit` |

### 3.3 نهاية الجلسة

```bash
# الخطوة 1: تسجيل ما تعلمناه
/session-learning-capture
# تُسجَّل: قرارات المشاكل + الدروس + ما تبقى

# الخطوة 2: تحديث التوثيق
/update-docs
# تحديث أي وثيقة تغيّرت

# الخطوة 3: commit
git add [الملفات المحددة]
git commit -m "feat(phase-X): وصف مختصر"
```

---

## 4. Hooks النشطة الفعلية

المصدر: `.claude/settings.json`

### 4.1 الـ 5 Hooks النشطة

#### PreToolUse — backup-before-edit
- **المُشغِّل:** `Edit | MultiEdit`
- **الوظيفة:** ينسخ الملف إلى `.backups/` قبل أي تعديل
- **السبب:** حماية من التغييرات غير المقصودة

#### PreToolUse — file-protection
- **المُشغِّل:** `Edit | MultiEdit | Write`
- **الوظيفة:** يمنع التعديل على المسارات المحمية (`*/etc/*`، `*/usr/bin/*`، `*.production.*`، `*prod*config*`، `*/node_modules/*`، `*/vendor/*`)
- **السبب:** صون الملفات الحرجة من الكتابة الخاطئة

#### PostToolUse — change-tracker
- **المُشغِّل:** `Edit | MultiEdit | Write`
- **الوظيفة:** يسجّل كل تعديل أو إنشاء ملف في `~/.claude/changes.log` مع الوقت
- **السبب:** سجل تدقيق للتغييرات أثناء الجلسة

#### PostToolUse — security-scanner
- **المُشغِّل:** `Edit | Write`
- **الوظيفة:** يُشغِّل تلقائياً (إن توفّرت): semgrep، bandit (Python)، gitleaks — ويُحذِّر من أي secret مُضمَّن
- **السبب:** اكتشاف المشاكل الأمنية فور حدوثها (Rule 4)

#### SessionStart — agents-md-loader
- **المُشغِّل:** `startup | resume`
- **الوظيفة:** يُحمِّل محتوى `AGENTS.md` كـ context إضافي عند بدء كل جلسة
- **السبب:** ضمان استيعاب Claude لقواعد الوكلاء فور الانطلاق

### ملاحظة
الـ Hooks الـ 16 الموجودة في النسخة القديمة من هذه الوثيقة كانت مخطط نظري. الواقع الفعلي هو 5 hooks فقط كما هو موثَّق في `.claude/settings.json`.

---

## 5. خريطة الـ Agents النشطين

الـ Agents الـ 9 النشطة فقط (المصدر: CLAUDE.md).

| الـ Agent | الدور | المراحل ذات الصلة |
|----------|-------|------------------|
| `backend-architect` | تصميم البنية، Middleware، Workers | Phase 3، 4، 7 |
| `typescript-pro` | TypeScript patterns، type safety | جميع المراحل |
| `database-optimizer` | Schema، RLS، Migration، Queries | Phase 2 |
| `security-auditor` | Security review، PDPL، secrets | Phase 8، مستمر |
| `ai-engineer` | OpenRouter، fallback logic، token counting | Phase 5 |
| `error-detective` | Debugging، stack traces، root cause | مستمر |
| `api-documenter` | توثيق endpoints، OpenAPI | Phase 3، 4 |
| `prompt-engineer` | System Prompt للبوت، جودة الردود | Phase 5 |
| `technical-writer` | توثيق المشروع، Runbooks | Phase 9 |

**الـ Agents المؤرشفة** — لا تُستخدم في MVP:
- موقعها: `.claude_archive/future/agents/` (11 agent) و `not_useful/agents/` (12 agent)
- تشمل: dx-optimizer، sql-pro، frontend-developer، legal-advisor، إلخ

---

## 6. بروتوكول الطوارئ

### 6.1 الحوادث الحرجة (تتطلب إغلاقاً فورياً)

| نوع الحادثة | الإجراء الفوري |
|------------|--------------|
| تسرّب بيانات متاجر | إغلاق Chat Widget فوراً |
| اختراق قاعدة البيانات | إيقاف جميع الـ Workers |
| استخدام غير مصرح به للـ tokens | إلغاء جميع الـ tokens + إعادة توليد |
| Bug يُعطي بيانات خاطئة للعملاء | إيقاف AI Orchestrator + escalate manually |

### 6.2 Runbook الإغلاق الفوري

```bash
# 1. إيقاف Chat Widget
wrangler deploy --env production apps/widget/ --var BOT_ENABLED=false

# 2. إيقاف API
wrangler deploy --env production apps/api/ --var MAINTENANCE_MODE=true

# 3. إشعار التجار
# (يدوي عبر Dashboard أو إيميل مباشر)

# 4. تحقيق وإصلاح
# ...

# 5. إعادة التشغيل بعد التأكيد
```

---

## 7. Checklists الأساسية

### 7.1 Pre-Commit Checklist
```
□ الاختبارات تنجح (0 failures)
□ Biome لا يُظهر أخطاء
□ security-scanner لم يجد مشاكل
□ لا secrets مُضمَّنة في الكود
□ CLAUDE.md محدَّث إذا تغيّر الهيكل أو حالة المراحل
```

### 7.2 Pre-Deploy Checklist
```
□ جميع الاختبارات تنجح
□ Test Coverage ≥ 80%
□ Security Audit: 0 Critical Issues
□ Supabase Migrations مُطبَّقة
□ Cloudflare Secrets مُعدَّة (wrangler secret put ...)
□ Monitoring Alerts مُعدَّة
□ Rollback Plan موثَّق
□ موافقة صريحة على النشر (Rule 6)
```

### 7.3 New Feature Checklist
```
□ الميزة موجودة في PRD (Rule 1)
□ /ultra-think تم تشغيله
□ التصميم موثَّق في docs/claude/plans/
□ RLS محدَّث إذا لزم (Rule 3)
□ اختبارات مكتوبة
□ توثيق محدَّث
□ Security Review
```

---

## القرارات المعمارية المحفوظة
- CLAUDE.md واحد في الجذر — المرجع الوحيد الذي يُحدَّث دائماً
- نهج per-app CLAUDE.md تم التخلي عنه نهائياً
