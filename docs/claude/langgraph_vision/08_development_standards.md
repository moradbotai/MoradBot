# مراد بوت — دليل قواعد العمل
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.1 (محدَّث أبريل 2026 — يشمل قواعد LangGraph)
**الحالة:** معتمد

---

## 1. ملف CLAUDE.md الرئيسي

المرجع الوحيد المحدَّث دائماً: `/Users/mohammedaljohani/Documents/Proj/moradbot/CLAUDE.md`

---

## 2. بروتوكول جلسة العمل اليومي

### 2.1 بداية الجلسة
```bash
/prime   # Claude يُحمّل CLAUDE.md ويستوعب السياق
/resume  # Claude يُحدّد المرحلة الحالية
```

### 2.2 أثناء العمل

| الموقف | الفعل الصحيح |
|--------|-------------|
| قرار معماري جديد | `/ultra-think` أولاً |
| تغيير في Schema | `/ultra-think` + أوامر supabase |
| ميزة غير في PRD | رفض مباشر (Rule 1) |
| خطأ غير مفهوم | `/debug-error` |
| كود جديد جاهز | `/code-review` |
| قرار LangGraph | `/ultra-think` إلزامي |

### 2.3 نهاية الجلسة
```bash
/session-learning-capture  # تسجيل ما تعلمناه
/update-docs               # تحديث التوثيق
git add [الملفات المحددة]
git commit -m "feat(phase-X): وصف مختصر"
```

---

## 3. قواعد LangGraph (مضافة أبريل 2026)

### 3.1 هيكل Python Service
```
packages/ai-orchestrator/
├── pyproject.toml          # Python dependencies
├── src/
│   ├── graphs/             # StateGraph definitions
│   ├── nodes/              # Node functions
│   ├── state.py            # TypedDict State
│   ├── config.py           # Merchant config
│   └── server.py           # FastAPI
└── tests/
```

### 3.2 قواعد الـ State
- كل حقل جديد في `MoradBotState` يحتاج `ultra-think` أولاً.
- `store_id` في الـ State إلزامي دائماً (Rule 3).
- Checkpointing يعمل دائماً عبر Supabase PostgresSaver.

### 3.3 قواعد Nodes
- كل Node يتلقى State ويُعيد dict جزئي.
- `validate_input` دائماً أول node (Rule: Prompt Injection).
- لا LLM call بدون المرور بـ `validate_input`.

### 3.4 الاتصال بين Hono وLangGraph
- Workers → LangGraph: `POST /chat` مع JWT للمصادقة.
- Timeout في Workers: 8 ثوانٍ hard limit.
- LangGraph يُعيد: `{ bot_response, escalated, ticket_id }`.

---

## 4. Hooks النشطة الفعلية (8 hooks في 3 أحداث)

> المصدر: `.claude/settings.json` — تحقق أبريل 2026

### PreToolUse (قبل تنفيذ الأدوات)

| الـ Hook | الوظيفة |
|---------|---------|
| backup-before-edit | يُنسخ الملف لـ `.backups/` قبل أي تعديل |
| file-protection | يمنع تعديل الملفات المحمية (patterns محددة) |

### PostToolUse (بعد تنفيذ الأدوات)

| الـ Hook | الوظيفة |
|---------|---------|
| change-tracker (modified) | يسجل تعديل الملفات في `~/.claude/changes.log` |
| change-tracker (created) | يسجل إنشاء الملفات الجديدة |
| security-scanner | يُشغّل semgrep تلقائياً على كل ملف مُعدَّل |
| gsd-context-monitor | يراقب السياق عبر `.claude/hooks/gsd-context-monitor.js` |

### SessionStart (عند بدء الجلسة)

| الـ Hook | الوظيفة |
|---------|---------|
| agents-md-loader | يُحمّل `AGENTS.md` كـ context تلقائياً |
| gsd-check-update | يتحقق من تحديثات GSD عبر `.claude/hooks/gsd-check-update.js` |

---

## 5. خريطة الـ Agents النشطين

| الـ Agent | الدور | المراحل |
|----------|-------|---------|
| `backend-architect` | تصميم البنية والـ Middleware | Phase 3، 4، 7 |
| `typescript-pro` | TypeScript patterns | جميع المراحل |
| `database-optimizer` | Schema، RLS، Migration | Phase 2 |
| `security-auditor` | Security review، PDPL | Phase 8 |
| `ai-engineer` | LangGraph + OpenRouter + LangSmith | Phase 5 |
| `error-detective` | Debugging | مستمر |
| `api-documenter` | توثيق endpoints | Phase 3، 4 |
| `prompt-engineer` | System Prompt للبوت | Phase 5 |
| `technical-writer` | توثيق المشروع | Phase 9 |

---

## 6. بروتوكول الطوارئ

| نوع الحادثة | الإجراء الفوري |
|------------|--------------|
| تسرّب بيانات متاجر | إغلاق Chat Widget فوراً |
| اختراق قاعدة البيانات | إيقاف جميع الـ Workers |
| LangGraph يُرجع بيانات خاطئة | تعطيل AI Service + escalate manually |
| استخدام غير مصرح به للـ tokens | إلغاء جميع الـ tokens + إعادة توليد |

---

## 7. Checklists الأساسية

### Pre-Commit Checklist
```
□ الاختبارات تنجح (0 failures)
□ Biome لا يُظهر أخطاء
□ security-scanner لم يجد مشاكل
□ لا secrets مُضمَّنة
□ CLAUDE.md محدَّث إذا تغيّر الهيكل
```

### Pre-Deploy Checklist
```
□ جميع الاختبارات تنجح
□ Test Coverage ≥ 80%
□ Security Audit: 0 Critical Issues
□ Supabase Migrations مُطبَّقة
□ Cloudflare Secrets مُعدَّة
□ LangGraph Service نشط وصحي
□ LangSmith traces تعمل
□ Rollback Plan موثَّق
```

### New Feature Checklist
```
□ الميزة موجودة في PRD (Rule 1)
□ /ultra-think تم تشغيله
□ التصميم موثَّق في docs/claude/plans/
□ RLS محدَّث إذا لزم (Rule 3)
□ اختبارات مكتوبة
□ توثيق محدَّث
□ Security Review
□ [للميزات AI] LangGraph graph design مُوثَّق
```

---

## القرارات المعمارية المحفوظة
- CLAUDE.md واحد في الجذر — المرجع الوحيد
- نهج per-app CLAUDE.md تم التخلي عنه نهائياً
- **قواعد LangGraph مُضافة أبريل 2026 (validate_input إلزامي، store_id في State)**
