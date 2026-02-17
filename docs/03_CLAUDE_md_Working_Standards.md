# ⚙️ دليل CLAUDE.md وقواعد العمل — MoradBot
**الإصدار:** 1.0.0 | **التاريخ:** فبراير 2026

---

## 📊 ملخص إحصائي

| المعيار | القيمة |
|---------|--------|
| **قرارات تتطلب ultra-think** | 8 قرارات |
| **قواعد لا تُخترق** | 6 قواعد |
| **أوامر بداية الجلسة** | 2 أوامر |
| **أوامر نهاية الجلسة** | 2 أوامر |
| **ملفات CLAUDE.md** | 5 ملفات (root + كل app) |
| **Hooks تلقائية نشطة** | 16 hook |

---

## 1. 📄 ملف CLAUDE.md الرئيسي (النص الكامل)

```markdown
# MoradBot — CLAUDE.md

## 🎯 هوية المشروع
- **الاسم:** MoradBot
- **النوع:** B2B SaaS — تطبيق سلة للرد التلقائي على FAQ
- **الجمهور:** تجار سعوديون (30-300 طلب/شهر)
- **اللغة الوحيدة:** عربي فقط في Chat Widget
- **التاريخ:** فبراير 2026

## 🛠️ التقنيات المستخدمة
| المكوّن | التقنية |
|---------|---------|
| Runtime | Cloudflare Workers (TypeScript) |
| Database/Auth | Supabase (PostgreSQL + RLS) |
| AI Provider | OpenRouter → Gemini 2.0 Flash |
| AI Fallback 1 | GPT-4 Mini |
| AI Fallback 2 | Claude 3.5 Sonnet |
| Chat Widget | Vanilla TypeScript → JS |
| Dashboard | React + Cloudflare Pages |
| Package Manager | pnpm + Turborepo |
| Testing | Vitest + Playwright |
| Linting | Biome |

## 🚫 قواعد لا تُخترق (6 قواعد)

### القاعدة 1 — نطاق MVP محدود للغاية
أي ميزة غير موجودة في docs/morad_bot_product_requirements_document_prd_v_1.md
تُرفض فوراً. لا استثناءات.

**ممنوع منعاً باتاً في MVP:**
- تتبع الطلبات
- WhatsApp integration
- لغة إنجليزية
- Analytics متقدمة
- رفع الملفات
- الرسائل الاستباقية

### القاعدة 2 — سلة قراءة فقط
MoradBot لا يُعدّل أي بيانات في سلة.
لا كتابة، لا حذف، لا تعديل.
فقط: GET Products.

### القاعدة 3 — عزل البيانات صفري التسامح
كل استعلام يجب أن يتضمن store_id.
لا استعلام بدون RLS Policy.
فشل عزل البيانات = إغلاق فوري للنظام.

### القاعدة 4 — الأسرار في Cloudflare Secrets فقط
ممنوع وضع أي secret في:
- الكود المصدري
- قاعدة البيانات
- ملفات .env في Production
- أي مكان غير Cloudflare Secrets

### القاعدة 5 — عربي فقط في Chat Widget
البوت يرد بالعربية فقط.
إذا كتب العميل بالإنجليزية → الرد بالعربية.
لا استثناء حتى لو طلب المستخدم الإنجليزية.

### القاعدة 6 — النشر يدوي دائماً
لا CI/CD للنشر التلقائي في Production.
كل نشر يمر بـ Pre-Deploy Checklist يدوياً.

## 📁 هيكل المشروع
```
moradbot/
├── .claude/          # Claude Code config (975+ components)
├── apps/
│   ├── api/          # Cloudflare Worker - Backend
│   ├── widget/       # Chat Widget (Vanilla TS)
│   └── dashboard/    # Merchant Dashboard (React)
├── packages/
│   ├── shared/       # TypeScript Types
│   ├── ai-orchestrator/
│   └── salla-client/
├── supabase/
│   └── migrations/
└── docs/             # المرجع الأساسي دائماً
```

## 📚 مصادر المعرفة (بالأولوية)
1. `docs/morad_bot_system_requirements_document_srd_v_1.md`
2. `docs/morad_bot_product_requirements_document_prd_v_1.md`
3. `docs/morad_bot_extended_architecture_document_v_1.md`
4. `Open_source_projects/adk-samples-main/typescript/agents/customer_service`

## ⚡ أوامر الجلسة

### بداية كل جلسة
```
/prime
/resume
```

### نهاية كل جلسة
```
/session-learning-capture
/update-docs
```

## 🔴 قرارات تتطلب /ultra-think أولاً
- أي تغيير في Schema
- أي تغيير في System Prompt للبوت
- أي ميزة جديدة (حتى صغيرة)
- أي قرار أمني
- أي تغيير في بنية المشروع
- أي تغيير في OpenRouter Integration
- أي تغيير في RLS Policies
- أي قرار يخص PDPL

## 📊 مؤشرات الأداء المستهدفة
| المؤشر | الهدف |
|--------|-------|
| Chat Reply P50 | ≤ 1.5 ثانية |
| Chat Reply P95 | ≤ 3.0 ثانية |
| Chat Timeout | 8 ثوان |
| Dashboard Load P95 | ≤ 2.5 ثانية |
| Product Sync (1,000) | < 60 ثانية |
| Uptime شهري | ≥ 99% |
| Test Coverage | ≥ 80% |

## 🔄 دورة التطوير القياسية
```
ultra-think → code → test → security-scan → document → commit
```
```

---

## 2. 📄 ملفات CLAUDE.md الفرعية (لكل App)

### 2.1 apps/api/CLAUDE.md
```markdown
# API Worker — CLAUDE.md

## التقنية
Cloudflare Worker بـ TypeScript + Hono Framework

## Endpoints المطلوبة
| Endpoint | Method | الوظيفة |
|----------|--------|---------|
| /auth/salla/start | GET | بدء OAuth |
| /auth/salla/callback | GET | استقبال code |
| /auth/salla/refresh | POST | تجديد token |
| /api/chat | POST | استقبال رسائل Widget |
| /api/faq | GET/POST | قراءة/تحديث FAQ |
| /api/stats | GET | إحصاءات Dashboard |
| /api/tickets | GET | قائمة المحادثات |
| /api/escalations | GET/PATCH | التصعيدات |

## قواعد هذا الـ Worker
- كل request يجب أن يحمل store_id صحيح
- Rate Limiting على كل endpoint
- Audit Log لكل عملية حساسة
- Timeout: 30 ثانية max
```

### 2.2 apps/widget/CLAUDE.md
```markdown
# Chat Widget — CLAUDE.md

## المتطلبات
- Vanilla TypeScript فقط (بدون Framework)
- حجم النتيجة: < 50KB مضغوط
- RTL عربي كامل
- لا يظهر على مسارات /checkout/*

## حالات Widget (7 حالات)
1. مغلق → أيقونة صغيرة
2. مفتوح → نافذة فارغة
3. يكتب → مؤشر ...
4. استجابة → رسالة الرد
5. تصعيد → نموذج البيانات
6. خطأ → رسالة احترامية
7. ممتلئ → رسالة انتهاء الحد

## قواعد هذا الـ Widget
- أول رسالة دائماً: إفصاح AI
- موافقة قبل أي تخزين
- لا cookies بدون موافقة
- 20 رسالة/دقيقة max للزائر
```

### 2.3 apps/dashboard/CLAUDE.md
```markdown
# Merchant Dashboard — CLAUDE.md

## المتطلبات
- React + Vite + TypeScript
- Supabase Auth للدخول
- P95 Load ≤ 2.5 ثانية

## الأقسام (4 أقسام)
1. الرئيسية → استخدام + زر تشغيل
2. المحادثات → قراءة فقط
3. التصعيدات → عرض + إغلاق
4. الإعدادات → FAQ + اشتراك

## قواعد هذا الـ Dashboard
- لا يعرض بيانات متاجر أخرى
- كل نداء API يحمل JWT صحيح
- Real-time للتصعيدات فقط
```

---

## 3. 🔄 بروتوكول جلسة العمل اليومي

### 3.1 بداية الجلسة (3 خطوات)

```bash
# الخطوة 1: تهيئة الجلسة
claude
>> /prime
# Claude يُحمّل CLAUDE.md ويستوعب السياق

# الخطوة 2: استئناف من حيث توقف
>> /resume
# Claude يُحدّد المرحلة الحالية ويقترح الخطوة التالية

# الخطوة 3: فحص صحة سريع (اختياري يومياً)
>> /project-health-check
```

### 3.2 أثناء العمل — قواعد التعامل

| الموقف | الفعل الصحيح |
|--------|-------------|
| قرار معماري جديد | `/ultra-think` أولاً |
| تغيير في Schema | `/ultra-think` + `/supabase-migration-assistant` |
| ميزة غير موجودة في PRD | رفض مباشر |
| خطأ غير مفهوم | `/debug-error` |
| كود جديد جاهز | `/code-review` + `/security-scanner` تلقائياً |
| Performance بطيء | `/optimize-api-performance` |

### 3.3 نهاية الجلسة (3 خطوات)

```bash
# الخطوة 1: تسجيل ما تعلمناه
>> /session-learning-capture
# Claude يُسجّل: قرارات المشكلات + الدروس + ما تبقى

# الخطوة 2: تحديث التوثيق
>> /update-docs
# تحديث أي وثيقة تغيّرت

# الخطوة 3: commit
git add .
git commit -m "feat(phase-X): وصف مختصر"
```

---

## 4. 🪝 خريطة Hooks التلقائية

```
┌─────────────────────────────────────────────────────┐
│                  دورة حياة الأمر                    │
│                                                     │
│  [pre-tool]                                         │
│  backup-before-edit ──────→ نسخ الملف أولاً        │
│  dependency-checker ──────→ فحص المكتبات           │
│  file-protection ─────────→ حماية الملفات الحرجة   │
│                                                     │
│  [تنفيذ الأمر]                                      │
│  ↕ Claude Code ينفّذ التغييرات                      │
│                                                     │
│  [post-tool]                                        │
│  security-scanner ────────→ مسح أمني فوري          │
│  change-tracker ──────────→ تسجيل التغيير          │
│  run-tests-after-changes ─→ تشغيل الاختبارات       │
│  build-on-change ─────────→ إعادة البناء           │
│  nextjs-code-quality ─────→ فحص جودة الكود         │
│                                                     │
│  [events]                                           │
│  telegram-error-notifications → عند أي خطأ         │
│  telegram-detailed-notifications → عند الأحداث     │
│  performance-monitor ─────→ مراقبة مستمرة         │
│                                                     │
│  [startup]                                          │
│  agents-md-loader ────────→ تحميل سياق الوكلاء    │
└─────────────────────────────────────────────────────┘
```

---

## 5. 🗺️ خريطة الـ Agents حسب المرحلة

| المرحلة | الـ Agents الرئيسية |
|---------|-------------------|
| 0 — الإعداد | dx-optimizer, technical-writer |
| 1 — الهيكل | backend-architect, typescript-pro |
| 2 — Supabase | sql-pro, database-optimization, security-auditor |
| 3 — OAuth | backend-architect, typescript-pro, legal-advisor |
| 4 — Widget | frontend-developer, ui-ux-designer, javascript-pro |
| 5 — AI Agent | ai-engineer, prompt-engineer, typescript-pro |
| 6 — Sync | data-engineer, backend-architect, typescript-pro |
| 7 — Dashboard | frontend-developer, ui-ux-designer, typescript-pro |
| 8 — Billing | backend-architect, data-engineer, payment-integration |
| 9 — Security | security-auditor, api-security-audit, load-testing-specialist |
| 10 — Deploy | incident-responder, technical-writer |

---

## 6. ⚠️ بروتوكول الطوارئ

### 6.1 الحوادث الحرجة (تتطلب إغلاق فوري)
```
حادثة أمنية → إغلاق فوري → تحقيق → إصلاح → إعادة فتح
```

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
# (إيميل يدوي أو عبر Dashboard)

# 4. تحقيق وإصلاح
# ...

# 5. إعادة التشغيل بعد التأكيد
```

---

## 7. 📋 Checklists الأساسية

### 7.1 Pre-Commit Checklist
```
□ الاختبارات تنجح (0 failures)
□ Biome لا يُظهر أخطاء
□ security-scanner لم يجد مشاكل
□ لا secrets مُضمَّنة في الكود
□ CLAUDE.md محدَّث إذا تغيّر الهيكل
```

### 7.2 Pre-Deploy Checklist
```
□ جميع الاختبارات تنجح
□ Test Coverage ≥ 80%
□ Security Audit: 0 Critical Issues
□ Supabase Migrations مُطبَّقة
□ Cloudflare Secrets مُعدَّة
□ Monitoring Alerts مُعدَّة
□ Rollback Plan موثَّق
□ موافقة على النشر
```

### 7.3 New Feature Checklist
```
□ الميزة موجودة في PRD
□ /ultra-think تم تشغيله
□ التصميم موثَّق
□ RLS محدَّث إذا لزم
□ اختبارات مكتوبة
□ توثيق محدَّث
□ Security Review
```

---

*آخر تحديث: فبراير 2026 | المشروع: MoradBot SaaS*
