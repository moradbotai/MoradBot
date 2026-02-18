# MoradBot — CLAUDE.md

## 🎯 هوية المشروع
- **الاسم:** MoradBot
- **النوع:** B2B SaaS — تطبيق سلة للرد التلقائي على FAQ
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

## 🚫 قواعد لا تُخترق (8 قواعد)

### القاعدة 1 — نطاق MVP محدود للغاية
أي ميزة غير موجودة في docs/morad_bot_product_requirements_document_prd_v_1.md
تُرفض فوراً. لا استثناءات.

**ممنوع منعاً باتاً في MVP:**
- تتبع الطلبات
- WhatsApp integration
- لغة إنجليزية
- Analytics متقدمة
- رفع الملفات
- الرسائل الاستباقية

### القاعدة 2 — سلة قراءة فقط
MoradBot لا يُعدّل أي بيانات في سلة.
لا كتابة، لا حذف، لا تعديل.
فقط: GET Products.

### القاعدة 3 — عزل البيانات صفري التسامح
كل استعلام يجب أن يتضمن store_id.
لا استعلام بدون RLS Policy.
فشل عزل البيانات = إغلاق فوري للنظام.

### القاعدة 4 — الأسرار في Cloudflare Secrets فقط
ممنوع وضع أي secret في:
- الكود المصدري
- قاعدة البيانات
- ملفات .env في Production
- أي مكان غير Cloudflare Secrets

### القاعدة 5 — عربي فقط في Chat Widget
البوت يرد بالعربية فقط.
إذا كتب العميل بالإنجليزية → الرد بالعربية.
لا استثناء حتى لو طلب المستخدم الإنجليزية.

### القاعدة 6 — النشر يدوي دائماً
لا CI/CD للنشر التلقائي في Production.
كل نشر يمر بـ Pre-Deploy Checklist يدوياً.

## 📁 هيكل المشروع
moradbot/
├── .claude/ # Claude Code config (975+ components)
├── apps/
│ ├── api/ # Cloudflare Worker - Backend
│ ├── widget/ # Chat Widget (Vanilla TS)
│ └── dashboard/ # Merchant Dashboard (React)
├── packages/
│ ├── shared/ # TypeScript Types
│ ├── ai-orchestrator/
│ └── salla-client/
├── supabase/
│ └── migrations/
└── docs/ # المرجع الأساسي دائماً


## 📚 مصادر المعرفة (بالأولوية)
1. `docs/morad_bot_system_requirements_document_srd_v_1.md`
2. `docs/morad_bot_product_requirements_document_prd_v_1.md`
3. `docs/morad_bot_extended_architecture_document_v_1.md`
4. `Open_source_projects/adk-samples-main/typescript/agents/customer_service`
5. `docs/claude/salla_api_reference.md` — مرجع Salla API (OAuth, Products, Errors, Rate Limits)

## ⚡ أوامر الجلسة

### بداية كل جلسة
/prime
/resume


### نهاية كل جلسة
/session-learning-capture
/update-docs


## 🔴 قرارات تتطلب /ultra-think أولاً
- أي تغيير في Schema
- أي تغيير في System Prompt للبوت
- أي ميزة جديدة (حتى صغيرة)
- أي قرار أمني
- أي تغيير في بنية المشروع
- أي تغيير في OpenRouter Integration
- أي تغيير في RLS Policies
- أي قرار يخص PDPL

## 📊 مؤشرات الأداء المستهدفة
| المؤشر | الهدف |
|--------|-------|
| Chat Reply P50 | ≤ 1.5 ثانية |
| Chat Reply P95 | ≤ 3.0 ثانية |
| Chat Timeout | 8 ثوان |
| Dashboard Load P95 | ≤ 2.5 ثانية |
| Product Sync (1,000) | < 60 ثانية |
| Uptime شهري | ≥ 99% |
| Test Coverage | ≥ 80% |

## 🔄 دورة التطوير القياسية
ultra-think → code → test → security-scan → document → commit
### القاعدة 7 — تنظيم الوثائق  
**جميع الوثائق التي ينشئها Claude تذهب إلى `docs/claude/` فقط.**

- ✅ `docs/claude/` - كل ما ينشئه Claude
- ✅ `docs/claude/plans/` - خطط التنفيذ (phase-XX_اسم.md)
- ❌ `docs/` - الوثائق الأصلية فقط (لا تُمس، لا تُعدّل)

**الوثائق الأصلية في `docs/` محمية:**
- `morad_bot_system_requirements_document_srd_v_1.md`
- `morad_bot_product_requirements_document_prd_v_1.md`
- `morad_bot_extended_architecture_document_v_1.md`
- وجميع المستندات الأخرى

### القاعدة 8 — خطط التنفيذ
كل مرحلة تطوير لها ملف خطة خاص في `docs/claude/plans/`.

**تنسيق الاسم:** `phase-XX_اسم_المرحلة.md`

**مثال:**
- `docs/claude/plans/phase-01_environment.md`
- `docs/claude/plans/phase-02_database.md`
- `docs/claude/plans/phase-03_api_foundation.md`

