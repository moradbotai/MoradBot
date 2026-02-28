# مراد بوت — دليل التوثيق
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.0
**الحالة:** معتمد
تاريخ الإنشاء: فبراير 2026

---

## الهدف
دليل ملاحة مرجعي لجميع وثائق مشروع مراد بوت — الوثائق الأصلية المحمية في `docs/` والوثائق المحدَّثة في `docs/claude/v2/`.

## النطاق
✅ يغطي:
- قائمة الوثائق الأصلية السبع المحمية
- قائمة وثائق Claude المولَّدة وأماكنها
- دليل القراءة حسب الدور
- حالة المراحل التطويرية

❌ لا يغطي:
- توثيق الكود — يوجد في `CLAUDE.md`
- قرارات المعمارية التفصيلية — يوجد في `extended_architecture_v2.md`

---

## الوثائق الأصلية المحمية

هذه الوثائق السبع في `docs/` **محمية ولا تُعدَّل أبداً** (Rule 7). تمثل القرارات التأسيسية للمشروع.

| # | الملف | المحتوى |
|---|-------|---------|
| 1 | `morad_bot_market_requirements_document_mrd_v_1.md` | MRD — تعريف السوق والمشكلة |
| 2 | `morad_bot_business_requirements_document_brd_v_1.md` | BRD — النموذج التجاري والإيرادات |
| 3 | `morad_bot_product_requirements_document_prd_v_1.md` | PRD — متطلبات المنتج وتعريف MVP |
| 4 | `morad_bot_non_functional_requirements_nfr_v_1.md` | NFR — المتطلبات غير الوظيفية |
| 5 | `morad_bot_system_requirements_document_srd_v_1.md` | SRD — متطلبات النظام والمعمارية |
| 6 | `morad_bot_full_project_documentation_v_1.md` | ملخص تنفيذي شامل |
| 7 | `morad_bot_extended_architecture_document_v_1.md` | قرارات المعمارية الموسعة (v1) |

---

## وثائق Claude المولَّدة

جميع الوثائق التي تُولَّد بواسطة Claude تذهب إلى `docs/claude/` فقط — **لا في `docs/` الجذر** (Rule 7).

### docs/claude/ — الوثائق العامة
| الملف | المحتوى | الحالة |
|-------|---------|--------|
| `tools_report_v2.md` | تقرير مراجعة النظام البيئي لـ Claude (فبراير 2026) | نشط |
| `salla_api_reference.md` | مرجع Salla API: OAuth، المنتجات، الأخطاء، حدود المعدلات | نشط |
| `session-2026-02-18_summary.md` | ملخص جلسة العمل (18 فبراير 2026) | مؤرشف |

> ملاحظة: `01_Tools_Report.md` الأصلي أُرشف — النسخة الحالية هي `tools_report_v2.md`. الملفات `02_` و`03_` نُقلت إلى `docs/claude/`.

### docs/claude/plans/ — خطط المراحل
| الملف | المرحلة |
|-------|---------|
| `phase-01_dev-environment.md` | Phase 1: بيئة التطوير |
| `phase-02_database.md` | Phase 2: قاعدة البيانات |
| `phase-03_api-foundation.md` | Phase 3: أساس API |

### docs/claude/v2/ — الوثائق المحدَّثة (النسخة الثانية)
| الملف | يعدّل | ما الجديد |
|-------|-------|-----------|
| `extended_architecture_v2.md` | `docs/morad_bot_extended_architecture_document_v_1.md` | قيم Rate Limiting، إزالة Multi-Agent، المتمكّن بدل Higher، RPO/RTO، Worker timeout، CORS |
| `readme_v2.md` | `docs/readme_morad_bot_documentation.md` | هذا الملف نفسه |

---

## دليل القراءة حسب الدور

### الأعمال والاستراتيجية
ابدأ بـ:
1. `docs/morad_bot_market_requirements_document_mrd_v_1.md`
2. `docs/morad_bot_business_requirements_document_brd_v_1.md`

### المنتج
ابدأ بـ:
1. `docs/morad_bot_product_requirements_document_prd_v_1.md`
2. `docs/morad_bot_system_requirements_document_srd_v_1.md`

### الأمان والموثوقية
ابدأ بـ:
1. `docs/morad_bot_non_functional_requirements_nfr_v_1.md`
2. `docs/morad_bot_system_requirements_document_srd_v_1.md`
3. `docs/claude/v2/extended_architecture_v2.md`

### الهندسة والتطوير
اقرأ بالترتيب:
1. `CLAUDE.md` — قواعد العمل والأوامر وحالة المراحل
2. `docs/morad_bot_product_requirements_document_prd_v_1.md`
3. `docs/morad_bot_non_functional_requirements_nfr_v_1.md`
4. `docs/morad_bot_system_requirements_document_srd_v_1.md`
5. `docs/claude/v2/extended_architecture_v2.md`

---

## حالة المراحل التطويرية

| المرحلة | الحالة | ما تم بناؤه |
|---------|--------|------------|
| Phase 1 | مكتملة | بيئة التطوير، Turborepo، Biome، الحزم الأساسية |
| Phase 2 | مكتملة | 12 جدولاً، 5 migrations، سياسات RLS، TypeScript types |
| Phase 3 | مكتملة | Hono API: 16 نقطة نهاية، middleware stack، هيكل الأخطاء، Supabase clients |
| Phase 4 | قادمة | Salla Client Package (OAuth + `GET /products`) |
| Phase 5+ | معلقة | AI Orchestrator، Widget UI، Dashboard UI |

---

## قاعدة سلامة القرارات

جميع الوثائق تمثل قرارات صريحة. لا يجوز تنفيذ أي ميزة تتعارض مع:
- نطاق PRD
- قيود NFR
- معمارية SRD

التغييرات تستوجب مراجعة رسمية وزيادة رقم الإصدار.

---

## القرارات المعمارية المحفوظة
- جميع وثائق Claude تذهب إلى `docs/claude/` فقط (Rule 7)
- الوثائق الأصلية في `docs/` محمية ولا تُعدَّل
- النسخ v2 تعيش في `docs/claude/v2/` ولا تحل محل الأصليات
