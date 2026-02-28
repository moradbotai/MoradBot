# Documentation Action Plan — MoradBot

آخر تحديث: 22 فبراير 2026
الأولوية: من الأكثر إلحاحاً إلى الأقل

---

## الأولوية 1 — فوري (خطر أو حاجب لـ Phase 4)

### الإجراء 1.1: أرشفة `docs/01_Tools_Report.md` ⚫

**السبب:** الوثيقة خطيرة — أمر التثبيت الشامل يُثبّت 975 مكوناً معظمها مرفوض. أي مطور يرجع إليها سيؤمن بوجود 31 agent و20 MCP server بينما الواقع 9 و5.

**الإجراء المحدد:**
```bash
mv docs/01_Tools_Report.md .claude_archive/not_useful/
```
أضف هذا البانر في أعلى الملف قبل النقل:
```
⚠️ SUPERSEDED — هذه الوثيقة تعكس تخطيطاً قبل التنفيذ.
معظم الأدوات المُدرجة لم تُثبَّت.
المرجع الصحيح: docs/claude/tools_report_v2.md
```

**الجهد:** منخفض (5 دقائق) | **التأثير:** يزيل محتوى خطيراً

---

### الإجراء 1.2: نقل `docs/02_Implementation_Plan.md` إلى `docs/claude/plans/` 🟠

**السبب:** انتهاك Rule 7. الوثيقة قيّمة (تحتوي معايير اكتمال المراحل المستقبلية) لكنها في المكان الخطأ.

**الإجراء المحدد:**
```bash
mv docs/02_Implementation_Plan.md docs/claude/plans/implementation-plan-original.md
```
أضف بانر في أعلى الملف:
```
⚠️ PARTIAL REFERENCE — Phase 0 متقادمة، Phases 1-2 مكتملة مع انحرافات.
الانحرافات: 12 جدولاً (لا 10)، Preact+Vite (لا Vanilla TS)، Next.js 15 (لا React+Vite).
مرحلة API Foundation (Phase 3 الفعلية) غائبة كلياً من هذه الخطة.
```

**الجهد:** منخفض (10 دقائق) | **التأثير:** يُصحح انتهاك Rule 7

---

### الإجراء 1.3: نقل `docs/03_CLAUDE_md_Working_Standards.md` إلى `docs/claude/` 🟠

**السبب:** انتهاك Rule 7. قوائم المراجعة وبروتوكول الطوارئ (Sections 6-7) ذات قيمة حقيقية وغير مكررة في مكان آخر.

**الإجراء المحدد:**
```bash
mv docs/03_CLAUDE_md_Working_Standards.md docs/claude/working-standards.md
```

**الجهد:** منخفض (5 دقائق) | **التأثير:** يُصحح انتهاك Rule 7

---

### الإجراء 1.4: إنشاء `docs/claude/plans/phase-04_salla-client.md`

**السبب:** Rule 8 + Phase 4 بدأت. لا توجد خطة مرحلة محددة.

**المحتوى المطلوب:**
- الهدف: Salla OAuth (Custom Mode) + GET /products + cron sync
- تفاصيل SRD 2.1 و2.5 مع الإضافات المحددة:
  - تخزين tokens مشفر (AES-256-GCM) في `stores.access_token`
  - Cloudflare KV لـ rate limiting (`RATE_LIMIT_KV`)
  - قرار multi-store: تأكيد أن UNIQUE constraint يعني متجر واحد لكل تاجر في MVP
  - Salla rate limits من `docs/claude/salla_api_reference.md`
- معايير الاكتمال القابلة للقياس

**الجهد:** متوسط (1-2 ساعة) | **التأثير:** يُمكّن Phase 4 بأمان

---

## الأولوية 2 — مهم (قبل Phase 5 أو أول إطلاق)

### الإجراء 2.1: إصلاح `morad_bot_full_project_documentation_v_1.md` 🟠

**تنبيه:** هذه وثيقة أصلية محمية بموجب Rule 7. **لا تُعدَّل مباشرة.** الإصلاحات تُوثَّق في `docs/claude/` كملاحق.

**الإنحرافات الواجب توثيقها في ملحق:**

| القسم | المشكلة | القيمة الصحيحة |
|-------|---------|----------------|
| القسم 1 | ICP: "10-500 طلب/شهر" ❌ | 30-300 طلب/شهر |
| القسم 7 | Infrastructure: لا ذكر لـ Hono/Preact/Next.js 15/Biome/Turborepo | أضف الـ stack المُنفَّذ |
| القسم 9 | Branding/white-label كـ MVP feature | صنّفها: "مؤجلة — ما بعد MVP" |
| القسم 4 | نموذج الأعمال: مثال واحد فقط | أضف: Basic (500/99 SAR)، Mid (2000/299 SAR)، Premium (10000/799 SAR) |

**أنشئ:** `docs/claude/full-project-doc-corrections.md` يوثّق الانحرافات.

---

### الإجراء 2.2: إضافة ملحق لـ PRD — متطلبات Widget المفقودة

**تنبيه:** الوثيقة الأصلية محمية. الإضافات في `docs/claude/`.

**أنشئ:** `docs/claude/plans/phase-05_widget-supplement.md`

يتضمن المتطلبات الوظيفية الغائبة من PRD:
1. إفصاح AI في الرسالة الأولى (نص عربي محدد)
2. تدفق الموافقة الصريحة قبل تخزين بيانات الزائر (PDPL Rule)
3. سلوك اللغة العربية (الرد بالعربية حتى لو الزائر كتب بالإنجليزية — Rule 5)
4. حالة `error` — تعرض زر إعادة المحاولة
5. حالة `limit-reached` — تعطيل الإدخال بشكل دائم عند code 4293
6. حالة `bot_disabled` — تعرض معلومات التواصل عند codes 4401/4402
7. الـ Widget bundle لا يتجاوز 50KB gzipped
8. لا يُعرض على مسارات `/checkout/*`

**الجهد:** منخفض (1 ساعة) | **التأثير:** يمنع ثغرات Phase 5

---

### الإجراء 2.3: إضافة ملحق لـ SRD — Phase 5 AI Orchestrator

**أنشئ:** `docs/claude/plans/phase-05_ai-orchestrator-spec.md`

يحدد ما هو غائب من SRD Section 2.6:
- Provider: OpenRouter → Gemini 2.0 Flash (primary)
- Fallback: GPT-4 Mini → Claude 3.5 Sonnet
- Timeout صارم: 8 ثوانٍ لكل طلب
- سلوك rate limiting على مستوى AI: KV-based enforcement
- مراقبة تكلفة tokens: > 200% baseline = تنبيه (من SRD 2.11)
- آلية system prompt confidentiality (من CLAUDE.md — LLM Security Policy)

**الجهد:** منخفض-متوسط (1-2 ساعة)

---

### الإجراء 2.4: توثيق انحرافات SRD في ملحق

**أنشئ:** `docs/claude/srd-corrections.md`

يوثّق:
1. القسم 2.4: FAQ CRUD مُنفَّذ بالفعل في Phase 3 (لم يعد "future/optional")
2. القسم 2.1: UNIQUE constraint يعني متجر واحد لكل تاجر في MVP (قرار multi-store مؤجل)
3. القسم 2.5: "Higher"/"Highest" = "Mid"/"Premium" في الكود والـ schema
4. القسم 4.4: آلية rate limiting = Cloudflare KV (ليس in-memory، ليس Durable Objects)
5. القسم 2.11: Telegram alerts مؤجلة — email-only للـ MVP (Resend)

---

### الإجراء 2.5: تحديث `docs/claude/plans/implementation-plan-original.md` بملاحظات

بعد نقل الملف في الإجراء 1.2، أضف ملاحظات مضمّنة للانحرافات:
- Phase 0 (tooling): OBSOLETE → راجع tools_report_v2.md
- Phase 1 (environment): مكتمل — مع Biome بدلاً من ESLint + Prettier
- Phase 2 (DB): مكتمل — 12 جدولاً لا 10، naming conventions مختلفة
- Phase 3 (Salla OAuth في الخطة الأصلية): تم تحديثه ليكون Phase 4 الفعلية
- Phase 3 المُضاف (API Foundation): لم تكن في الخطة الأصلية — 16 endpoint، Hono v4
- Widget: Preact + Vite (لا Vanilla TypeScript)
- Dashboard: Next.js 15 App Router (لا React + Vite)

---

## الأولوية 3 — منخفضة (تحسينات غير عاجلة)

### الإجراء 3.1: تحديث `docs/readme_morad_bot_documentation.md` 🟡

**تنبيه:** هذه borderline case — الـ README كُتب بواسطة Claude لكنه يخدم `docs/` الأصلية. لا تعدّله مباشرة — أنشئ نسخة محدثة في `docs/claude/`.

**أو:** إذا كان المستخدم يملك الملف الأصلي، يمكن تعديله مع التوثيق.

التحديثات المطلوبة:
- أضف تاريخ إنشاء/تحقق
- أضف ملاحظة: "وثائق Claude في `docs/claude/` — ليس هنا"
- حدّث قسم "Current Phase": Phases 1-3 مكتملة، Phase 4 قادمة
- أضف ملاحظة عن الوثائق الثلاث Claude-generated الموجودة خطأً

---

### الإجراء 3.2: توثيق BRD في ملحق 🟡

**أنشئ:** ملاحظة قصيرة في `docs/claude/` تلاحظ:
1. ICP range: القسم 1 يذكر "30-150"؛ القسم 4 يذكر "30-300"؛ المرجع: 30-300
2. الأسعار: 99/299/799 SAR (Basic/Mid/Premium) غائبة من BRD

**الجهد:** منخفض جداً (15 دقيقة)

---

### الإجراء 3.3: تحديث `docs/claude/working-standards.md` بعد النقل 🟠

بعد تنفيذ الإجراء 1.3، أعد كتابة الأقسام التالية في الملف المنقول:

| القسم | الإجراء |
|-------|---------|
| القسم 1 | أضف Rules 7+8، صحّح تقنيات Widget/Dashboard، أضف 5th knowledge source — أو استبدل بـ "راجع root CLAUDE.md" |
| القسم 2 | استبدل بـ "تم التخلي عن نهج per-app CLAUDE.md. CLAUDE.md الجذر هو المعيار الوحيد" |
| القسم 4 | استبدل مخطط الـ 16-hook بمخطط الـ 5-hooks الفعلية |
| القسم 5 | فعّل الـ agents النشطة فقط؛ أرجع إلى tools_report_v2.md |
| Sections 6-7 | احتفظ بها كما هي — بروتوكول الطوارئ وقوائم المراجعة دقيقة وقيّمة |

**الجهد:** متوسط (2-3 ساعات)

---

## ملخص الإجراءات حسب الترتيب الزمني

```
Phase 4 (فوري — قبل البدء):
  1.1 → أرشف 01_Tools_Report.md          (5 دقائق)
  1.2 → انقل 02_Implementation_Plan.md   (10 دقائق)
  1.3 → انقل 03_CLAUDE_md_Working_Standards.md  (5 دقائق)
  1.4 → أنشئ phase-04_salla-client.md    (1-2 ساعة) ★ الأهم

قبل Phase 5 (مهم):
  2.1 → وثّق تصحيحات Full Project Doc   (30 دقيقة)
  2.2 → أنشئ phase-05_widget-supplement  (1 ساعة)
  2.3 → أنشئ phase-05_ai-orchestrator-spec (1-2 ساعة)
  2.4 → وثّق انحرافات SRD               (30 دقيقة)
  2.5 → حدّث implementation-plan-original بملاحظات (30 دقيقة)

لاحقاً (منخفضة الأولوية):
  3.1 → حدّث README                     (30 دقيقة)
  3.2 → لاحظ انحرافات BRD              (15 دقيقة)
  3.3 → حدّث working-standards.md       (2-3 ساعات)
```
