# Documentation Classification — MoradBot

آخر تحديث: 22 فبراير 2026

---

## نظام التصنيف

| اللون | المعنى |
|-------|--------|
| 🟢 GREEN | دقيق، مكتمل، محدّث — لا يحتاج إجراء |
| 🟡 YELLOW | صالح مع ثغرات محددة — يحتاج تحديثات بسيطة |
| 🟠 ORANGE | قيّم جزئياً مع مشاكل جوهرية — يحتاج إعادة كتابة مستهدفة |
| ⚫ BLACK | متقادم كلياً أو خطير — أرشفة أو حذف فوري |

---

## جدول ملخص التصنيف

| # | الوثيقة | التصنيف | الملخص بجملة واحدة |
|---|---------|---------|-------------------|
| 1 | morad_bot_market_requirements_document_mrd_v_1.md | 🟢 GREEN | تحليل سوق دقيق وصادق؛ مشكلة واحدة فقط (سعر 49 SAR غير موجود) |
| 2 | morad_bot_business_requirements_document_brd_v_1.md | 🟡 YELLOW | سليم تجارياً؛ تعارض ICP (150 مقابل 300) وغياب أسعار الخطط |
| 3 | morad_bot_extended_architecture_document_v_1.md | 🟡 YELLOW | قرارات تشغيلية سليمة؛ "Higher" مقابل "Premium" + قيم محددة مفقودة |
| 4 | morad_bot_product_requirements_document_prd_v_1.md | 🟡 YELLOW | النطاق محدد بدقة؛ متطلبات وظيفية للـ widget مفقودة (إفصاح AI، موافقة، أخطاء) |
| 5 | morad_bot_system_requirements_document_srd_v_1.md | 🟡 YELLOW | شامل وسليم تقنياً؛ تخلف عن Phase 3 + مواصفة AI provider مفقودة |
| 6 | readme_morad_bot_documentation.md | 🟡 YELLOW | دليل تنقل صالح؛ يحتاج تاريخاً، مؤشراً لـ docs/claude/، وتحديث المرحلة |
| 7 | morad_bot_full_project_documentation_v_1.md | 🟠 ORANGE | خطأ ICP فعلي (10-500 بدلاً من 30-300) + Infrastructure قديم + branding خاطئ |
| 8 | 03_CLAUDE_md_Working_Standards.md | 🟠 ORANGE | قوائم المراجعة وبروتوكول الطوارئ ذات قيمة؛ ~40% متقادم (Sections 2 و4) |
| 9 | 02_Implementation_Plan.md | 🟠 ORANGE | خارطة طريق المراحل المستقبلية قيّمة؛ تقنيات خاطئة + أرقام مراحل + مرحلة API مفقودة |
| 10 | 01_Tools_Report.md | ⚫ BLACK | متجاوز كلياً بـ tools_report_v2.md؛ أمر تثبيت خطير (975 مكوناً)؛ أرشفة فورية |

---

## التصنيفات المفصّلة

### 🟢 GREEN — morad_bot_market_requirements_document_mrd_v_1.md

**المبرر:** وثيقة MRD دقيقة وصادقة وجيدة البنية وما زالت صالحة كإطار لتحليل السوق. مشكلة واحدة فقط: الحد الأدنى "49 SAR" المذكور في Risk #3 لا يقابل أي خطة مُنفَّذة (الخطة الأساسية = 99 SAR). هذا غير دقيق ولكنه لا يؤثر على قرارات هندسية. لا يُعدّل الأصل بموجب Rule 7.

**نقاط القوة:** أمانة نادرة في التقييم النقدي، تحليل منافسين واقعي، قسم "Not the ICP" مفيد لتطبيق النطاق.

---

### 🟡 YELLOW — morad_bot_business_requirements_document_brd_v_1.md

**المبرر:** الـ BRD مكتوب باحترافية وما زال صالحاً كأساس تجاري. مشكلتان تمنعان GREEN:
1. **تعارض ICP:** القسم 1 يذكر 30-150 طلب/شهر؛ القسم 4 KPIs يذكر 30-300. القيمة الموثوقة في CLAUDE.md هي 30-300.
2. **أسعار الخطط غائبة:** 99/299/799 SAR موجودة في قاعدة البيانات ولكن ليس في الوثيقة المرجعية التجارية.

---

### 🟡 YELLOW — morad_bot_extended_architecture_document_v_1.md

**المبرر:** القرارات المعمارية سليمة وصحيحة في معظمها، لكن بمشكلتين محددتين:
1. **"Higher Plan" بدلاً من "Premium"** — تعارض مع schema قاعدة البيانات و CLAUDE.md. يجب حله قبل Phase 4.
2. **قيم غائبة:** حدود rate limiting الفعلية (20 رسالة/دقيقة للزائر، 3000/ساعة للمتجر)، تفاصيل CORS، RPO/RTO — موجودة في الكود ولكن غير موثقة هنا.

---

### 🟡 YELLOW — morad_bot_product_requirements_document_prd_v_1.md

**المبرر:** الـ PRD صحيح وجيد الهيكل ولكنه يفتقر لعدة متطلبات وظيفية حرجة لـ Phase 5:
- إفصاح AI في الرسالة الأولى
- تدفق الموافقة الصريحة قبل تخزين بيانات الزائر
- سلوك اللغة العربية (الرد بالعربية حتى لو الزائر كتب بالإنجليزية)
- حالات الـ widget: `error` و `limit-reached`
- غموض "Custom FAQ field" (المخطط ينفّذ enum `general`، ليس حقلاً حراً)

---

### 🟡 YELLOW — morad_bot_system_requirements_document_srd_v_1.md

**المبرر:** الـ SRD شامل وسليم تقنياً ولكنه تخلف عن التنفيذ في نقاط:
- القسم 2.4: FAQ editing كـ "future/optional" — لكنه مُنفَّذ بالفعل في Phase 3
- القسم 2.6: لا ذكر لـ OpenRouter أو Gemini 2.0 Flash أو سلسلة الـ fallback
- القسم 2.1: ادعاء multi-store لا يدعمه UNIQUE constraint في `salla_merchant_id`
- القسم 2.5: "Higher"/"Highest" بدلاً من "Mid"/"Premium"
- القسم 4.4: آلية rate limiting (KV؟ Durable Objects؟) غير محددة لـ stateless Workers
- القسم 2.11: Telegram alerts كمتطلب نشط بدون خطة تنفيذ

---

### 🟡 YELLOW — readme_morad_bot_documentation.md

**المبرر:** دليل التنقل صحيح ومفيد، لكن:
- لا تاريخ إنشاء أو تحقق
- لا مؤشر لـ `docs/claude/` كموطن للوثائق التي ينشئها Claude
- قسم "Current Phase" لا يعكس أن Phases 1-3 مكتملة
- لا ذكر للوثائق الثلاث Claude-generated الموجودة خطأً في `docs/`

---

### 🟠 ORANGE — morad_bot_full_project_documentation_v_1.md

**المبرر:** الوثيقة تحتوي خطأً واقعياً ذا تأثير عالٍ يتناقض مع جميع الوثائق الأخرى:
1. **خطأ ICP:** القسم 1 يذكر "10-500 طلب/شهر" — هذه القيمة غير موجودة في أي وثيقة أخرى. القيمة الصحيحة 30-300.
2. **Infrastructure قديم:** القسم 7 لا يذكر Hono v4، Preact+Vite، Next.js 15، Biome، Turborepo — هذه قرارات مؤكدة من Phases 1-3.
3. **Branding غير متسق:** القسم 9 يصف white-label كميزة MVP بينما PRD يصنفها صراحةً كـ "Deferred".

التصحيح يتطلب إعادة كتابة مستهدفة لأقسام 1، 7، و9 — ليس استبدالاً كاملاً.

---

### 🟠 ORANGE — 03_CLAUDE_md_Working_Standards.md

**المبرر:** ~60% من المحتوى ذو قيمة حقيقية (بروتوكول الجلسة، بروتوكول الطوارئ، قوائم المراجعة) لكن ~40% متقادم كلياً:
- **القسم 2 (OBSOLETE كلياً):** ملفات CLAUDE.md لكل تطبيق — هذا النهج تم التخلي عنه. الملفات الثلاثة حُذفت.
- **القسم 4 (OBSOLETE كلياً):** مخطط دورة حياة 16-hook — الواقع: 5 hooks فقط نشطة.
- **القسم 1 (جزئياً):** تقنيات Widget/Dashboard خاطئة + 6 قواعد فقط (مفقود 7 و8) + 4 مصادر معرفة.
- **القسم 5 (جزئياً):** agents مؤرشفة موجودة في الجدول.
- **انتهاك Rule 7:** يجب نقله من `docs/` إلى `docs/claude/`.

---

### ⚫ BLACK — 01_Tools_Report.md

**المبرر:** متجاوز كلياً بـ `docs/claude/tools_report_v2.md`. المحتوى خطير إذا تم الرجوع إليه:
- يدّعي 31 agent نشطاً → الواقع: 9
- يدّعي 20 MCP server → الواقع: 5
- يدّعي 16 hook → الواقع: 5
- يدّعي 35 command → الواقع: 16
- أمر التثبيت الشامل يُثبّت 975 مكوناً معظمها مرفوض
- **انتهاك Rule 7:** موجود في `docs/` بدلاً من `docs/claude/`

---

## الاكتشافات المشتركة عبر جميع الوثائق

### 1. تعارض نطاق ICP عبر وثيقتين
| الوثيقة | القيمة المذكورة |
|---------|----------------|
| BRD القسم 1 | 30-150 طلب/شهر |
| BRD القسم 4 (KPIs 6 أشهر) | 30-300 طلب/شهر |
| Full Project Doc القسم 1 | 10-500 طلب/شهر ❌ خطأ |
| MRD، PRD، SRD | 30-300 طلب/شهر ✅ |
| CLAUDE.md | 30-300 طلب/شهر ✅ |

**القيمة الموثوقة:** 30-300 طلب/شهر (CLAUDE.md كمرجع).

### 2. تناقض تسمية خطة "Premium"
| الوثيقة | الاسم المستخدم |
|---------|----------------|
| Extended Architecture | "Higher Plan" |
| SRD | "Higher" / "Highest" |
| BRD، Full Project، Database schema، CLAUDE.md | "Mid" / "Premium" ✅ |

### 3. غياب تفاصيل التسعير
أسعار الخطط (99/299/799 SAR) وحدود الردود (500/2000/10000) موجودة في DB و CLAUDE.md لكن **لا تظهر في MRD أو PRD أو SRD** — البيانات اليتيمة الأكثر خطورة في الوثائق.

### 4. لا تواريخ في أي وثيقة
لا توجد وثيقة أصلية تحمل تاريخ إنشاء أو آخر تحديث. هذا يجعل من المستحيل تقييم قِدَم أي قسم دون مراجعة الكود.

### 5. انتهاكات Rule 7
ثلاث وثائق Claude-generated في `docs/` بدلاً من `docs/claude/`:
- `docs/01_Tools_Report.md` ← `docs/claude/` (أو أرشفة)
- `docs/02_Implementation_Plan.md` ← `docs/claude/plans/`
- `docs/03_CLAUDE_md_Working_Standards.md` ← `docs/claude/`
