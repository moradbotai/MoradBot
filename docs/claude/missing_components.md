# مراد بوت — المكونات المفقودة من الخطة الحالية
**تاريخ التحليل:** 28 فبراير 2026
**المرجع:** docs_v2/ (SRD، PRD، implementation_plan.md، CLAUDE.md)
**النطاق:** MVP فقط — لا يخرج عن المتطلبات الحالية

---

## ملخص تنفيذي

تم مراجعة الخطة الكاملة (Phases 1–9) مقابل متطلبات SRD وPRD وCLAUDE.md.
تم تحديد **8 مكونات مفقودة** غير مغطاة بأي مرحلة حالية.

| # | المكون | الخطورة | المرحلة المقترحة |
|---|--------|---------|-----------------|
| 1 | Rate Limiting عبر KV (إصلاح الـ in-memory) | 🔴 حرج | Phase 4 |
| 2 | مكتبة التشفير المشتركة | 🔴 حرج | Phase 4 |
| 3 | Retry Utility للخدمات الخارجية | 🟠 مهم | Phase 4 + 5 |
| 4 | Observability Middleware (latency tracking) | 🟠 مهم | Phase 3 → (تحديث) |
| 5 | Salla Webhook Handler (app lifecycle) | 🟠 مهم | Phase 4 |
| 6 | PDPL Data Cleanup Cron | 🟠 مهم | Phase 7 |
| 7 | Onboarding API + FAQ Bulk Setup | 🟡 متوسط | Phase 6 |
| 8 | Spike Detection في Usage Metering | 🟡 متوسط | Phase 7 |

---

## التفصيل الكامل

---

### 1. Rate Limiting عبر Cloudflare KV (إصلاح خطأ قائم)
**الخطورة: 🔴 حرج**

#### ما هو؟
الملف `apps/api/src/middleware/rate-limit.ts` يستخدم حالياً `Map` في الذاكرة (`in-memory`) لتتبع حدود الطلبات. الكود نفسه يحمل تعليقاً:
```ts
// Simple in-memory rate limiter (for MVP)
// TODO: Use Cloudflare KV or Durable Objects for production
```

#### لماذا هو مهم؟
- **انتهاك صريح لـ SRD §4.4:** "آلية التنفيذ: Cloudflare KV حصرًا (in-memory محظور في Cloudflare Workers العديمة الحالة)"
- Cloudflare Workers عديمة الحالة — كل request قد يذهب لـ Worker instance مختلف، مما يجعل الـ in-memory Map عديمة الفائدة عملياً
- الحد الحالي (20 رسالة/دقيقة للزائر، 3000 رد/ساعة للمتجر) لن يُنفَّذ بشكل صحيح في Production

#### أين يُضاف؟
**Phase 4** — يجب أن يكون أول خطوة قبل نشر أي شيء في Production. Phase 4 تنشئ KV namespace لمزامنة المنتجات، فالمنطقي دمج إعداد KV للـ rate limiting في نفس المرحلة.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `backend-architect` | تصميم KV key schema وإدارة الـ window |
| **Agents** | `typescript-pro` | كتابة الـ generic rate limiter مع KV binding |
| **Commands** | `/ultra-think` | قرار معماري: KV vs Durable Objects |
| **Commands** | `/code-review` | مراجعة الكود بعد الكتابة |
| **Commands** | `/security-audit` | التحقق من عدم تجاوز الحدود |
| **MCPs** | `context7` | توثيق Cloudflare KV API |
| **Scripts** | `wrangler kv namespace create RATE_LIMIT_KV` | إنشاء الـ namespace |
| **Hooks** | `security-scanner` | كشف أي تسريب لـ keys |

---

### 2. مكتبة التشفير المشتركة
**الخطورة: 🔴 حرج**

#### ما هو؟
دالة موحّدة لتشفير وفك تشفير البيانات الحساسة باستخدام AES-256-GCM عبر Web Crypto API (built-in في Workers)، تُستخدم في مكانين على الأقل:
- Phase 4: تشفير Salla `access_token` و`refresh_token` قبل تخزينها
- Phase 7: تشفير بيانات تواصل الزوار (email/هاتف) في `escalations`

#### لماذا هو مهم؟
- **SRD §4.4:** "الحقول الحساسة (email/هاتف الزائر، Salla tokens) مشفرة في قاعدة البيانات (AES-256-GCM)"
- **SRD §2.1:** "تخزين tokens مشفرة (AES-256-GCM)"
- بدون مكتبة مشتركة، سيُنفَّذ التشفير بطريقتين مختلفتين في Phase 4 وPhase 7، مما يزيد احتمالية وجود ثغرة
- Web Crypto API على Workers لها quirks (كل العمليات async، IV عشوائي لكل تشفير)

#### أين يُضاف؟
**Phase 4 (خطوة أولى)** — قبل تخزين أي token. يُنشأ كـ utility في `packages/shared/src/crypto.ts` أو `apps/api/src/lib/crypto.ts`.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `security-auditor` | مراجعة تنفيذ AES-256-GCM |
| **Agents** | `typescript-pro` | كتابة الـ generic encrypt/decrypt functions |
| **Commands** | `/security-audit` | التحقق من صحة التنفيذ |
| **Commands** | `/write-tests` | اختبار encrypt → decrypt round-trip |
| **MCPs** | `context7` | توثيق Web Crypto API على Workers |
| **Hooks** | `security-scanner` | كشف أي plaintext sensitive data |

---

### 3. Retry Utility للخدمات الخارجية
**الخطورة: 🟠 مهم**

#### ما هو؟
دالة مشتركة لـ exponential backoff retry تُستخدم في:
- استدعاءات Salla API (مزامنة المنتجات) — SRD §2.5: "Retry عند فشل المزامنة خلال 5 دقائق"
- استدعاءات OpenRouter — implementation_plan.md Phase 5: "Retry: 2 محاولات"

#### لماذا هو مهم؟
- بدون retry، أي فشل مؤقت في Salla أو OpenRouter سيرسل خطأً للمستخدم أو يتوقف عن المزامنة
- الـ P95 latency target (3.0s) يعتمد على retry سريع (وليس بطيئاً)
- مزامنة المنتجات كل ساعة (للمتمكّن) لا تتحمل الفشل دون retry

#### أين يُضاف؟
- **Phase 4:** retry لمزامنة Salla
- **Phase 5:** retry لـ OpenRouter
- المنطقي: إنشاء utility مشترك في بداية Phase 4

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `backend-architect` | تصميم retry strategy (max attempts، backoff) |
| **Agents** | `typescript-pro` | كتابة generic `withRetry<T>()` function |
| **Commands** | `/write-tests` | اختبار retry مع mock failures |
| **MCPs** | `context7` | توثيق Salla API error codes |

---

### 4. Observability Middleware (تتبع الكمون)
**الخطورة: 🟠 مهم**

#### ما هو؟
middleware يسجّل `duration_ms` لكل request، و`ai_duration_ms` خصيصاً لـ `/api/chat`، وفق متطلبات CLAUDE.md:
```
- Every request logs duration_ms from entry to response.
- /api/chat logs AI provider latency separately as ai_duration_ms.
- Supabase query time is tracked per query using wrapper timing.
- All 5xx errors: logged at error level with stack, store_id, and alert: true
```

#### لماذا هو مهم؟
- **CLAUDE.md Pre-production monitoring checklist:** "Error rate for /api/chat < 1% over 24h in dev" — مستحيل التحقق بدون logging
- أهداف الأداء (P50 ≤ 1.5s / P95 ≤ 3.0s) لا يمكن قياسها بدون `duration_ms`
- **SRD §2.11:** "تسجيل duration_ms لكل طلب، وai_duration_ms لـ /api/chat"
- التنبيهات في Phase 9 تعتمد على هذه البيانات

#### أين يُضاف؟
**Phase 3 → تحديث** أو في بداية Phase 4 — الـ `auditLog` middleware الحالي يجب أن يُوسَّع ليشمل timing. حالياً الـ middleware الثلاثة (`cors`، `errorHandler`، `auditLog`) لا يتتبع أياً منها الـ latency.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `backend-architect` | تصميم logging schema |
| **Agents** | `typescript-pro` | كتابة timing middleware مع Hono context |
| **Commands** | `/optimize-api-performance` | قياس الـ baseline بعد التنفيذ |
| **Commands** | `/code-review` | مراجعة التنفيذ |

---

### 5. Salla Webhook Handler (App Lifecycle)
**الخطورة: 🟠 مهم**

#### ما هو؟
endpoint لاستقبال Salla lifecycle webhooks، أهمها حدث **إلغاء تثبيت التطبيق** من قِبل التاجر. عندما يُلغي التاجر التثبيت من Salla App Store، يُرسل Salla webhook لإشعار التطبيق.

#### لماذا هو مهم؟
- **PDPL §4.6:** "البيانات الشخصية تُحذف خلال 30 يومًا من تاريخ الإلغاء" — يجب معرفة متى يتم الإلغاء لبدء العداد
- بدون webhook handler، لن نعرف متى أُلغي التثبيت إلا بالصدفة أو بفشل OAuth
- يُحدَّث حقل `stores.is_active` فور استقبال حدث الإلغاء
- يُنظَّف Widget فور الإلغاء (لا يظهر في المتجر بعد الإلغاء)

#### أين يُضاف؟
**Phase 4** — في نفس وقت بناء `packages/salla-client`. Endpoint مقترح: `POST /webhooks/salla`.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `backend-architect` | تصميم webhook validation وevent handling |
| **Agents** | `security-auditor` | التحقق من signature الـ webhook |
| **Commands** | `/ultra-think` | قرار: أي Salla events تستوجب المعالجة في MVP |
| **MCPs** | `context7` | توثيق Salla Webhooks API |
| **MCPs** | `firecrawl-mcp` | البحث في وثائق Salla عن webhook events |

---

### 6. PDPL Data Cleanup Cron
**الخطورة: 🟠 مهم**

#### ما هو؟
Cron job مجدوَل يُنفِّذ دورياً (يومياً أو أسبوعياً) حذف البيانات الشخصية للمتاجر التي أُلغيت اشتراكاتها:
- يحذف بيانات تواصل الزوار (email/هاتف) خلال 30 يوماً من الإلغاء
- يُجهِّل بيانات المحادثات بحذف `visitor_id` بعد 30 يوماً
- يحذف `product_snapshots` للمتاجر المُلغاة

#### لماذا هو مهم؟
- **SRD §4.6 (PDPL):** "البيانات الشخصية تُحذف خلال 30 يومًا من تاريخ الإلغاء"
- **SRD §4.6:** "بيانات المحادثات المجهولة يمكن الاحتفاظ بها حتى 90 يومًا"
- بدون آلية تلقائية، يتحول الالتزام القانوني إلى مسؤولية يدوية
- Supabase يدعم Row Deletion عبر `supabase db` ويمكن استدعاؤه من Cloudflare Cron

#### أين يُضاف؟
**Phase 7** — مع منطق Billing وEmail. يُضاف كـ Cron Trigger جديد في `wrangler.toml`.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `database-optimizer` | تصميم queries الحذف بكفاءة (batch delete) |
| **Agents** | `security-auditor` | التحقق من PDPL compliance |
| **Commands** | `/ultra-think` | قرار: ما البيانات تُحذف وما يُجهَّل |
| **Commands** | `/security-audit` | مراجعة سياسة الاحتفاظ |
| **MCPs** | `supabase` | فحص schema والتحقق من الجداول المتأثرة |
| **Scripts** | `wrangler.toml` cron entry | جدولة الـ cleanup |

---

### 7. Onboarding API + FAQ Bulk Setup
**الخطورة: 🟡 متوسط**

#### ما هو؟
تدفق إعداد أولي للتاجر الجديد بعد تثبيت التطبيق مباشرة:
- PRD §1: "أجيب على 5 أسئلة إعداد أساسية"
- إنشاء 5 إدخالات FAQ (واحد لكل فئة: shipping, payment, returns, products, general) دفعةً واحدة
- يمكن تنفيذه كـ `POST /api/faq/bulk` أو `POST /api/onboarding`

#### لماذا هو مهم؟
- **PRD §3 (User Story 1):** "بوصفي تاجراً، أريد تثبيت التطبيق وتفعيله في أقل من 10 دقائق"
- بدون onboarding flow، يضطر التاجر لإنشاء كل FAQ إدخالاً بإدخال (5 API calls منفصلة)
- تجربة الإعداد الأولى تؤثر مباشرة على Activation Rate
- الـ 5 فئات الثابتة (enum) تجعل الـ bulk setup منطقياً وسهل التنفيذ

#### أين يُضاف؟
**Phase 6** — مع بناء Dashboard. يُنفَّذ كـ endpoint جديد في `apps/api/src/routes/faq.ts` أو ملف منفصل.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `backend-architect` | تصميم bulk insert مع transaction |
| **Agents** | `api-documenter` | توثيق الـ endpoint الجديد |
| **Commands** | `/write-tests` | اختبار bulk insert و idempotency |
| **MCPs** | `supabase` | التحقق من RLS على الـ bulk insert |

---

### 8. Spike Detection في Usage Metering
**الخطورة: 🟡 متوسط**

#### ما هو؟
منطق يكتشف ارتفاعاً مفاجئاً في استهلاك ردود البوت لمتجر معين، ويُطلق تحذيراً:
- **SRD §2.8:** "Spike detection: تجاوز 200% من المعدل الساعي للخطة يُطلق warn log وإشعار Dashboard"
- المعدل الساعي المتوقع لكل خطة:
  - الانطلاق: 1,000 رد ÷ 720 ساعة = ~1.4 رد/ساعة
  - النمو: 3,000 ÷ 720 = ~4.2 رد/ساعة
  - المتمكّن: 8,000 ÷ 720 = ~11.1 رد/ساعة

#### لماذا هو مهم؟
- يحمي المتجر من استهلاك حصته بسرعة بسبب هجوم أو bug
- يُنبِّه المالك مبكراً قبل وصول حد الـ 80%/100%
- يُساعد على اكتشاف abuse patterns
- التكلفة على مستوى النظام متقلبة — spike في متجر واحد يؤثر على التكاليف الكلية

#### أين يُضاف؟
**Phase 7** — مع منطق Usage Metering. يُضاف كـ check في `POST /api/chat` route بعد تسجيل كل رد.

#### الأدوات المناسبة

| الفئة | الأداة | الاستخدام |
|-------|--------|-----------|
| **Agents** | `backend-architect` | تصميم الـ rolling window detection |
| **Agents** | `database-optimizer` | استعلام usage_tracking بكفاءة للكشف |
| **Commands** | `/optimize-api-performance` | التأكد من أن الـ check لا يضيف latency |
| **Commands** | `/write-tests` | اختبار الـ threshold logic |
| **MCPs** | `supabase` | فحص `usage_tracking` table structure |

---

## ترتيب الأولوية للتنفيذ

### Phase 4 (أضف هذه قبل البدء):
1. ✅ KV-based Rate Limiting — إصلاح خطأ قائم
2. ✅ Encryption Utility — مطلوب لتخزين tokens
3. ✅ Retry Utility — مطلوب للـ Salla API calls
4. ✅ Salla Webhook Handler — مطلوب لـ lifecycle management
5. ✅ Observability Middleware — مطلوب قبل أي نشر

### Phase 6 (أضف هذه):
6. ✅ Onboarding API + FAQ Bulk Setup

### Phase 7 (أضف هذه):
7. ✅ PDPL Data Cleanup Cron
8. ✅ Spike Detection

---

## ما هو ليس مفقوداً (كان مخططاً)

هذه عناصر مطلوبة لكنها **مغطاة بالفعل** في الخطة الحالية:
- Vitest + Test Coverage → Phase 8 ✅ مخطط
- Security Audit + PDPL Checklist → Phase 8 ✅ مخطط
- 80%/100% Usage Notifications → Phase 7 ✅ مخطط
- Escalation 3-attempt logic → Phase 7 ✅ مخطط
- Load Testing → Phase 8 ✅ مخطط
- KV namespace creation → Phase 4 (للمنتجات) ✅ مخطط جزئياً

---

## ملاحظة ختامية

جميع المكونات الـ 8 تقع ضمن نطاق MVP الحالي وتخدم متطلبات موثقة في SRD وPRD وCLAUDE.md. لا يُقترح أي ميزة جديدة خارج النطاق المحدد.
