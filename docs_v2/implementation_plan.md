# مراد بوت — خطة التنفيذ
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.0
**الحالة:** معتمد

---

## الهدف
خارطة طريق تنفيذية كاملة لبناء مراد بوت من الصفر حتى Production، مرحلةً بمرحلة، مع معايير اكتمال قابلة للتحقق.

## النطاق
✅ يغطي:
- المراحل 1–9 بالتسلسل الفعلي الحالي
- حالة كل مرحلة (مكتملة / التالية / معلّقة)
- معايير الاكتمال لكل مرحلة
- التقنيات الدقيقة المستخدمة فعلاً

❌ لا يغطي:
- Phase 0 (أمر تثبيت 975 مكوّناً) — OBSOLETE
- تتبع الطلبات، WhatsApp، لغة إنجليزية، Analytics متقدمة
- Enterprise Scaling أو ميزات ما بعد MVP

---

## ملخص المراحل

| # | المرحلة | الحالة | المدة |
|---|---------|--------|-------|
| 1 | هيكل المشروع والبيئة | ✅ مكتملة | يوم 1-2 |
| 2 | قاعدة البيانات — Supabase | ✅ مكتملة | يوم 3-5 |
| 3 | API Foundation — Hono | ✅ مكتملة | يوم 6-9 |
| 4 | Salla Client — OAuth + Sync | 🔜 التالية | يوم 10-13 |
| 5 | AI Orchestrator + Widget | ⏳ معلّقة | يوم 14-21 |
| 6 | Merchant Dashboard | ⏳ معلّقة | يوم 22-26 |
| 7 | Escalation + Billing + Email | ⏳ معلّقة | يوم 27-30 |
| 8 | الأمان والاختبار الشامل | ⏳ معلّقة | يوم 31-35 |
| 9 | Production Deployment | ⏳ معلّقة | يوم 36-40 |

---

## المرحلة 1: هيكل المشروع والبيئة
**الحالة:** ✅ مكتملة | **المدة:** يوم 1-2

### ما تم بناؤه
- Monorepo بـ pnpm + Turborepo
- Biome للـ Linting والـ Formatting (لا ESLint+Prettier)
- Packages الأساسية: `shared`، `ai-orchestrator`، `salla-client`
- `tsconfig.base.json` مشترك
- `turbo.json` مع pipeline كامل

### هيكل المشروع الفعلي
```
moradbot/
├── apps/
│   ├── api/          # Cloudflare Worker — Hono v4
│   ├── widget/       # Preact + Vite → bundled JS
│   └── dashboard/    # Next.js 15 App Router
├── packages/
│   ├── shared/       # TypeScript types (Database interface)
│   ├── ai-orchestrator/
│   └── salla-client/
├── supabase/
│   └── migrations/
└── docs/
```

### معايير الاكتمال
- [x] `turbo build` يعمل بدون أخطاء
- [x] TypeScript يُترجم بدون أخطاء
- [x] Biome مُعدَّل: tabs، double quotes، 100 char line width
- [x] Supabase CLI متصل

---

## المرحلة 2: قاعدة البيانات — Supabase
**الحالة:** ✅ مكتملة | **المدة:** يوم 3-5

### ما تم بناؤه
- 12 جدولاً (لا 10) مع RLS محكم على جميعها
- 5 migration files منظّمة في `supabase/migrations/`
- TypeScript types مُولَّدة في `packages/shared/`

### الجداول الـ 12

| # | الجدول | الغرض |
|---|--------|-------|
| 1 | `plans` | باقات الاشتراك |
| 2 | `stores` | المتاجر المشتركة |
| 3 | `subscriptions` | اشتراكات نشطة + عداد الاستخدام |
| 4 | `faq_entries` | إجابات FAQ |
| 5 | `product_snapshots` | لقطات منتجات سلة |
| 6 | `visitor_sessions` | جلسات الزوار |
| 7 | `tickets` | تذاكر المحادثة |
| 8 | `messages` | رسائل المحادثة |
| 9 | `escalations` | حالات التصعيد |
| 10 | `audit_logs` | سجل التدقيق |
| 11 | `usage_tracking` | تتبع الاستخدام |
| 12 | `bot_configurations` | إعدادات البوت لكل متجر |

### معايير الاكتمال
- [x] 12 جدولاً مُنشأة بنجاح
- [x] RLS مُفعَّل على جميع الجداول (policy: `auth.uid() = store_id`)
- [x] TypeScript types مُولَّدة في `packages/shared/src/types/`
- [x] 5 migration files مُطبَّقة بنجاح
- [x] اختبارات عزل البيانات تنجح

---

## المرحلة 3: API Foundation — Hono
**الحالة:** ✅ مكتملة | **المدة:** يوم 6-9

### ما تم بناؤه
- 16 endpoint على Hono v4 + Cloudflare Workers
- Middleware chain: `cors` → `errorHandler` → `auditLog`
- Supabase dual-client pattern
- Error hierarchy كامل

### الـ 16 Endpoints

| Method | Endpoint | الوظيفة |
|--------|----------|---------|
| GET | `/health` | فحص صحة الـ Worker |
| GET | `/auth/salla/start` | بدء OAuth |
| GET | `/auth/salla/callback` | استقبال code وتبادله |
| POST | `/auth/salla/refresh` | تجديد token |
| POST | `/api/chat` | استقبال رسائل Widget |
| GET | `/api/faq` | قائمة FAQ |
| POST | `/api/faq` | إنشاء FAQ entry |
| PUT | `/api/faq/:id` | تعديل FAQ entry |
| DELETE | `/api/faq/:id` | حذف ناعم |
| GET | `/api/stats` | إحصاءات Dashboard |
| GET | `/api/stats/usage` | مؤشرات الاستخدام |
| GET | `/api/tickets` | قائمة المحادثات |
| GET | `/api/tickets/:id` | تفاصيل محادثة |
| GET | `/api/escalations` | قائمة التصعيدات |
| PATCH | `/api/escalations/:id` | تحديث حالة تصعيد |

### البنية التقنية المُنجزة
```
src/
├── index.ts          # Entry point
├── app.ts            # createApp() + middleware
├── env.ts            # Env interface (Cloudflare bindings)
├── lib/
│   ├── supabase.ts   # dual-client pattern
│   └── errors.ts     # AppError hierarchy
├── middleware/
│   ├── cors.ts
│   ├── errorHandler.ts
│   └── auditLog.ts
└── routes/
    ├── auth.ts
    ├── chat.ts
    ├── faq.ts
    ├── stats.ts
    ├── tickets.ts
    └── escalations.ts
```

**Supabase dual-client:**
- `createSupabaseClient(env, storeId)` — anon key + `x-store-id` header (RLS enforced)
- `createSupabaseAdmin(env)` — service role key (bypass RLS للعمليات النظامية)

**Error hierarchy:**
`AppError` → `ValidationError` | `AuthenticationError` | `AuthorizationError` | `NotFoundError` | `RateLimitError` | `DatabaseError`

### معايير الاكتمال
- [x] 16 endpoint تستجيب بشكل صحيح
- [x] Middleware chain يعمل بالترتيب الصحيح
- [x] كل request يحمل `store_id` صحيح
- [x] Audit log لكل عملية حساسة
- [x] Error handling موحّد عبر جميع الـ endpoints

---

## المرحلة 4: Salla Client — OAuth + Sync
**الحالة:** 🔜 التالية | **المدة:** يوم 10-13

### الهدف
تاجر يضغط Install → يُعطي إذن → Widget يظهر تلقائياً. ومزامنة دورية لمنتجات سلة.

### الخطوات

#### 4.1 — بناء `packages/salla-client`
```
packages/salla-client/src/
├── oauth.ts        # Authorization Code Flow (Custom Mode)
├── products.ts     # GET /products فقط (read-only)
├── types.ts        # Salla API response types
└── index.ts
```

**OAuth Custom Mode (Salla):**
1. تاجر يضغط Install في Salla App Store
2. redirect إلى `GET /auth/salla/start` (موجود بالفعل في Phase 3)
3. redirect إلى Salla Authorization Server
4. Salla يُعيد إلى `GET /auth/salla/callback` مع code
5. نُبادل code بـ access_token + refresh_token
6. نُخزّن tokens في `stores` table (مُشفَّرة)

**قواعد صارمة:**
- فقط `GET /products` — لا كتابة، لا حذف، لا تعديل (Rule 2)
- state parameter لمنع CSRF
- Rate Limiting باحترام Salla API limits

#### 4.2 — Product Sync + Cloudflare KV

| الباقة | التكرار | Cron |
|--------|---------|------|
| الانطلاق | كل 24 ساعة | `0 2 * * *` |
| النمو | كل 6 ساعات | `0 */6 * * *` |
| المتمكّن | كل ساعة | `0 * * * *` |

- Rate limiting state مُخزَّن في Cloudflare KV
- Batch processing: 100 منتج/request
- Target: 1,000 منتج < 60 ثانية
- تحديث `product_snapshots.synced_at` بعد كل مزامنة

### معايير الاكتمال
- [ ] OAuth flow يعمل end-to-end
- [ ] Tokens مُخزَّنة مُشفَّرة
- [ ] `GET /products` يجلب ويُخزّن في `product_snapshots`
- [ ] 3 Cron schedules تعمل
- [ ] 1,000 منتج تُزامَن < 60 ثانية
- [ ] Cloudflare KV يُدير rate limiting state
- [ ] لا كتابة على Salla (read-only مؤكّد)

---

## المرحلة 5: AI Orchestrator + Widget
**الحالة:** ⏳ معلّقة | **المدة:** يوم 14-21

### الهدف
وكيل واحد يرد بالعربية + Widget Preact خفيف < 50KB يعمل على جميع صفحات المتجر.

### AI Orchestrator (`packages/ai-orchestrator`)

**OpenRouter Integration:**
- Primary: google/gemini-2.0-flash (سرعة + تكلفة منخفضة)
- Fallback: GPT-4o Mini
- Timeout: 8 ثوان | Retry: 2 محاولات

**قيود AI:**
- عربي فقط حتى لو الزائر كتب بالإنجليزية (Rule 5)
- لا يُجيب خارج نطاق FAQ + product_snapshots
- يُصعِّد بعد 3 محاولات فاشلة

### Chat Widget (`apps/widget`)

**التقنية:** Preact + Vite → bundled JS (لا Vanilla TypeScript)

**7 حالات Widget:**
1. مغلق → أيقونة في الزاوية
2. مفتوح → نافذة محادثة
3. يكتب → مؤشر (...)
4. استجابة → رسالة الرد
5. تصعيد → نموذج جمع البيانات
6. خطأ → رسالة احترامية
7. ممتلئ → "انتهى الحد الشهري للمتجر"

**قواعد Widget:**
- أول رسالة: إفصاح AI إلزامي
- لا cookies/storage بدون موافقة صريحة
- لا يظهر على مسارات `/checkout/*`
- 20 رسالة/دقيقة للزائر

### معايير الاكتمال
- [ ] دقة الردود ≥ 90% على أسئلة FAQ
- [ ] P50 ≤ 1.5s | P95 ≤ 3.0s
- [ ] Widget < 50KB gzipped
- [ ] لا يظهر على Checkout
- [ ] Consent flow مُطبَّق
- [ ] Fallback Model يعمل
- [ ] جميع الـ 7 حالات تعمل

---

## المرحلة 6: Merchant Dashboard
**الحالة:** ⏳ معلّقة | **المدة:** يوم 22-26

### الهدف
لوحة تحكم بسيطة للتاجر (P95 ≤ 2.5 ثانية) على Next.js 15 App Router.

### التقنية: Next.js 15 App Router + Cloudflare Pages + `@cloudflare/next-on-pages`

> **تحذير تقني:** Next.js 15 على Cloudflare Pages يتطلب حزمة `@cloudflare/next-on-pages` adapter إلزامياً. بدونها `next build` ينتج artifacts لـ Node.js لا تعمل على Edge runtime. أضف الحزمة وعدّل `next.config.ts` قبل بدء البناء:
> ```bash
> cd apps/dashboard && pnpm add -D @cloudflare/next-on-pages
> ```

**4 أقسام:**

| القسم | المحتوى |
|-------|---------|
| الرئيسية | استخدام الباقة (شريط بصري) + زر تشغيل/إيقاف البوت |
| المحادثات | قائمة Tickets — قراءة فقط |
| التصعيدات | عرض + زر إغلاق يدوي |
| الإعدادات | إدارة FAQ + معلومات الاشتراك |

**شريط الاستخدام:**
- 0–79%: أخضر
- 80–99%: برتقالي (تحذير)
- 100%: أحمر (ممتلئ)

**قواعد Dashboard:**
- لا يعرض بيانات متاجر أخرى أبداً
- كل نداء API يحمل JWT صحيح
- Real-time subscriptions للتصعيدات فقط (Supabase Realtime)

### معايير الاكتمال
- [ ] Dashboard يُحمَّل < 2.5 ثانية (P95)
- [ ] 4 أقسام تعمل
- [ ] Real-time updates للتصعيدات
- [ ] زر التشغيل/الإيقاف يعمل فوراً
- [ ] JWT Auth مُطبَّق على جميع النداءات
- [ ] عزل البيانات بين المتاجر مؤكَّد

---

## المرحلة 7: Escalation + Billing + Email
**الحالة:** ⏳ معلّقة | **المدة:** يوم 27-30

### الهدف
منطق التصعيد الكامل + عداد دقيق للاستخدام + إشعارات Resend.

### Escalation Flow

```
المحاولة 1: Bot يحاول الإجابة
المحاولة 2: Bot يطلب توضيح
المحاولة 3: Bot يُعلن التصعيد → نموذج جمع البيانات
```

عند التصعيد:
- جمع: وصف المشكلة (إلزامي) + رقم الطلب (اختياري) + بريد/هاتف (واحد إلزامي)
- تشفير بيانات التواصل
- حفظ في `escalations` table
- إشعار في Dashboard التاجر (Real-time)

### Usage Metering + Resend Notifications

| الحدث | الإجراء |
|-------|---------|
| وصول 80% | إيميل تحذير للتاجر عبر Resend |
| وصول 100% | إيميل للتاجر + Widget يعرض رسالة بديلة |
| بداية دورة جديدة | إعادة العداد إلى 0 |

### معايير الاكتمال
- [ ] Escalation يُشغَّل بعد المحاولة الثالثة تحديداً
- [ ] بيانات التواصل مُشفَّرة في DB
- [ ] عداد الاستخدام دقيق (يرتفع عند كل `is_bot_reply = true`)
- [ ] إيميل 80% يُرسَل عبر Resend
- [ ] إيميل 100% يُرسَل + Widget يتغير للحالة 7
- [ ] التاجر يرى التصعيدات في Dashboard فوراً

---

## المرحلة 8: الأمان والاختبار الشامل
**الحالة:** ⏳ معلّقة | **المدة:** يوم 31-35

### الهدف
مراجعة أمنية + اختبارات شاملة بـ Vitest + Playwright + PDPL Compliance.

> **ملاحظة:** Vitest غير مُثبَّت حالياً في package.json للحزم. أول خطوة في هذه المرحلة هي تثبيته:
> ```bash
> pnpm add -D vitest @vitest/coverage-v8 --filter @moradbot/api
> pnpm add -D vitest @vitest/coverage-v8 --filter @moradbot/widget
> ```
> ثم إضافة `vitest.config.ts` لكل حزمة.

### Security Audit (قائمة المراجعة)

```
✓ TLS 1.2+ على جميع الاتصالات
✓ تشفير tokens وبيانات escalation contacts
✓ Rate Limiting: 20 msg/min للزائر
✓ لا أسرار في الكود أو .env
✓ Audit Logs retention 90+ يوم
✓ RLS على 12 جداول
✓ عزل البيانات بين المتاجر
✓ CSRF protection في OAuth
✓ Input validation لجميع النماذج
```

### PDPL Compliance (نظام حماية البيانات الشخصية السعودي)

```
✓ موافقة صريحة موثَّقة في consent_logs (visitor)
✓ حذف البيانات الشخصية 30-90 يوم بعد إلغاء الاشتراك
✓ عدم مشاركة البيانات مع أطراف ثالثة
✓ وضوح سياسة الخصوصية
```

### أهداف Test Coverage

| المكوّن | الهدف |
|--------|-------|
| AI Orchestrator | ≥ 85% |
| Escalation Flow | ≥ 90% |
| Usage Metering | ≥ 90% |
| Data Isolation | 100% |
| Rate Limiting | ≥ 85% |
| OAuth Flow | ≥ 80% |
| الإجمالي | ≥ 80% |

### معايير الاكتمال
- [ ] Security Audit: 0 Critical Issues
- [ ] PDPL Checklist 100% مكتمل
- [ ] Test Coverage ≥ 80% (Vitest)
- [ ] E2E tests تنجح (Playwright)
- [ ] Load Test: P50 ≤ 1.5s, P95 ≤ 3.0s, Error Rate < 0.1%

---

## المرحلة 9: Production Deployment
**الحالة:** ⏳ معلّقة | **المدة:** يوم 36-40

### الهدف
نشر يدوي آمن (Rule 6) مع مراقبة كاملة ثم Beta مغلق.

### Pre-Deploy Checklist الإلزامي

```
□ جميع الاختبارات تنجح (0 failures)
□ Test Coverage ≥ 80%
□ Security Audit: 0 Critical Issues
□ لا secrets في الكود
□ Cloudflare Secrets مُعدَّة (wrangler secret put ...)
□ Supabase Migrations مُطبَّقة على Production
□ Monitoring Alerts مُعدَّة
□ Rollback Plan موثَّق
```

### ترتيب النشر (الترتيب مهم)

```bash
# 1. تطبيق Migrations على Production DB
supabase db push

# 2. نشر API Worker
cd apps/api && pnpm deploy

# 3. نشر Widget
cd apps/widget && pnpm deploy

# 4. نشر Dashboard على Cloudflare Pages
cd apps/dashboard && pnpm deploy
```

### Monitoring Alerts

| القاعدة | الشرط | الإجراء |
|---------|-------|---------|
| Error Rate | > 5% لمدة 5 دقائق | تنبيه فوري |
| Latency | Avg > 4s لمدة 5 دقائق | تنبيه + مراجعة |
| Sync Failure | > 30 دقيقة بدون sync | تنبيه + إعادة محاولة |
| Security Error | أي خطأ أمني | إغلاق فوري |

### Beta Launch Plan

```
الأسبوع 1: 3-5 متاجر مختارة (Beta مغلق)
الأسبوع 2: جمع Feedback + إصلاح المشاكل
الأسبوع 3: 10-20 متجر
الأسبوع 4: فتح التسجيل المحدود
```

### معايير الاكتمال
- [ ] Pre-Deploy Checklist 100% مكتمل
- [ ] نشر ناجح بدون أخطاء (يدوي)
- [ ] Monitoring يرصد جميع الـ Metrics
- [ ] Widget يعمل في متجر تجريبي في Production
- [ ] أول 3 متاجر Beta تعمل

---

## القرارات المعمارية المحفوظة
- Cloudflare Workers (لا Edge Functions أو Lambda)
- Supabase dual-client: anon key للـ RLS + service role للعمليات النظامية
- OpenRouter كـ AI gateway (لا direct API calls)
- Preact + Vite للـ Widget (لا Vanilla TypeScript، لا React)
- Next.js 15 App Router للـ Dashboard (لا React + Vite) + `@cloudflare/next-on-pages` للنشر
- Biome للـ Linting/Formatting (لا ESLint + Prettier)
- CLAUDE.md واحد في الجذر (لا per-app CLAUDE.md)
- Vitest للاختبارات (يُثبَّت في Phase 8 — غير مثبت حالياً)
- `compatibility_date = "2025-11-01"` و `compatibility_flags = ["nodejs_compat"]` — ✅ مُطبَّق (فبراير 2026)
