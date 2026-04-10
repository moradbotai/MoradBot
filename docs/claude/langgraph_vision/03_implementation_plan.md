# مراد بوت — خطة التنفيذ (محدَّثة بـ LangGraph)
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.1 (محدَّث أبريل 2026 — Phase 5 يعتمد LangGraph)
**الحالة:** معتمد

> ⚡ **التحديث الجوهري في هذه النسخة:** Phase 5 الآن يصف بناء **LangGraph Python Service** بدلاً من AI Orchestrator عام.

---

## ملخص المراحل

| # | المرحلة | الحالة | المدة |
|---|---------|--------|-------|
| 1 | هيكل المشروع والبيئة | ✅ مكتملة | يوم 1-2 |
| 2 | قاعدة البيانات — Supabase | ✅ مكتملة | يوم 3-5 |
| 3 | API Foundation — Hono | ✅ مكتملة | يوم 6-9 |
| 3.5 | Landing Page + PostHog | ✅ مكتملة | (مارس 2026) |
| 4 | Salla Client — OAuth + Sync | 🔜 التالية | يوم 10-13 |
| 5 | **LangGraph AI Service + Widget** | ⏳ معلّقة | يوم 14-24 |
| 6 | Merchant Dashboard + Enhanced Agents | ⏳ معلّقة | يوم 25-32 |
| 7 | Escalation + Billing + Email | ⏳ معلّقة | يوم 33-37 |
| 8 | الأمان والاختبار الشامل | ⏳ معلّقة | يوم 38-43 |
| 9 | Production Deployment | ⏳ معلّقة | يوم 44-48 |

---

## المرحلة 1: هيكل المشروع والبيئة
**الحالة:** ✅ مكتملة

### ما تم بناؤه
- Monorepo بـ pnpm + Turborepo
- Biome للـ Linting والـ Formatting
- Packages الأساسية: `shared`، `ai-orchestrator`، `salla-client`
- `tsconfig.base.json` + `turbo.json` كاملان

---

## المرحلة 2: قاعدة البيانات — Supabase
**الحالة:** ✅ مكتملة

### الجداول الـ 14 (بعد إضافة waitlist)

| # | الجدول | الغرض |
|---|--------|-------|
| 1 | `plans` | باقات الاشتراك |
| 2 | `stores` | المتاجر المشتركة |
| 3 | `subscriptions` | اشتراكات نشطة |
| 4 | `faq_entries` | إجابات FAQ (GIN full-text) |
| 5 | `product_snapshots` | لقطات منتجات سلة |
| 6 | `visitor_sessions` | جلسات الزوار |
| 7 | `tickets` | تذاكر المحادثة |
| 8 | `messages` | رسائل المحادثة |
| 9 | `escalations` | حالات التصعيد |
| 10 | `usage_events` | تتبع الاستخدام (الاسم الفعلي في DB) |
| 11 | `consent_logs` | سجلات الموافقة |
| 12 | `audit_logs` | سجل التدقيق |
| 13 | `waitlist_submissions` | قائمة الانتظار |

> ⚠️ `bot_configurations` غير موجود في أي migration file حالياً. يُذكر في بعض الوثائق لكن لم يُنشأ بعد.

### معايير الاكتمال
- [x] 14 جدولاً مُنشأة بنجاح
- [x] RLS مُفعَّل (policy: `auth.uid() = store_id`)
- [x] 8 migration files مُطبَّقة

---

## المرحلة 3: API Foundation — Hono
**الحالة:** ✅ مكتملة

### الـ 16 Endpoints + /health

جميعها مبنية ومختبرة. `chat.ts` يعيد mock response حالياً — ينتظر Phase 5.

---

## المرحلة 3.5: Landing Page + PostHog
**الحالة:** ✅ مكتملة (مارس–أبريل 2026)

- Landing page كاملة مع PWA manifest
- PostHog Browser SDK + reverse proxy عبر Cloudflare Worker
- Waitlist form → Supabase
- Stats dashboard (`stats.html`) بيانات PostHog + Supabase

---

## المرحلة 4: Salla Client — OAuth + Sync
**الحالة:** 🔜 التالية

### الهدف
تاجر يضغط Install → يُعطي إذن → Widget يظهر تلقائياً + مزامنة دورية للمنتجات.

### الخطوات

#### 4.1 — بناء `packages/salla-client`
```
packages/salla-client/src/
├── oauth.ts        # Authorization Code Flow (Custom Mode)
├── products.ts     # GET /products فقط (read-only)
├── types.ts        # Salla API response types
└── index.ts
```

#### 4.2 — Product Sync + Cloudflare KV

| الباقة | التكرار | Cron |
|--------|---------|------|
| الانطلاق | كل 24 ساعة | `0 2 * * *` |
| النمو | كل 6 ساعات | `0 */6 * * *` |
| المتمكّن | كل ساعة | `0 * * * *` |

### معايير الاكتمال
- [ ] OAuth flow يعمل end-to-end
- [ ] Tokens مُخزَّنة مُشفَّرة
- [ ] `GET /products` يجلب ويُخزّن في `product_snapshots`
- [ ] 1,000 منتج تُزامَن < 60 ثانية

---

## المرحلة 5: LangGraph AI Service + Widget
**الحالة:** ⏳ معلّقة

### الهدف
**بناء LangGraph Python Service منفصل** + Widget Preact خفيف < 50KB.

### 5.1 — LangGraph AI Service (`packages/ai-orchestrator`)

**التقنية:** Python + LangGraph + FastAPI
**الاستضافة:** Railway / Fly.io (~$5/شهر)

**هيكل الكود:**
```
packages/ai-orchestrator/
├── pyproject.toml
├── src/
│   ├── graphs/
│   │   └── chat_graph.py        ← StateGraph الأساسي
│   ├── nodes/
│   │   ├── validate.py          ← Prompt injection guard
│   │   ├── enrich.py            ← Page context enrichment
│   │   ├── retrieve.py          ← FAQs + Products من Supabase
│   │   ├── classify.py          ← Intent classification
│   │   ├── generate.py          ← OpenRouter → Gemini 2.0 Flash
│   │   └── escalate.py          ← Trigger escalation
│   ├── state.py                 ← MoradBotState TypedDict
│   ├── config.py                ← Per-merchant configuration
│   ├── checkpointer.py          ← PostgresSaver setup
│   └── server.py                ← FastAPI + LangSmith
└── tests/
```

**Graph Flow (MVP):**
```
START
  → [validate_input]       ← منع Prompt Injection
  → [enrich_context]       ← إضافة سياق الصفحة
  → [retrieve_knowledge]   ← FAQs + Product snapshots
  → [classify_intent]      ← FAQ | Product | Unknown | Escalate
  → [generate_response]    ← OpenRouter → Gemini 2.0 Flash
  → [check_clarification]  ← > 3 محاولات → تصعيد تلقائي
  → [save_to_db]           ← Supabase write
  → END
```

**Checkpointing:** PostgresSaver → Supabase (نفس قاعدة البيانات)

**التكامل مع Hono API:**
```typescript
// apps/api/src/routes/chat.ts
// POST /api/chat يُرسل الطلب للـ LangGraph service
const response = await fetch(env.AI_SERVICE_URL + "/chat", {
  method: "POST",
  body: JSON.stringify({ storeId, message, context, ticketId })
});
```

### 5.2 — Chat Widget (`apps/widget`)

**التقنية:** Preact + Vite → bundled JS (هدف: <50KB gzipped)

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
- يُرسل `context` (page_url، page_type، product_id) مع كل رسالة
- PostHog: تتبع كل تفاعل (مع الحفاظ على حد الـ <50KB gzipped)

### معايير الاكتمال
- [ ] LangGraph Service يعمل على Railway/Fly.io
- [ ] دقة الردود ≥ 90% على أسئلة FAQ
- [ ] P50 ≤ 1.5s | P95 ≤ 3.0s (شاملاً latency الخدمة)
- [ ] Widget < 50KB gzipped
- [ ] LangSmith traces تعمل
- [ ] Checkpointing يعمل عبر Supabase PostgresSaver
- [ ] جميع الـ 7 حالات تعمل

---

## المرحلة 6: Merchant Dashboard + Enhanced Agents
**الحالة:** ⏳ معلّقة

### 6.1 Dashboard (Next.js 15 App Router + Cloudflare Pages)
4 أقسام: الرئيسية، المحادثات، التصعيدات، الإعدادات.

> **تحذير:** يتطلب `@cloudflare/next-on-pages` adapter.

### 6.2 Enhanced Agents (LangGraph)
```
packages/ai-orchestrator/src/agents/
├── customer_success.py  ← مراقبة "لا أعرف" → تنبيه التاجر
├── proactive_cs.py      ← مراقبة الشحن → إشعار العميل
└── onboarding.py        ← Onboarding Wizard مع interrupt()
```

### معايير الاكتمال
- [ ] Dashboard يُحمَّل < 2.5 ثانية
- [ ] Customer Success Agent يرسل تنبيهات صحيحة
- [ ] Onboarding Wizard يعمل end-to-end

---

## المرحلة 7: Escalation + Billing + Email
**الحالة:** ⏳ معلّقة

### Escalation Flow الكامل
```
المحاولة 1: Bot يحاول الإجابة
المحاولة 2: Bot يطلب توضيح
المحاولة 3: Bot يُعلن التصعيد → نموذج جمع البيانات
```

### Usage Metering + Resend Notifications
- وصول 80%: إيميل تحذير عبر Resend
- وصول 100%: إيميل + Widget يعرض رسالة بديلة

---

## المرحلة 8: الأمان والاختبار الشامل
**الحالة:** ⏳ معلّقة

### أهداف Test Coverage

| المكوّن | الهدف |
|--------|-------|
| LangGraph Graph | ≥ 85% |
| Escalation Flow | ≥ 90% |
| Usage Metering | ≥ 90% |
| Data Isolation | 100% |
| Rate Limiting | ≥ 85% |
| OAuth Flow | ≥ 80% |

---

## المرحلة 9: Production Deployment
**الحالة:** ⏳ معلّقة

### ترتيب النشر (الترتيب مهم)

```bash
# 1. تطبيق Migrations على Production DB
supabase db push

# 2. نشر LangGraph AI Service
cd packages/ai-orchestrator && railway deploy

# 3. نشر API Worker
cd apps/api && pnpm deploy

# 4. نشر Widget
cd apps/widget && pnpm deploy

# 5. نشر Dashboard على Cloudflare Pages
cd apps/dashboard && pnpm deploy
```

### Beta Launch Plan
```
الأسبوع 1: 3-5 متاجر مختارة (Beta مغلق)
الأسبوع 2: جمع Feedback + إصلاح
الأسبوع 3: 10-20 متجر
الأسبوع 4: فتح التسجيل المحدود
```

---

## القرارات المعمارية المحفوظة
- Cloudflare Workers (لا Edge Functions أو Lambda)
- Supabase dual-client: anon key للـ RLS + service role للعمليات النظامية
- OpenRouter كـ AI gateway (عبر LangGraph — لا direct API calls)
- **LangGraph Python كـ AI Orchestrator (خدمة مستقلة) — قرار أبريل 2026**
- Preact + Vite للـ Widget (< 50KB gzipped)
- Next.js 15 App Router للـ Dashboard + `@cloudflare/next-on-pages`
- Biome للـ Linting/Formatting
- Vitest للاختبارات (يُثبَّت في Phase 8)
