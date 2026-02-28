# Technology Compatibility Review — MoradBot

تاريخ المراجعة: 22 فبراير 2026
حالة المراجعة: **مُطبَّقة** — التحديثات المُنصَح بها نُفِّذت في نفس الجلسة.
حالة المشروع: Phases 1–3 مكتملة. Phase 4 التالية.

---

## الخلاصة التنفيذية

| النتيجة | العدد |
|---------|-------|
| ✅ متوافق — أبقِ كما هو | 13 |
| ✅ تم تطبيقه | 2 |
| ⚠️ يُفضل معالجته (مؤجل لمرحلته) | 2 |

**التحديثات المُطبَّقة:**
- `wrangler.toml`: `compatibility_date = "2025-11-01"` + `compatibility_flags = ["nodejs_compat"]` + KV binding + non-secret vars
- `CLAUDE.md`: جدول Tech Stack محدَّث + قائمة secrets كاملة + KV setup commands + dashboard adapter note

**المؤجل لمرحلته:**
- `@cloudflare/next-on-pages` → Phase 6 (عند بناء Dashboard UI)
- Vitest → Phase 8 (عند مرحلة الاختبار الشامل)

---

## الخطوة 1 — الحصر الكامل للتقنيات

### أ. الـ Stack الرئيسي

| الطبقة | التقنية | الإصدار المُثبَّت |
|--------|---------|------------------|
| API Runtime | Cloudflare Workers | Workers runtime (Wrangler 3.114.17) |
| API Framework | Hono | 4.11.9 |
| Database | Supabase (PostgreSQL + RLS) | supabase-js 2.48.x |
| Widget | Preact + Vite | Preact 10.28.3 / Vite 6.4.1 |
| Dashboard | Next.js + React | Next.js 15.5.12 / React 19.2.4 |
| Dashboard Hosting | Cloudflare Pages | — |
| Language | TypeScript | 5.9.3 |

### ب. المكتبات والأدوات

| الأداة | الإصدار | الاستخدام |
|--------|---------|-----------|
| Zod | 3.24.x | Schema validation (API + Widget + Shared) |
| @hono/zod-validator | 0.4.1 | Route-level validation في API |
| Biome | 1.9.4 | Lint + Format (بديل ESLint+Prettier) |
| Turborepo | 2.8.9 | Monorepo build orchestration |
| pnpm | 8.15.0 | Package manager |
| @cloudflare/workers-types | 4.20260217.0 | TypeScript types للـ Workers |
| @preact/preset-vite | 2.9.2 | Vite plugin للـ Preact |

### ج. الخدمات الخارجية

| الخدمة | الغرض | المرحلة |
|--------|-------|---------|
| Salla OAuth + API | تفويض التاجر + GET /products | Phase 4 |
| OpenRouter | AI routing → Gemini 2.0 Flash (fallbacks: GPT-4 Mini, Claude 3.5) | Phase 5 |
| Resend | إشعارات البريد الإلكتروني (80%/100% quota) | Phase 5 |
| Cloudflare KV | Rate limiting (visitor + store) | Phase 4 |
| Cloudflare Workers | API runtime | Phase 3 ✅ |
| Cloudflare Pages | Dashboard hosting | Phase 5 |

### د. الحزم الداخلية (Monorepo)

| الحزمة | المحتوى | الحالة |
|--------|---------|--------|
| `@moradbot/shared` | TypeScript types من DB schema | Phase 2 ✅ |
| `@moradbot/ai-orchestrator` | OpenRouter integration | Phase 5 — scaffold فقط |
| `@moradbot/salla-client` | Salla OAuth + Products client | Phase 4 — scaffold فقط |

---

## الخطوة 2 — التوافق مع MVP الحالي

### 1. Cloudflare Workers + Hono

**الإصدار:** Hono 4.11.9 (أحدث stable) / Wrangler 3.114.17

**التقييم:** ✅ **أبقِ كما هو**

- Hono مصمم أساساً للـ Workers. التوافق تام.
- 16 endpoint مبنية وتعمل (Phase 3 مكتملة).
- Middleware chain (cors → errorHandler → auditLog) نظيف وقابل للتوسع.
- `@hono/zod-validator` يضمن validation مدمج في الـ routes.
- **ملاحظة:** `compatibility_date = "2024-02-17"` في `wrangler.toml` قديم (من 2024). يُنصح بتحديثه إلى تاريخ أحدث في Phase 4 لاستفادة من أحدث Workers APIs والإصلاحات.

### 2. Supabase (PostgreSQL + RLS)

**الإصدار:** supabase-js ^2.48.0

**التقييم:** ✅ **أبقِ كما هو**

- 12 جدول بـ RLS policies مكتملة (Phase 2).
- النمط المزدوج (anon client + admin client) صحيح ومناسب.
- Supabase v2 stable، لا تغييرات جذرية متوقعة.
- قاعدة البيانات تدعم multi-tenant بالكامل (`store_id` في كل جدول).

### 3. Preact + Vite (Widget)

**الإصدار:** Preact 10.28.3 / Vite 6.4.1

**التقييم:** ✅ **أبقِ كما هو**

- Preact هو الخيار الأمثل للـ Widget (< 50KB gzipped).
- حجم Preact نفسه ~3KB مقارنةً بـ React 45KB+.
- Vite 6 أحدث stable، سريع للـ dev وبناء ممتاز.
- التكامل مع `@preact/preset-vite` نظيف.

### 4. Next.js 15 + React 19 (Dashboard)

**الإصدار:** Next.js 15.5.12 / React 19.2.4

**التقييم:** ⚠️ **يُفضل معالجته قبل Phase 5**

- الإصدارات نفسها صحيحة وحديثة — App Router هو المعيار.
- **المشكلة:** غياب `@cloudflare/next-on-pages` adapter. Next.js 15 على Cloudflare Pages يتطلب هذه الحزمة وإلا فإن بناء المشروع ينتج artifacts لـ Node.js لا تعمل على Edge runtime.
- حالياً الـ Dashboard placeholder فقط، لذا لا تأثير فوري.
- **الإجراء:** قبل بناء Dashboard UI الحقيقي، أضف `@cloudflare/next-on-pages` وعدّل `next.config.ts`.

### 5. TypeScript 5.9.3

**التقييم:** ✅ **أبقِ كما هو**

- أحدث إصدار. Config صارم (جميع strict flags مفعّلة).
- `verbatimModuleSyntax: true` يضمن import صحيح للـ Workers.
- `target: ES2022` مناسب للـ Workers runtime.

### 6. Biome 1.9.4

**التقييم:** ✅ **أبقِ كما هو**

- أسرع بكثير من ESLint+Prettier.
- يدعم TypeScript و JSX.
- ضبط الـ tabs + double quotes + 100 chars محدد في CLAUDE.md.

### 7. Turborepo 2.8.9 + pnpm 8.15.0

**التقييم:** ⚠️ **يُفضل ترقية pnpm**

- Turborepo 2.8.9 أحدث stable. ✅
- pnpm مُقيَّد بـ 8.15.0 في `packageManager` field لكن آخر إصدار هو pnpm 10.x.
- pnpm 8 يعمل دون مشاكل لكن يفوّت تحسينات الأداء في pnpm 9/10.
- الترقية آمنة لكن تتطلب تحديث `pnpm-lock.yaml`.

### 8. Zod 3.24.x

**التقييم:** ✅ **أبقِ كما هو**

- Zod 3 مستقر وواسع الانتشار. Zod 4 أُعلن عنه لكنه يتطلب migration.
- مشترك بين API + Widget + Dashboard + الحزم الداخلية — الإصدار الموحد مهم.
- لا حاجة للترقية الآن.

### 9. Cloudflare KV (Rate Limiting)

**التقييم:** ✅ **أبقِ كما هو مع ملاحظة**

- KV مناسب للـ rate limiting بهذا الحجم (20 req/min/visitor، 3000 req/hour/store).
- **ملاحظة مهمة:** KV eventual consistent — مستخدم قد يتجاوز الحد بضع طلبات في حالات edge. هذا مقبول لـ MVP وغير حرج.
- Durable Objects أدق لكن أغلى وأعقد — مؤجل لما بعد MVP.

### 10. AES-256-GCM via Web Crypto API

**التقييم:** ✅ **أبقِ كما هو**

- Web Crypto API مدمجة في Workers runtime، لا dependency خارجي.
- AES-256-GCM هو المعيار الصناعي للتشفير المتماثل.
- مناسب لتشفير Salla OAuth tokens في قاعدة البيانات.

---

## الخطوة 3 — التوافق مع الخطط المستقبلية

*بناءً على ما ورد في الوثائق: WhatsApp، multi-store، منصات إضافية (Zid)، analytics متقدمة.*

### WhatsApp Integration

**هل البنية الحالية تدعمها؟** نعم.

- Workers + Hono: أضف routes جديدة `/api/whatsapp/...` — لا تغيير في البنية.
- Supabase: أضف `channel` enum في `tickets` (حالياً widget فقط).
- الحزم: أضف `@moradbot/whatsapp-client` بنفس نمط `salla-client`.
- **ما يُضاف:** Meta Business API credentials في Cloudflare Secrets.

### Multi-Store Support

**هل البنية الحالية تدعمها؟** تقريباً.

- DB schema يدعمها بالكامل (`store_id` في كل مكان، RLS جاهز).
- القيد الحالي هو UNIQUE constraint على `salla_merchant_id` — يُحذف عند التوسع.
- OAuth flow يدعم multiple stores بتعديل بسيط.

### منصات إضافية (Zid، Shopify)

**هل البنية الحالية تدعمها؟** نعم.

- نمط `@moradbot/salla-client` package منفصل يسهّل إضافة `@moradbot/zid-client` إلخ.
- Workers + Hono يدعم routing بناءً على platform header.
- DB: يكفي إضافة `platform` enum في جدول `stores`.

### Advanced Analytics

**هل البنية الحالية تدعمها؟** جزئياً.

- `audit_logs` و `usage_tracking` يوفران بيانات خام.
- للـ analytics الثقيلة (aggregations، historical queries)، Supabase PostgreSQL سيُبطئ.
- عند الحاجة: Cloudflare Analytics Engine أو ClickHouse كـ analytics store منفصل.
- **لا إجراء مطلوب الآن** — المشكلة لن تظهر إلا عند عدد كبير من المتاجر.

---

## الخطوة 4 — المشاكل والفرص

### 🚨 Critical (يجب معالجته)

**لا يوجد ما يوقف العمل حالياً.** المشكلة الوحيدة التي تحتاج معالجة قبل المرحلة المحددة:

| # | المشكلة | التأثير | متى؟ |
|---|---------|---------|------|
| 1 | غياب `@cloudflare/next-on-pages` في Dashboard | Dashboard لن يُنشر على Cloudflare Pages | قبل Phase 5 |

### ⚠️ Recommended (يُفضل معالجته)

| # | المشكلة | التأثير | الأولوية |
|---|---------|---------|---------|
| 2 | `compatibility_date = "2024-02-17"` قديم | يفوّت Workers bug fixes وAPI جديدة من 2024–2026 | Phase 4 |
| 3 | pnpm 8.15.0 بدلاً من 9/10 | يفوّت تحسينات أداء، لكن يعمل | منخفضة |
| 4 | Vitest غير مُثبَّت رغم أنه في CLAUDE.md | لا اختبارات تعمل حالياً | قبل Phase 5 |

### ✅ لا إجراء مطلوب

| التقنية | السبب |
|---------|-------|
| Hono 4.11.9 | أحدث stable، توافق تام مع Workers |
| Supabase v2 | Stable، لا breaking changes متوقعة |
| Preact + Vite 6 | أمثل اختيار للـ Widget |
| TypeScript 5.9.3 | أحدث، config صارم ومناسب |
| Biome 1.9.4 | أداء ممتاز، لا بديل أفضل |
| Zod 3 | ثابت، migration لـ Zod 4 غير ضرورية الآن |
| React 19 + Next.js 15 | أحدث، متوافقان تماماً |
| Turborepo 2.8.9 | أحدث stable |
| Cloudflare KV | مناسب لحجم MVP |
| AES-256-GCM (Web Crypto) | الحل الصحيح لـ Workers runtime |
| OpenRouter | يدعم fallback chain ومرن مع AI providers |
| Resend | حديث، مناسب للـ transactional emails |

---

## الخطوة 5 — التوصيات المصنّفة

### 🚨 يجب تغييره — Critical

#### 1. إضافة `@cloudflare/next-on-pages` للـ Dashboard
**المشكلة:** بدونه، `next build` ينتج output لـ Node.js لا يعمل على Cloudflare Pages edge runtime.

**الإجراء:**
```bash
cd apps/dashboard
pnpm add -D @cloudflare/next-on-pages
```

**تعديل `next.config.ts`:**
```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@moradbot/shared"],
  outputFileTracingRoot: process.cwd(),
  // Required for Cloudflare Pages
  experimental: {
    runtime: "edge",
  },
};

export default nextConfig;
```

**وقت التنفيذ:** قبل بدء بناء Dashboard UI (Phase 5).

---

### ⚠️ يُفضل تغييره — Recommended

#### 2. تحديث `compatibility_date` في `wrangler.toml`
```toml
# من:
compatibility_date = "2024-02-17"
# إلى:
compatibility_date = "2025-11-01"
```
يتيح الاستفادة من Workers API improvements وbug fixes من 2024–2025.
**ملاحظة:** راجع [Cloudflare Compatibility Flags](https://developers.cloudflare.com/workers/configuration/compatibility-dates/) قبل التحديث.

#### 3. إضافة Vitest للاختبارات
مذكور في CLAUDE.md كجزء من stack الاختبار لكنه غير مثبّت.

```bash
# في كل حزمة تحتاج اختبارات
pnpm add -D vitest @vitest/coverage-v8
```

أضف `"test": "vitest"` في scripts كل package.json، وأضف config في `vitest.config.ts`.

#### 4. ترقية pnpm (منخفضة الأولوية)
```bash
# في package.json root
"packageManager": "pnpm@10.0.0"
```
ثم حذف `pnpm-lock.yaml` وإعادة `pnpm install`.

---

### ✅ لا يُنصح بتغييره — Keep as is

| التقنية | السبب |
|---------|-------|
| Hono → Express أو Elysia | Hono هو الأنسب للـ Workers — لا ميزة للتغيير |
| Preact → React | React يخل بـ <50KB هدف Widget |
| Supabase → PlanetScale أو Neon | RLS في Supabase حل مدمج فريد لمتطلبات data isolation |
| OpenRouter → مزود مباشر | OpenRouter يوفر fallback chain، مرونة في تغيير النماذج |
| Biome → ESLint+Prettier | Biome أسرع وأبسط للـ monorepo |
| Zod → Valibot أو Yup | Zod موحّد عبر كل الحزم، migration مكلف |
| Cloudflare KV → Redis | KV كافٍ لـ rate limiting في MVP، Redis تعقيد غير ضروري |

---

## ملخص التكامل بين المكونات

```
Salla App Store
     ↓ OAuth
[Workers + Hono API]  ←→  [Supabase PostgreSQL]
     ↑                          ↑ RLS
[Preact Widget]           [Shared Types Package]
     ↑
[Salla Storefront]

[Next.js Dashboard]  ←→  [Workers + Hono API]
     ↑
[Cloudflare Pages]

External:
  OpenRouter → Gemini 2.0 Flash / GPT-4 Mini / Claude 3.5
  Resend → Email notifications
  Cloudflare KV → Rate limiting
  Cloudflare Secrets → All credentials
```

**نقاط التكامل المُثبَّتة (Phase 3):**
- Widget → API: `POST /api/chat` ✅
- Dashboard → API: JWT bearer على كل endpoint ✅
- API → Supabase: dual-client pattern ✅
- Shared types: يُستخدم في api + widget + dashboard ✅

**نقاط التكامل المعلّقة (Phase 4+):**
- salla-client → API: Phase 4
- ai-orchestrator → API: Phase 5
- Resend → API: Phase 5

---

## القرار النهائي

**الـ Stack الحالي سليم.** جميع الاختيارات مناسبة لـ MVP وقابلة للتوسع للمستقبل. إجراء واحد حرج (Next.js adapter) قبل Phase 5. ثلاثة إجراءات مستحسنة بأولوية متوسطة إلى منخفضة.

لا توجد تقنيات تحتاج استبدالاً.
