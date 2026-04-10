# دليل التقنيات والتكاليف الكاملة — MoradBot
**الإصدار:** 2.1 (محدَّث أبريل 2026 — يشمل LangGraph Python Service)

> آخر تحديث: أبريل 2026
> **التغيير الجوهري:** إضافة تكلفة خدمة LangGraph Python (Railway ~$5/شهر في البداية)

---

## القسم 1 — Stack التقني الكامل

### أ. مكتبات وأدوات مجانية (Open Source)

| التقنية | الإصدار | الاستخدام |
|---------|---------|-----------|
| TypeScript | 5.9.3 | اللغة الرئيسية |
| Hono | 4.11.9 | API framework |
| Preact | 10.28.3 | Chat Widget UI |
| Vite | 6.4.1 | Widget bundler |
| Next.js | 15.5.12 | Dashboard UI |
| Zod | 3.24.x | Schema validation |
| Biome | 1.9.4 | Lint + Format |
| Turborepo | 2.8.9 | Monorepo orchestration |
| pnpm | 8.15.0 | Package manager |
| **LangGraph** | **≥0.2** | **AI Orchestration (جديد)** |
| **FastAPI** | **≥0.110** | **Python AI Service (جديد)** |

### ب. خدمات البنية التحتية (SaaS)

| الخدمة | الدور | التكلفة |
|--------|-------|---------|
| Cloudflare Workers | API runtime | $5/شهر |
| Cloudflare KV | Rate limiting | مدمج مع Workers |
| Cloudflare Pages | Dashboard hosting | $0 |
| Supabase | Database + Auth + Checkpointing | $25/شهر |
| OpenRouter | AI gateway | pay-as-you-go |
| **Railway / Fly.io** | **LangGraph Python Service** | **$5–20/شهر** |
| **LangSmith** | **AI tracing + debugging** | **مجاني حتى 5K traces** |
| Resend | Email notifications | $0 (→ $20 عند ~1,500 متجر) |
| VPS لـ N8N | أتمتة العمليات الداخلية | 46 ريال/شهر |
| Moyasar | بوابة الدفع | 450 ريال مرة + 2.75% |

---

## القسم 2 — تفاصيل خدمة LangGraph

### Railway (الخيار المفضل للبداية)

| الخطة | التكلفة | الذاكرة | CPU |
|-------|---------|---------|-----|
| Hobby (مجاني) | $0 | 512MB | مشترك |
| Starter | $5/شهر | 512MB | مخصص |
| Pro | $20/شهر | 8GB | مخصص |

**للـ MVP:** Starter كافٍ ($5/شهر).
**عند 100+ متجر:** قد يحتاج ترقية حسب الحمل.

### Fly.io (بديل)

- مجاني حتى استهلاك معين
- أنسب للـ proof of concept

### LangSmith (Tracing)

| الخطة | الـ Traces | التكلفة |
|-------|-----------|---------|
| Free | 5,000/شهر | $0 |
| Developer | 50,000/شهر | $39/شهر |
| Plus | unlimited | ~$299/شهر |

للـ MVP: الخطة المجانية (5K traces) كافية.

---

## القسم 3 — تكاليف OpenRouter (الذكاء الاصطناعي)

| النموذج | الدور | Input ($/1M token) | Output ($/1M token) |
|---------|-------|--------------------|--------------------|
| `google/gemini-2.0-flash` | Primary | ~$0.10 | ~$0.40 |
| `gemini-2.0-flash-exp:free` | الباقة المجانية | $0 | $0 |
| `openai/gpt-4o-mini` | Fallback | $0.15 | $0.60 |

**تكلفة رد واحد (Gemini 2.0 Flash):**
```
Input:  1,400 token × ($0.10/1M) = $0.000140
Output:   200 token × ($0.40/1M) = $0.000080
المجموع/رد:                     ≈ $0.000220
أو: ~0.82 ريال لكل 1,000 رد
```

---

## القسم 4 — تكاليف التشغيل الشهرية المُحدَّثة

### التكاليف الثابتة الكاملة شهرياً (بعد LangGraph)

```
Cloudflare Workers Standard:    $5.00
Supabase Pro:                  $25.00
Railway (LangGraph Service):    $5.00   ← جديد
VPS N8N:                       $12.00
التكاليف الثابتة للشركة:       $77.00
──────────────────────────────────────
الإجمالي الثابت الشهري:        ~$124/شهر  (~465 ريال)
```

*تزيد $15/شهر مقارنة بالنسخة السابقة — تأثير ضئيل جداً على نقطة التعادل.*

### جدول التكاليف الشاملة (محدَّث)

| السيناريو | المتاجر | OpenRouter | Infrastructure | **الإجمالي ($)** |
|-----------|---------|------------|---------------|-----------------|
| قبل الإطلاق | 0 | $0 | $124 | **$124** |
| Beta | 5 | $4.18 | $124 | **$128** |
| إطلاق مبكر | 20 | $16.72 | $124 | **$141** |
| نمو | 50 | $41.81 | $124 | **$166** |
| مئة متجر | 100 | $83.62 | $124 | **$208** |
| مئتان وخمسون | 250 | $209.05 | $135* | **$344** |
| خمسمائة | 500 | $418.10 | $144* | **$562** |

*عند نمو كبير: Railway قد يحتاج ترقية + Workers يتجاوز حدوده.

---

## القسم 5 — نقطة التعادل (محدَّثة)

```
التكلفة الثابتة الكاملة:  ~465 ريال/شهر
متوسط إيراد المشترك:       ~232 ريال/شهر

نقطة التعادل = 465 / 232 ≈ 2.0 مشترك

→ مشتركان فقط يُغطيان جميع التكاليف الثابتة!
```

**تأثير LangGraph على نقطة التعادل:** +0.06 مشترك (ضئيل جداً).

---

## القسم 6 — مقارنة التكلفة: مع LangGraph vs. بدونه

| البند | بدون LangGraph | مع LangGraph | الفرق |
|-------|---------------|-------------|-------|
| التكاليف الثابتة | ~$109/شهر | ~$124/شهر | +$15 |
| نقطة التعادل | ~1.9 مشترك | ~2.0 مشترك | +0.1 |
| قابلية التوسع | محدودة | 25+ ميزة | ✅ |
| ROI | MVP فقط | MVP + كل المستقبل | ✅ |

**الخلاصة:** +$15/شهر مقابل قدرة تطوير لا محدودة — قرار اقتصادي واضح.

---

## القسم 7 — التحقق والمراجعة

| الخدمة | الرابط |
|--------|--------|
| Cloudflare Workers | [pricing](https://developers.cloudflare.com/workers/platform/pricing/) |
| Supabase | [pricing](https://supabase.com/pricing) |
| OpenRouter | [pricing](https://openrouter.ai/pricing) |
| Railway | [pricing](https://railway.app/pricing) |
| LangSmith | [pricing](https://www.langchain.com/langsmith) |
| Resend | [pricing](https://resend.com/pricing) |
| Moyasar | [moyasar.com/ar](https://moyasar.com/ar) |

> **تاريخ التحقق:** أبريل 2026
> **USD/SAR:** 1 USD = 3.75 SAR (سعر ثابت)
