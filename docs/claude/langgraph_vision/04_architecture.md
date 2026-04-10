# مراد بوت — قرارات المعمارية الموسعة والتشغيلية
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.1 (محدَّث أبريل 2026 — يشمل LangGraph Hybrid Architecture)
**الحالة:** معتمد

> ⚡ **التحديث الجوهري:** إضافة قسم كامل عن **LangGraph Hybrid Architecture** — القرار الأهم في تاريخ المشروع.

---

## 1. نشر الويدجت داخل سلة

- يثبّت التاجر مراد بوت من متجر تطبيقات سلة.
- يظهر الويدجت تلقائياً في المتجر بعد التثبيت.
- لا يظهر الويدجت في صفحة الدفع (`/checkout/*`).

---

## 2. استراتيجية مزامنة بيانات المنتجات

**القرار: مزامنة دورية (لا تحديث لحظي)**

| الخطة | التكرار |
|-------|---------|
| الانطلاق | كل 24 ساعة |
| النمو | كل 6 ساعات |
| المتمكّن | كل ساعة |

- سلوك البوت للبيانات الديناميكية: يضيف إخلاء مسؤولية "حسب آخر تحديث..."
- القيم مُعرَّفة في `plans.sync_frequency_hours`

---

## 3. Rate Limiting والحماية من الإساءة

**آلية الإنفاذ الحالية (MVP):** In-memory Map داخل Worker على مستويين:
- **الزائر:** 20 رسالة/دقيقة
- **المتجر:** 3000 رد بوت/ساعة

> ⚠️ **ملاحظة تقنية:** الكود الحالي في `apps/api/src/middleware/rate-limit.ts` يستخدم `Map` في الذاكرة (وليس Cloudflare KV). هذا يعني أن الـ limits لا تُشارك بين Worker instances مختلفة. TODO موثَّق في الكود للانتقال إلى KV أو Durable Objects في Phase 4 قبل الإنتاج.

---

## 4. المراقبة والتنبيهات

**ما يُراقَب:**
- استخدام ردود البوت لكل متجر
- تكلفة التوكن (عبر OpenRouter + LangSmith)
- معدل الأخطاء وزمن الاستجابة
- معدل فشل النماذج

**التنبيهات:** Resend (البريد الإلكتروني فقط في MVP)

---

## 5. سياسة النسخ الاحتياطي

- الاعتماد على النسخ الاحتياطية التلقائية لـ Supabase
- **RPO:** ≤ 24 ساعة | **RTO:** ≤ 4 ساعات

---

## 6. سياسة الاستجابة للحوادث

الحالات التي تستوجب الإيقاف الفوري:
- تسريب بيانات بين متاجر مختلفة
- خطأ منطقي جسيم
- إساءة استخدام الموارد
- اختراق أمني

---

## 7. بيئات التطوير والإنتاج

| البيئة | الوصف |
|--------|-------|
| Development | Wrangler dev + Supabase local + LangGraph local |
| Production | CF Workers + Supabase Cloud + Railway/Fly.io |

لا بيئة Staging — قرار مؤكد ونهائي.

**مهلة Worker:** 30 ثانية (لهذا LangGraph على خدمة منفصلة).

---

## 8. إدارة المفاتيح والأسرار

جميع البيانات الحساسة في Cloudflare Secrets:
```bash
wrangler secret put AI_SERVICE_URL     # URL خدمة LangGraph
wrangler secret put AI_SERVICE_KEY     # مفتاح المصادقة بين Workers ↔ LangGraph
```

---

## 9. ⭐ LangGraph Hybrid Architecture (قرار أبريل 2026)

### لماذا لا يعمل LangGraph داخل Workers؟

| متطلب LangGraph | Cloudflare Workers | الحالة |
|----------------|-------------------|--------|
| عمليات > 30 ثانية | حد 30 ثانية CPU | ❌ |
| PostgreSQL مباشر (TCP) | لا يدعم TCP | ❌ |
| Python runtime | TypeScript/JS فقط | ❌ |
| Background agents | Request-response فقط | ❌ |
| 128MB+ RAM | 128MB محدودة | ❌ |

### المعمارية الهجينة المعتمدة

```
┌─────────────────────────────────────────────┐
│      CLOUDFLARE WORKERS (Hono API)           │
│  ✅ HTTP routing  ✅ JWT Auth               │
│  ✅ Rate limiting ✅ FAQ CRUD               │
│  ✅ Stats/Tickets ✅ Escalations            │
└──────────────────┬──────────────────────────┘
                   │ POST /api/chat
                   │ { storeId, message, context }
                   ▼
┌─────────────────────────────────────────────┐
│    LANGGRAPH PYTHON SERVICE (Railway)        │
│                                             │
│  FAQ Agent     ← Phase 5 (MVP)             │
│  CS Agent      ← Phase 6 (Enhanced)        │
│  Upsell Agent  ← Phase 7 (Advanced)        │
│  Voice Agent   ← Phase 7                   │
└──────────────────┬──────────────────────────┘
                   │ PostgresSaver + Store
                   ▼
┌─────────────────────────────────────────────┐
│         SUPABASE (PostgreSQL)                │
│  ✅ Business data  ✅ RLS policies          │
│  ✅ Checkpointing  ✅ Cross-thread memory   │
└─────────────────────────────────────────────┘
```

### سير طلب المحادثة

```
1. Visitor → Widget → POST /api/chat (Hono)
2. Hono: JWT auth + rate limit check
3. Hono → LangGraph Service: { storeId, message, context }
4. LangGraph: validate → retrieve → classify → generate → save
5. LangGraph → Hono: { bot_response, escalated, ticket_id }
6. Hono → Widget: JSON response
```

**الزمن المتوقع:** 10-50ms (Hono) + 1,000-2,500ms (LangGraph+LLM) = P50 ~1.5s

### الاتصال بين Hono و LangGraph

```typescript
// apps/api/src/lib/ai-service.ts
export async function callAIService(env: Env, payload: ChatPayload) {
  const resp = await fetch(env.AI_SERVICE_URL + "/chat", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.AI_SERVICE_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(8000) // 8s timeout
  });
  return resp.json();
}
```

### LangGraph Checkpointing عبر Supabase

```python
# packages/ai-orchestrator/src/checkpointer.py
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

def get_checkpointer(supabase_conn_string: str) -> PostgresSaver:
    conn = psycopg.connect(supabase_conn_string)
    return PostgresSaver(conn)
```

---

## 10. نطاق المنتج في MVP

**الفلسفة:** أداة وحيدة الغرض — الرد الذكي على الأسئلة الشائعة.
**البنية:** مصممة للتوسع → LangGraph يتيح 25+ ميزة مستقبلية بدون إعادة بناء.

---

## 11. بيانات المحادثات وعزل المتاجر

- **العزل:** `store_id` في كل استعلام (Rule 3)
- **التشفير:** البيانات الشخصية الحساسة مشفرة AES-256-GCM
- **الحذف:** بيانات المحادثات تُحذف 30 يوماً من إلغاء الاشتراك

---

## القرارات المعمارية المحفوظة
- Rate Limiting حالياً: in-memory Map (MVP) — TODO: ترقية إلى KV في Phase 4: زائر (20/دقيقة) + متجر (3000/ساعة)
- مزامنة المنتجات دورية حسب خطة الاشتراك
- الأسرار في Cloudflare Secrets فقط (Rule 4)
- **LangGraph Python على خدمة مستقلة — لا داخل Workers** (قرار أبريل 2026)
- **Supabase PostgresSaver للـ Checkpointing** (قرار أبريل 2026)
- لا Staging — بيئتان: Development وProduction
- CORS: `localhost` في Dev، `*.salla.sa` في Production
- إيقاف فوري عند انتهاك عزل البيانات (Rule 3)
