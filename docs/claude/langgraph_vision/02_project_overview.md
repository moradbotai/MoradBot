# مراد بوت — توثيق المشروع الكامل
**المؤسسة:** مؤسسة محمد إبراهيم الجهني
**الإصدار:** 2.1 (محدَّث بقرار LangGraph — أبريل 2026)
**الحالة:** معتمد

> ⚡ **التحديث الجوهري:** Phase 5 يعتمد الآن **LangGraph Python** كمحرك الذكاء الاصطناعي — خدمة مستقلة على Railway/Fly.io تتصل بـ Hono API.

---

## الهدف

توثيق شامل لمشروع مراد بوت يغطي البنية والنطاق والقرارات التقنية والتجارية.

---

## 1. نظرة عامة على المشروع

**اسم المشروع:** مراد بوت
**نوع المشروع:** منصة B2B SaaS للدعم الآلي بالذكاء الاصطناعي
**السوق المستهدف:** تجار التجارة الإلكترونية السعوديون على منصة Salla
**حجم التاجر المستهدف:** 30–300 طلب/شهر
**وضع الإطلاق الأولي:** Beta مغلق (تجار مختارون فقط)

---

## 2. الهدف الأساسي (MVP)

> أتمتة الردود على الأسئلة الشائعة داخل متاجر Salla باستخدام مساعد AI.

### الأهداف الرئيسية
- توفير 5 ساعات يومياً من الردود اليدوية
- توفير ردود آلية على مدار 24/7
- تقديم إجابات فورية مبنية على بيانات المتجر الموثّقة

لا منطق مبيعات. لا upselling. لا رسائل استباقية.
أتمتة FAQ فقط في MVP.

---

## 3. نطاق MVP

### 3.1 مدرج في MVP

#### أتمتة FAQ (5 فئات)
- توفر المنتج وسعره (عبر Salla Products API)
- مدة الشحن (FAQ يقدمه التاجر — فئة `shipping`)
- طرق الدفع (FAQ يقدمه التاجر — فئة `payment`)
- سياسة الإرجاع (FAQ يقدمه التاجر — فئة `returns`)
- الأسئلة العامة الأخرى (FAQ يقدمه التاجر — فئة `general`)

#### تكامل Salla
- ربط OAuth (Custom Mode) — متجر واحد لكل تاجر
- صلاحية قراءة المنتجات فقط (`GET /products`)

#### Chat Widget
- مضمّن داخل متجر Salla — عربي فقط
- نص فقط (لا ملفات، لا صور، لا صوت في MVP)
- 7 حالات: `closed` → `open` → `typing` → `response` → `escalation` → `error` → `limit-reached`

#### لوحة تحكم التاجر (أساسية)
- عرض المحادثات (قراءة فقط)
- إحصائيات أساسية + زر On/Off
- إدارة FAQ + معلومات الاشتراك

#### نظام التصعيد
- 3 محاولات توضيح فاشلة تُفعّل التصعيد
- التصعيد يظهر في لوحة التحكم

### 3.2 مستبعد من MVP

تتبع الطلبات، تحليلات متقدمة، دعم الإنجليزية، تكامل WhatsApp، رفع الملفات، رسائل استباقية، دعم متعدد المتاجر.

---

## 4. نموذج الأعمال

| الخطة | الردود/شهر | السعر |
|-------|-----------|-------|
| الانطلاق | 1,000 رد | 97 SAR |
| النمو | 3,000 رد | 197 SAR |
| المتمكّن | 8,000 رد | 449 SAR |

- إشعار عند 80% و100% من الحد
- لا ترقية تلقائية ولا رسوم إضافية

---

## 5. سياسة البيانات

- البيانات الشخصية مشفرة (AES-256-GCM)
- تتبع الزائر عبر cookie — بعد موافقة صريحة فقط
- لا بيانات شخصية بدون موافقة (PDPL)
- حذف البيانات الشخصية 30-90 يوم بعد الإلغاء

---

## 6. العزل متعدد المستأجرين

- تسامح صفري لتسريب البيانات
- جميع الاستعلامات تتضمن `store_id` إلزامياً
- RLS على مستوى قاعدة البيانات: `auth.uid() = store_id`

---

## 7. البنية التقنية المُحدَّثة

### Stack الكامل (بعد قرار LangGraph)

| المكوّن | التقنية | ملاحظة |
|--------|---------|--------|
| API Runtime | Cloudflare Workers + Hono v4 (TypeScript) | يبقى كما هو |
| قاعدة البيانات والمصادقة | Supabase PostgreSQL + RLS | + LangGraph Checkpointing |
| **AI Orchestrator** | **LangGraph Python** | **جديد — Phase 5** |
| AI Provider | OpenRouter → Gemini 2.0 Flash | عبر LangGraph |
| AI Fallback | GPT-4o Mini | عبر LangGraph |
| Chat Widget | Preact + Vite (هدف: <50KB gzipped) | |
| لوحة التحكم | Next.js 15 App Router + Cloudflare Pages | |
| مدير الحزم | pnpm + Turborepo | |
| Linting/Formatting | Biome | |
| Rate Limiting | Cloudflare KV | |
| التشفير | AES-256-GCM | |
| البريد الإلكتروني | Resend | |
| Analytics | PostHog (Browser + posthog-node) | مُفعَّل |
| **Python Service Hosting** | **Railway / Fly.io** | **جديد — Phase 5** |
| **AI Tracing** | **LangSmith** | **جديد — Phase 5** |

### المعمارية ثلاثية الطبقات

```
Cloudflare Workers (Hono)
    ↕ HTTP (POST /api/chat)
LangGraph Python Service (Railway/Fly.io)
    ↕ TCP + PostgresSaver
Supabase (PostgreSQL + RLS)
```

### قاعدة البيانات (13 جدول فعلي)

`plans`, `stores`, `subscriptions`, `faq_entries`, `product_snapshots`, `visitor_sessions`, `tickets`, `messages`, `escalations`, `usage_events`, `consent_logs`, `audit_logs`, `waitlist_submissions`

> ملاحظة: `bot_configurations` مذكور في الوثائق الأصلية لكن **غير موجود في migrations**. `usage_events` هو اسم الجدول الفعلي (وليس `usage_tracking`).

---

## 8. حالة التنفيذ (أبريل 2026)

| المرحلة | الحالة | ما تم بناؤه |
|---------|--------|------------|
| Phase 1 | ✅ مكتملة | Turborepo، Biome، الحزم الأساسية |
| Phase 2 | ✅ مكتملة | 14 جدول، 8 migrations، RLS |
| Phase 3 | ✅ مكتملة | Hono API: 16 endpoint، middleware |
| Phase 3.5 | ✅ مكتملة | Landing page + waitlist + PostHog |
| Phase 4 | 🔜 التالية | Salla Client Package (OAuth + GET /products) |
| Phase 5 | ⏳ معلّقة | **LangGraph AI Service + Widget UI** |
| Phase 6+ | ⏳ معلّقة | Enhanced Agents، Dashboard، Billing |

---

## 9. نموذج AI (مُحدَّث)

| المكوّن | التقنية |
|--------|---------|
| Framework | LangGraph Python (خدمة مستقلة) |
| Provider | OpenRouter |
| النموذج الأساسي | google/gemini-2.0-flash |
| Fallback | openai/gpt-4o-mini |
| Checkpointing | PostgresSaver → Supabase |
| Tracing | LangSmith |

---

## 10. فلسفة المنتج

MVP = أداة مُركّزة، حادة، أحادية الغرض.
مع بنية تحتية (LangGraph) تتيح التوسع لـ 25+ ميزة مستقبلية بدون إعادة بناء.

---

## القرارات المعمارية المحفوظة

- تسامح صفري لأي استعلام بدون `store_id` — خطأ فادح يوقف النظام
- OAuth Custom Mode فقط — لا Partner Mode في MVP
- الردود دائماً بالعربية — فرض على مستوى system prompt
- لا نشر تلقائي — كل نشر يمر بـ Pre-Deploy Checklist يدوياً
- الأسرار في Cloudflare Secrets فقط — لا في الكود أو `.env`
- **LangGraph Python على خدمة مستقلة (لا داخل Workers)** — قرار أبريل 2026
