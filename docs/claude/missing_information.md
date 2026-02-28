# Missing Information Analysis — MoradBot MVP

**Date:** February 21, 2026
**Basis:** Full project exploration (Phases 1–3 code + all documentation)
**Scope:** MVP only — no Phase 5+ speculation

---

## Status: Answers Received (Feb 21, 2026)

| # | المعلومة | الحالة | القيمة / الملاحظة |
|---|----------|--------|-------------------|
| 1.2 | Supabase project-ref | ✅ مكتمل | `qvujnhkfqwqfzkkweylk` |
| 1.2 | Supabase migrations | ⚠️ باقي | لم تُطبَّق على remote بعد — مطلوب في Phase 4 |
| 1.1 | Salla App credentials | ⚠️ باقي | حساب موجود لكن التطبيق لم يُنشأ — ينشأ عند بدء Phase 4 |
| 1.3 | Salla Mode | ✅ قرار | **Custom Mode** — للتطوير والاختبار. الكود الحالي جاهز. |
| 1.5 | Cloudflare setup | ⚠️ باقي | حساب فقط — Workers + KV يُنشآن في Phase 4 |
| 2.2 | آلية إشعار التاجر | ✅ قرار | Dashboard + Email عبر **Resend** |

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Critical (blocks development) | 5 | ❌ Need user input |
| Important (improves quality) | 5 | ⚠️ Some inferred, some need input |
| Supplementary (nice to have) | 4 | 🔵 Can be decided by developer |

---

## 1. Critical — Blocks Development

These gaps block actual code execution or testing. Development cannot proceed without resolving them.

---

### 1.1 Salla App Credentials (Client ID + Secret)

**ما هي؟**
`SALLA_CLIENT_ID` و `SALLA_CLIENT_SECRET` المُسجَّلَين في Salla Partner Panel.

**لماذا نحتاجها؟**
كل OAuth flow في Phase 4 يعتمد عليهما. بدونهما:
- لا يمكن اختبار `GET /auth/salla/start`
- لا يمكن استلام access token أو refresh token
- لا يمكن استدعاء Salla Products API
- يستحيل إثبات أن Phase 4 يعمل

**أين توثَّق؟**
`apps/api/.dev.vars` (محلي، لا يُتتبع في git) + `wrangler secret put` (للإنتاج)

**هل يمكن استنتاجها؟**
لا. هذه بيانات حساب خارجية لا وجود لها في المشروع.

**تأثير غيابها:**
Phase 4 يمكن كتابته بالكامل لكن لا يمكن تشغيله أو اختباره. وإذا كان التطبيق غير مسجّل في Salla App Store أصلاً، يجب التسجيل أولاً قبل بدء Phase 4.

---

### 1.2 Supabase Project Ref + Personal Access Token

**ما هي؟**
- `project-ref`: الـ ID المكوّن من 20 حرفاً في رابط Supabase (`https://[project-ref].supabase.co`)
- `personal-access-token`: من قسم Account → Access Tokens في لوحة Supabase

**لماذا نحتاجها؟**
- `.mcp.json` يحتوي placeholders (`<project-ref>`, `<personal-access-token>`)
- بدونهما: Supabase MCP Server لا يتصل، يعني فقدان الأداة الأكثر فائدة للعمل مع schema
- `supabase db push` يحتاج project-ref للتطبيق على remote DB

**أين توثَّق؟**
- `project-ref` → `.mcp.json` (مباشرة)
- `personal-access-token` → `~/.claude/settings.local.json` أو متغير بيئة محلي (ليس في git)

**هل يمكن استنتاجها؟**
لا. موجودة في لوحة تحكم Supabase فقط.

**تأثير غيابها:**
Supabase MCP server لا يعمل (موثّق في diagnostics_issues.md). يمكن التطوير بدون MCP، لكن بجهد أكبر.

---

### 1.3 نوع تطبيق Salla: Easy Mode أم Custom Mode؟

**ما هي؟**
Salla لديها نوعان من التطبيقات:
- **Easy Mode:** تطبيق عام في Salla App Store، التثبيت عبر زر "تثبيت" في المتجر
- **Custom Mode:** تطبيق خاص، OAuth مباشر، لا يظهر في App Store

**لماذا نحتاجها؟**
هذا القرار يؤثر على:
1. هل نحتاج webhook لحدث `app.store.authorize`؟ (Easy Mode يرسله تلقائياً)
2. كيف يكتشف Salla أن التطبيق مُثبَّت؟ وهل يعيد توجيه للـ callback تلقائياً؟
3. هل يحتاج المستخدم لرابط OAuth يدوي؟
الكود الحالي (`auth.ts`) يُنفّذ OAuth flow يدوياً — يناسب Custom Mode أكثر.

**أين توثَّق؟**
`docs/claude/salla_api_reference.md` (قسم App Types) + `docs/claude/plans/phase-04_salla-client.md`

**هل يمكن استنتاجها؟**
جزئياً. PRD يقول "يُثبَّت من Salla App Store" → يشير لـ Easy Mode. لكن الكود الحالي لا يحتوي على webhook handler لحدث التثبيت. القرار يحتاج تأكيداً.

**تأثير غيابها:**
إذا اخترنا الـ mode الخطأ: OAuth يعمل في الاختبار ويفشل في الإنتاج. أو نبني webhook غير ضروري.

---

### 1.4 استراتيجية تشفير OAuth Tokens في قاعدة البيانات

**ما هي؟**
`stores` table تحتوي:
```sql
salla_access_token TEXT,   -- Will be encrypted at application level
salla_refresh_token TEXT,  -- Will be encrypted at application level
```
لكن لا يوجد في الكود أي utility للتشفير/فك التشفير.

**لماذا نحتاجها؟**
Phase 4 يحتاج حفظ الـ tokens في DB (auth.ts:56 عنده `TODO: Store tokens in database`). لكن قبل الكتابة يجب القرار:
- ما المكتبة؟ (CF Workers يدعم `crypto.subtle` Web Crypto API بدون dependencies)
- ما خوارزمية التشفير؟ (AES-256-GCM موصى به)
- ما format مفتاح التشفير؟ (32 byte → 64 hex char)
- أين يُولَّد ويُخزَّن الـ `ENCRYPTION_KEY`؟

**أين توثَّق؟**
`apps/api/src/lib/encryption.ts` (utility جديد) + `docs/claude/environment_variables.md`

**هل يمكن استنتاجها؟**
نعم بالكامل — CF Workers Web Crypto API + AES-256-GCM هو القرار الصحيح دون dependencies. يمكن للمطور تنفيذه بدون سؤال المستخدم. **هذا بند لا يحتاج سؤال المستخدم.**

**تأثير غيابها:**
tokens تُحفظ كنص صريح في DB — خرق أمني واضح وانتهاك لـ Rule 4.

---

### 1.5 حل Rate Limiter الموزَّع (Cloudflare KV vs Durable Objects)

**ما هي؟**
Rate limiter الحالي (`rate-limit.ts:6`) يستخدم `Map` في الذاكرة:
```typescript
const requestCounts = new Map<string, { count: number; resetAt: number }>();
```
هذا لا يعمل في Cloudflare Workers الموزَّعة — كل instance لديه ذاكرته المستقلة.

**لماذا نحتاجها؟**
قرار يؤثر على `wrangler.toml` والبنية:
- **KV Namespace:** أبسط، eventual consistency، قد يسمح ببعض الطلبات الزائدة
- **Durable Objects:** دقيق تماماً، strong consistency، أعقد وأغلى

**أين توثَّق؟**
`apps/api/wrangler.toml` (إضافة KV binding) + `apps/api/src/middleware/rate-limit.ts` (إعادة الكتابة)

**هل يمكن استنتاجها؟**
نعم. **لـ MVP: KV هو القرار الصحيح** — بسيط، وفرق الدقة التقني مقبول. لكن يحتاج إنشاء KV namespace في Cloudflare Dashboard.

**هل KV namespace موجود بالفعل؟ هذا يحتاج تأكيد المستخدم.**

**تأثير غيابها:**
Rate limiting لا يعمل فعلياً في الإنتاج — يمكن تجاوزه بسهولة.

---

## 2. مهمة — تحسّن جودة التطوير

هذه المعلومات غير مانعة لكنها ضرورية لإنتاج كود صحيح من المرة الأولى.

---

### 2.1 نص إفصاح الذكاء الاصطناعي (عربي)

**ما هي؟**
الرسالة الأولى التي يرسلها البوت لكل زيارة جديدة قبل أي رد. Rule 5 يستوجبها، لكن النص الفعلي غير موجود في أي وثيقة.

**لماذا نحتاجها؟**
ستُضمَّن حرفياً في widget code (Phase 5). نص خاطئ أو غائب = انتهاك PDPL.

**أين توثَّق؟**
`docs/claude/pdpl_compliance_checklist.md` + `apps/widget/src/constants.ts`

**هل يمكن استنتاجها؟**
يمكن كتابة نص مقترح، لكن يجب موافقة المستخدم لأنه قرار قانوني/منتج.

**نص مقترح (للمراجعة):**
> "أهلاً! أنا مساعد ذكاء اصطناعي لهذا المتجر. أستطيع مساعدتك في الأسئلة الشائعة حول المنتجات والشحن والدفع. محادثاتك قد تُحفظ لتحسين الخدمة. هل تريد المتابعة؟"

---

### 2.2 آلية إشعار التاجر (80% / 100%)

**ما هي؟**
SRD يذكر: "عند 80% من الحد: إشعار للتاجر" و"عند 100%: إشعار + رسالة بديلة في الواجهة". لكن لا يوجد أي توضيح للقناة:
- هل عبر email؟ (أي خدمة؟ SendGrid؟ Resend؟)
- هل لوحة التحكم فقط؟ (real-time notification)
- هل كلاهما؟

**لماذا نحتاجها؟**
القرار يؤثر على:
- هل نحتاج email service dependency؟
- هل نحتاج Supabase Realtime subscription في Dashboard؟
- Phase 4 triggers يُكتَب بناءً على هذا القرار

**أين توثَّق؟**
`docs/claude/escalation_notification_spec.md` + `apps/api/src/routes/chat.ts`

**هل يمكن استنتاجها؟**
جزئياً. MVP الأبسط: لوحة تحكم فقط (badge/notification) + إمكانية إضافة email لاحقاً. لكن القرار يحتاج تأكيد.

---

### 2.3 هيكل أسئلة FAQ الأساسية (5 أسئلة)

**ما هي؟**
PRD يذكر أن التاجر يُجيب على 5 أسئلة أساسية عند الإعداد. قاعدة البيانات لديها جدول `faq_entries` بحقل `category` لكن لا تعريف للـ 5 أسئلة:
1. سياسة الشحن
2. طرق الدفع
3. سياسة الإرجاع
4. معلومات التواصل
5. سؤال مخصص

النصوص الحرفية للأسئلة (بالعربية) تؤثر على Dashboard UI (Phase 5) وكيف يُنظّم AI context.

**أين توثَّق؟**
`apps/dashboard/src/lib/faq-questions.ts` (ثوابت) + `docs/claude/onboarding_guide.md`

**هل يمكن استنتاجها؟**
نعم بشكل مقبول للـ MVP. يمكن للمطور كتابة أسئلة عربية معيارية. **هذا لا يحتاج سؤال المستخدم.**

---

### 2.4 نطاق الـ API في الإنتاج (Production Domain)

**ما هي؟**
`environment_variables.md` يذكر `https://api.moradbot.com/auth/salla/callback` كـ redirect URI للإنتاج.
لكن:
- هل `api.moradbot.com` هو الدومين الفعلي؟
- هل الدومين مسجَّل ومربوط بـ Cloudflare؟
- ما دومين Dashboard؟

**لماذا نحتاجها؟**
- CORS config (`cors.ts:4`) يسمح لـ `localhost:3000` و `*.salla.sa` فقط — لا يسمح لدومين Dashboard إذا كان غير `salla.sa`
- `SALLA_REDIRECT_URI` للإنتاج يجب تسجيله في Salla Partner Panel بالضبط

**أين توثَّق؟**
`apps/api/src/middleware/cors.ts` + `wrangler.toml` + `docs/claude/environment_variables.md`

**هل يمكن استنتاجها؟**
لا. موجودة عند المستخدم فقط.

---

### 2.5 Bot System Prompt (الـ Prompt الرئيسي للذكاء الاصطناعي)

**ما هي؟**
النص الكامل للـ system prompt الذي سيُرسَل لـ OpenRouter/Gemini مع كل محادثة.

**لماذا نحتاجها؟**
يحدد سلوك البوت بالكامل. يؤثر على:
- دقة الردود العربية
- منطق "لا تعرف" → طلب توضيح → تصعيد
- حدود ما يُجيب عنه البوت (FAQ + منتجات فقط)

**أين توثَّق؟**
`packages/ai-orchestrator/src/prompts/system.ts` + تحت حماية Rule CLAUDE.md (يحتاج `/ultra-think`)

**هل يمكن استنتاجها؟**
نعم — السلوك الكامل موثّق في PRD/SRD/Architecture. يمكن كتابة system prompt مقترح. لكن يجب مراجعة المستخدم قبل الاعتماد عليه في الإنتاج.

---

## 3. تكميلية — Nice to Have

هذه يمكن للمطور تحديدها بدون مدخلات المستخدم، لكن مدخلات المستخدم تُحسّن النتيجة.

---

### 3.1 رسائل الخطأ العربية في Widget

**ما هي؟** النصوص الدقيقة لـ 3 حالات:
- حالة `error`: "عذراً، حدث خطأ. يرجى المحاولة مجدداً."
- حالة `quota_exceeded`: "وصل المتجر للحد الأقصى من الردود هذا الشهر. تواصل مع المتجر مباشرة."
- حالة `timeout`: "استغرق الرد وقتاً طويلاً. يرجى المحاولة مجدداً."

**يمكن استنتاجها؟** نعم. المطور يكتبها.

---

### 3.2 OpenRouter API Key

**ما هي؟** مفتاح OpenRouter للوصول لـ Gemini 2.0 Flash.

**متى تحتاجها؟** Phase 5 (AI Orchestrator) — ليس Phase 4.

**يمكن استنتاجها؟** لا. لكن ليست ملحّة الآن.

---

### 3.3 ألوان Widget الافتراضية ومكانه

**ما هي؟** قاعدة البيانات تحتوي على defaults:
```json
{"color": "#2563eb", "position": "bottom-right"}
```
هل هذه هي القرارات النهائية؟ أم يريد المستخدم تغييرها؟

**يمكن استنتاجها؟** نعم — defaults موجودة في DB schema. يمكن البناء عليها.

---

### 3.4 اسم KV Namespace للـ Rate Limiter

**ما هي؟** اسم Cloudflare KV namespace الذي سيُنشأ لتخزين rate limit counters.

**يمكن استنتاجها؟** نعم. `MORADBOT_RATE_LIMIT` اسم مناسب. لكن يحتاج المستخدم إنشاءه في لوحة Cloudflare.

---

## جدول ملخص: القرارات المطلوبة

| # | المعلومة | يمكن استنتاجها؟ | يحتاج المستخدم؟ | متى تُحتاج؟ |
|---|----------|-----------------|-----------------|------------|
| 1.1 | Salla Client ID + Secret | ❌ لا | ✅ نعم | Phase 4 اختبار |
| 1.2 | Supabase project-ref + token | ❌ لا | ✅ نعم | الآن (MCP) |
| 1.3 | Easy Mode أم Custom Mode؟ | ⚠️ جزئياً | ✅ نعم | Phase 4 تصميم |
| 1.4 | استراتيجية التشفير (AES-256-GCM) | ✅ نعم | ❌ لا | Phase 4 كود |
| 1.5 | KV vs Durable Objects | ✅ KV | ⚠️ إنشاء namespace فقط | Phase 4 fix |
| 2.1 | نص إفصاح AI (عربي) | ⚠️ مقترح | ✅ موافقة | Phase 5 widget |
| 2.2 | آلية إشعار 80%/100% | ⚠️ جزئياً | ✅ نعم | Phase 4/5 |
| 2.3 | أسئلة FAQ الـ 5 | ✅ نعم | ❌ لا | Phase 5 |
| 2.4 | Production domain | ❌ لا | ✅ نعم | قبل الإنتاج |
| 2.5 | System prompt | ✅ مقترح | ✅ موافقة | Phase 5 |
| 3.1 | رسائل خطأ عربية | ✅ نعم | ❌ لا | Phase 5 |
| 3.2 | OpenRouter API Key | ❌ لا | ✅ نعم | Phase 5 |
| 3.3 | ألوان Widget | ✅ من DB | ❌ لا | Phase 5 |
| 3.4 | KV namespace name | ✅ نعم | ❌ (إنشاء فقط) | Phase 4 fix |

---

*آخر تحديث: 21 فبراير 2026*
