> ⚠️ **PARTIAL REFERENCE** — تعكس هذه الخطة ما قبل التنفيذ. راجع الملاحظات أدناه قبل الاستخدام.
>
> **الانحرافات الرئيسية:**
> - **Phase 0 (Tooling):** OBSOLETE — راجع `docs/claude/tools_report_v2.md`
> - **Phase 1 (Environment):** مكتملة — مع Biome بدلاً من ESLint+Prettier
> - **Phase 2 (DB):** مكتملة — 12 جدولاً لا 10، migration naming مختلف
> - **Phase 3 في هذه الخطة = Phase 4 الفعلية** (Salla OAuth)
> - **Phase 3 الفعلية (API Foundation):** 16 endpoint، Hono v4 — **غائبة من هذه الخطة كلياً**
> - **Widget:** Preact + Vite (لا Vanilla TypeScript)
> - **Dashboard:** Next.js 15 App Router (لا React + Vite)

# 🗺️ خطة تنفيذ MoradBot — Claude Code Implementation Plan
**الإصدار:** 1.0.0 | **التاريخ:** فبراير 2026 | **المدة الإجمالية:** 40 يوم عمل

---

## 📊 ملخص إحصائي للخطة

| المعيار | القيمة |
|---------|--------|
| **إجمالي المراحل** | 11 مرحلة (0 → 10) |
| **إجمالي الخطوات** | 42 خطوة |
| **المدة الإجمالية** | 40 يوم عمل |
| **نقاط التفتيش** | 11 Checkpoint |
| **قرارات تتطلب ultra-think** | 8 قرارات |
| **إجمالي الاختبارات المتوقعة** | 150+ اختبار |
| **هدف تغطية الكود** | ≥ 80% |
| **هدف الأداء P50** | ≤ 1.5 ثانية |
| **هدف الأداء P95** | ≤ 3.0 ثانية |
| **هدف Uptime** | ≥ 99% شهرياً |

---

## 🏗️ خارطة الطريق الكاملة

```
المرحلة 0:  إعداد بيئة Claude Code          [يوم 1]
المرحلة 1:  هيكل المشروع والبنية            [يوم 2-3]
المرحلة 2:  قاعدة البيانات — Supabase       [يوم 4-6]
المرحلة 3:  Salla OAuth والتثبيت            [يوم 7-9]
المرحلة 4:  Chat Widget                     [يوم 10-14]
المرحلة 5:  AI Orchestrator                 [يوم 15-18]
المرحلة 6:  Product Sync Service            [يوم 19-21]
المرحلة 7:  Merchant Dashboard              [يوم 22-26]
المرحلة 8:  Escalation & Billing Metering   [يوم 27-30]
المرحلة 9:  الأمان والاختبار الشامل         [يوم 31-35]
المرحلة 10: Production Deployment           [يوم 36-40]
```

---

## المرحلة 0: إعداد بيئة Claude Code
**المدة:** يوم 1 | **الأدوات:** health-check, setup-development-environment

### الهدف
بيئة Claude Code جاهزة بالكامل مع 975+ مكوّن مُثبَّت وـ CLAUDE.md محكم.

### الخطوات

#### الخطوة 0.1 — إنشاء هيكل .claude الأساسي
```bash
cd ~/moradbot
mkdir -p .claude/{agents,commands,skills,hooks,mcp}
```

#### الخطوة 0.2 — تثبيت Skills الكبيرة (antigravity)
```bash
npx antigravity-awesome-skills --claude
# النتيجة: 860+ Skill في .claude/skills/
```

#### الخطوة 0.3 — تثبيت جميع مكونات aitmpl
```bash
npx claude-code-templates@latest \
  --agent [31 وكيل] \
  --command [35 أمر] \
  --hook [16 خطاف] \
  --mcp [20 خادم] \
  --skill [3 مهارات] \
  --setting "mcp/enable-all-project-servers,telemetry/disable-telemetry"
# انظر الوثيقة 01 للأمر الكامل
```

#### الخطوة 0.4 — إنشاء CLAUDE.md الرئيسي
```markdown
# MoradBot — CLAUDE.md
المشروع: B2B SaaS تطبيق سلة للرد التلقائي على FAQ
Stack: TypeScript + Cloudflare Workers + Supabase + OpenRouter
البيئات: Development + Production فقط (لا Staging)
النشر: يدوي فقط

## قواعد لا تُخترق (6 قواعد)
1. MVP = FAQ Automation فقط
2. Salla = قراءة فقط
3. عزل البيانات صفري التسامح
4. الأسرار في Cloudflare Secrets فقط
5. عربي فقط في Chat Widget
6. لا نشر تلقائي أبداً
```

#### الخطوة 0.5 — التحقق من الإعداد
```bash
npx claude-code-templates@latest --health-check
```

### ✅ معايير الاكتمال
- [ ] جميع مكونات aitmpl مُثبَّتة
- [ ] antigravity-awesome-skills مُثبَّت (860+ skill)
- [ ] CLAUDE.md موجود وشامل
- [ ] Health Check يعطي نتيجة إيجابية
- [ ] جميع MCP Servers تستجيب

---

## المرحلة 1: هيكل المشروع والبنية
**المدة:** يوم 2-3 | **الأدوات:** ultra-think, architecture-scenario-explorer, setup-development-environment

### الهدف
Monorepo احترافي بـ Turborepo جاهز للتطوير.

### الخطوات

#### الخطوة 1.1 — تصميم البنية المعمارية
```bash
# أمر Claude Code
/ultra-think
>> "صمّم بنية Monorepo لـ MoradBot:
   - Cloudflare Workers (TypeScript)
   - Supabase
   - Chat Widget (Vanilla TS)
   - Dashboard (React)
   مع Turborepo وshared packages"
```

#### الخطوة 1.2 — هيكل المجلدات المستهدف
```
moradbot/
├── .claude/               # إعدادات Claude Code (860+ components)
├── apps/
│   ├── api/               # Cloudflare Worker — Backend API
│   │   ├── src/
│   │   │   ├── routes/    # API Route Handlers
│   │   │   ├── services/  # Business Logic
│   │   │   ├── middleware/ # Auth, Rate Limiting, CORS
│   │   │   └── index.ts   # Entry Point
│   │   └── wrangler.toml
│   ├── widget/            # Chat Widget (Vanilla TypeScript → JS)
│   │   ├── src/
│   │   │   ├── ui/        # DOM Components
│   │   │   ├── api/       # API Client
│   │   │   ├── state/     # State Management
│   │   │   └── index.ts   # Entry Point
│   │   └── wrangler.toml
│   └── dashboard/         # Merchant Dashboard (React + Cloudflare Pages)
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   └── hooks/
│       └── vite.config.ts
├── packages/
│   ├── shared/            # TypeScript Types مشتركة
│   │   └── src/types/
│   ├── ai-orchestrator/   # منطق AI (قابل للاختبار المستقل)
│   │   └── src/
│   └── salla-client/      # Salla API Client
│       └── src/
├── supabase/
│   ├── migrations/        # SQL Migration Files
│   ├── functions/         # Edge Functions (إن احتجنا)
│   └── seed/              # بيانات تجريبية
├── docs/                  # الوثائق الأصلية (MRD/BRD/PRD/SRD)
├── Open_source_projects/  # المشاريع المرجعية
├── package.json           # Monorepo Root
├── turbo.json             # Turborepo Config
└── tsconfig.base.json     # TypeScript Base Config
```

#### الخطوة 1.3 — إعداد Monorepo
```bash
/setup-development-environment
>> "TypeScript Monorepo مع:
   - Turborepo للـ build pipeline
   - Cloudflare Workers للـ API والـ Widget
   - Supabase CLI
   - Vitest للاختبارات
   - Biome (Linting + Formatting)"
```

#### الخطوة 1.4 — قراءة المشاريع المرجعية
```bash
/directory-deep-dive
>> "Open_source_projects/adk-samples-main/typescript/agents/customer_service"
# استخلاص أفضل الممارسات للـ Agent Pattern
```

### ✅ معايير الاكتمال
- [ ] Monorepo يعمل بـ `turbo build`
- [ ] TypeScript يُترجم بدون أخطاء
- [ ] CLAUDE.md يصف الهيكل الكامل
- [ ] Supabase CLI متصل

---

## المرحلة 2: قاعدة البيانات — Supabase
**المدة:** يوم 4-6 | **الأدوات:** design-database-schema, supabase-migration-assistant, supabase-type-generator, supabase-security-audit

### الهدف
Schema كامل مع RLS محكم وعزل تام بين المتاجر.

### الخطوات

#### الخطوة 2.1 — تصميم Schema من وثائق المشروع
```bash
/ultra-think
/design-database-schema
>> "راجع docs/morad_bot_system_requirements_document_srd_v_1.md
   وصمّم Schema كاملاً يشمل:
   1. stores + subscriptions + plans
   2. conversations + tickets + messages
   3. faq_entries + product_snapshots
   4. escalations + usage_events
   5. consent_logs + audit_logs
   مع RLS Policy على كل جدول"
```

#### الخطوة 2.2 — الجداول الرئيسية (10 جداول)

| # | الجدول | الغرض | الحقول الحرجة |
|---|--------|-------|---------------|
| 1 | `stores` | المتاجر المشتركة | id, salla_store_id, oauth_token_encrypted, is_active |
| 2 | `plans` | باقات الاشتراك | id, name, bot_replies_limit, sync_frequency_hours |
| 3 | `subscriptions` | اشتراكات نشطة | store_id, plan_id, current_cycle_replies, billing_cycle_start |
| 4 | `tickets` | تذاكر المحادثة | id, store_id, visitor_id, status, topic |
| 5 | `messages` | رسائل المحادثة | id, ticket_id, store_id, role, content, is_bot_reply |
| 6 | `visitor_sessions` | جلسات الزوار | id, store_id, cookie_id, has_consent |
| 7 | `faq_entries` | إجابات FAQ | id, store_id, category, question, answer |
| 8 | `product_snapshots` | لقطات المنتجات | id, store_id, salla_product_id, data_json, synced_at |
| 9 | `escalations` | حالات التصعيد | id, ticket_id, contact_email_encrypted, contact_phone_encrypted |
| 10 | `audit_logs` | سجل التدقيق | id, actor_type, action, target_table, created_at |

#### الخطوة 2.3 — إنشاء Migration Files
```bash
/supabase-migration-assistant
>> "أنشئ 5 migration files منظّمة:
   001_core_stores.sql         — المتاجر والباقات
   002_conversation_system.sql  — Tickets والرسائل
   003_faq_and_products.sql    — FAQ والمنتجات
   004_escalation_billing.sql  — التصعيد والفوترة
   005_audit_security.sql      — التدقيق والـ RLS"
```

#### الخطوة 2.4 — توليد TypeScript Types
```bash
/supabase-type-generator
>> "ولّد types في packages/shared/src/types/database.ts"
```

#### الخطوة 2.5 — التدقيق الأمني للـ Schema
```bash
/supabase-security-audit
>> "تحقق أن:
   - RLS مُفعَّل على جميع الجداول
   - لا يمكن الوصول لأي سجل بدون store_id صحيح
   - الحقول المشفّرة محمية من القراءة المباشرة
   - audit_logs لا يمكن حذفها أو تعديلها"
```

#### الخطوة 2.6 — اختبارات العزل (أهم اختبار في المشروع)
```bash
/generate-test-cases
>> "اختبارات عزل البيانات:
   - متجر A يحاول قراءة رسائل متجر B → مرفوض
   - متجر A يحاول كتابة في FAQ متجر B → مرفوض
   - تحقق من 10 سيناريوهات اختراق مختلفة"
```

### ✅ معايير الاكتمال
- [ ] 10 جداول مُنشأة بنجاح
- [ ] RLS مُفعَّل على جميع الجداول
- [ ] TypeScript Types مُولَّدة
- [ ] جميع اختبارات العزل تنجح
- [ ] Migration Files منظّمة ومعلّقة

---

## المرحلة 3: Salla OAuth والتثبيت
**المدة:** يوم 7-9 | **الأدوات:** ultra-think, backend-architect, auth-implementation-patterns

### الهدف
تاجر يضغط Install → يُعطي إذن → Widget يظهر تلقائياً.

### الخطوات

#### الخطوة 3.1 — تصميم OAuth Flow
```bash
/ultra-think
>> "صمّم Salla OAuth 2.0 flow:
   1. تاجر يضغط Install في Salla App Store
   2. redirect إلى /auth/salla/start
   3. redirect إلى Salla Authorization Server
   4. Salla يُعيد إلى /auth/salla/callback مع code
   5. نُبادل code بـ access_token + refresh_token
   6. نُخزّن tokens مُشفَّرة في Supabase
   7. نُنشئ سجل Store جديد
   8. Widget يظهر تلقائياً في المتجر"
```

#### الخطوة 3.2 — بناء OAuth Endpoints (3 endpoints)

| # | Endpoint | الوظيفة |
|---|----------|---------|
| 1 | `GET /auth/salla/start` | بدء OAuth Flow وتوليد state parameter |
| 2 | `GET /auth/salla/callback` | استقبال code وتبادله بـ tokens |
| 3 | `POST /auth/salla/refresh` | تجديد access_token قبل انتهائه |

```bash
>> backend-architect:
   "بناء هذه الـ 3 endpoints على Cloudflare Worker
    مع:
    - state parameter لمنع CSRF
    - تشفير tokens بـ AES-256 قبل التخزين
    - مفتاح التشفير من Cloudflare Secrets فقط"
```

#### الخطوة 3.3 — Widget Injection آلية التضمين
```bash
>> typescript-pro:
   "كيفية ظهور Widget في متجر سلة تلقائياً بعد التثبيت:
    - Salla يُضيف Script Tag في footer المتجر
    - Script يُحمّل widget.js من Cloudflare CDN
    - widget.js يتحقق من وجود store_id في meta tags
    - يبدأ تهيئة الـ Widget"
```

#### الخطوة 3.4 — Onboarding Flow (أول تجربة للتاجر)
```bash
>> ui-ux-designer + frontend-developer:
   "شاشة الإعداد الأولي تتضمن 5 حقول FAQ إلزامية:
    1. وقت ومواصفات الشحن
    2. طرق الدفع المتاحة
    3. سياسة الإرجاع والاستبدال
    4. معلومات التواصل مع المتجر
    5. حقل مخصص (اختياري)
    
    ثم زر 'تفعيل البوت' → toggle is_active = true"
```

### ✅ معايير الاكتمال
- [ ] OAuth flow يعمل end-to-end في Development
- [ ] Tokens مُخزَّنة مُشفَّرة في Supabase
- [ ] Widget يظهر في متجر سلة تجريبي
- [ ] Onboarding يُحفظ 5 حقول FAQ
- [ ] اختبار CSRF protection

---

## المرحلة 4: Chat Widget
**المدة:** يوم 10-14 | **الأدوات:** frontend-developer, ui-ux-designer, typescript-pro, playwright-mcp

### الهدف
Widget عربي خفيف، نصي، RTL، يعمل على جميع صفحات المتجر ما عدا Checkout.

### الخطوات

#### الخطوة 4.1 — تصميم UX Widget
```bash
/ultra-think
>> ui-ux-designer:
   "تصميم Chat Widget متطلباته:
    ✓ عربي RTL بالكامل
    ✓ نص فقط (بدون صور أو ملفات)
    ✓ يظهر على جميع الصفحات إلا /checkout/*
    ✓ أول رسالة: 'أنا مساعد ذكاء اصطناعي...'
    ✓ زر موافقة قبل حفظ أي بيانات
    ✓ حالة عند الوصول لـ 100% استخدام"
```

#### الخطوة 4.2 — مخطط حالات Widget

```
الحالات الممكنة (7 حالات):
1. مغلق      → أيقونة صغيرة في الزاوية
2. مفتوح     → نافذة محادثة فارغة
3. يكتب     → مؤشر typing (...)
4. استجابة  → رسالة الرد
5. تصعيد    → نموذج جمع البيانات
6. خطأ      → رسالة خطأ محترمة
7. ممتلئ    → "انتهى الحد اليومي للمتجر"
```

#### الخطوة 4.3 — بناء Widget كـ Vanilla TypeScript
```bash
>> typescript-pro + frontend-developer:
   "بناء Widget كـ Single File
    المتطلبات التقنية:
    - حجم < 50KB بعد الضغط
    - بدون أي Framework خارجي
    - RTL CSS مُضمَّن
    - Custom Events للتواصل مع الصفحة
    - Lazy Loading للـ Styles"
```

#### الخطوة 4.4 — Chat State Management
```bash
>> backend-architect:
   "نظام Ticket/Thread:
    - كل موضوع = ticket جديد
    - زائر بدون موافقة = anonymous (بدون cookie)
    - زائر بموافقة = مُرتبط بـ visitor_session
    - مدة الـ session: 24 ساعة"
```

#### الخطوة 4.5 — Rate Limiting على Widget
```bash
>> backend-architect:
   "تطبيق Rate Limiting:
    - 20 رسالة/دقيقة لكل IP/visitor
    - 3,000 reply/ساعة لكل متجر
    - عند تجاوز الحد: رسالة احترامية بالعربية"
```

#### الخطوة 4.6 — اختبار Widget
```bash
/write-tests + playwright-mcp:
   "اختبارات E2E:
    ✓ Widget يظهر على صفحة المنتج
    ✓ Widget يظهر على الصفحة الرئيسية
    ✗ Widget لا يظهر على /checkout
    ✓ رسالة الإفصاح تظهر أول مرة
    ✓ نموذج الموافقة يعمل
    ✓ Rate Limiting يعمل"
```

### ✅ معايير الاكتمال
- [ ] Widget يعمل في متصفحات: Chrome, Safari, Firefox
- [ ] حجم JS < 50KB
- [ ] لا يظهر على صفحة Checkout
- [ ] RTL يعمل بشكل صحيح
- [ ] Consent flow مُطبَّق
- [ ] جميع 6 اختبارات E2E تنجح

---

## المرحلة 5: AI Orchestrator
**المدة:** يوم 15-18 | **الأدوات:** ai-engineer, prompt-engineer, ultra-think, typescript-pro

### الهدف
وكيل واحد يرد بالعربية بدقة عالية بناءً على FAQ + بيانات المنتجات.

### الخطوات

#### الخطوة 5.1 — تصميم منطق الـ Agent
```bash
/ultra-think
>> ai-engineer:
   "صمّم Single Agent للـ MVP:
    
    Input المطلوب:
    - رسالة العميل الحالية
    - تاريخ المحادثة (آخر 10 رسائل)
    - FAQ المتجر (5 حقول)
    - Product Snapshot (للمنتجات المذكورة)
    
    Output المطلوب:
    - رد عربي نصي فقط
    - استدعاء escalate() بعد 3 محاولات
    
    القيود الصارمة:
    ✗ لا تُجرِ مقارنات بمتاجر أخرى
    ✗ لا تُعطِ معلومات غير موجودة في FAQ
    ✗ لا تُعِد على الدفع أو الاسترداد
    ✓ إضافة 'حسب آخر تحديث' للأسعار فقط"
```

#### الخطوة 5.2 — System Prompt الرئيسي
```bash
>> prompt-engineer:
   "اكتب System Prompt محكماً للبوت يتضمن:
    1. هوية البوت (مساعد المتجر)
    2. كيفية استخدام FAQ
    3. كيفية استخدام بيانات المنتجات
    4. قاعدة 3 محاولات للتوضيح
    5. متى وكيف يُصعِّد
    6. الممنوعات (8 ممنوعات صريحة)
    7. نبرة الردود (ودود، مهني، عربي فصيح)
    
    الطول المستهدف: < 500 token"
```

#### الخطوة 5.3 — OpenRouter Integration (3 Models)
```bash
>> typescript-pro + ai-engineer:
   "بناء OpenRouter Client مع:
    Primary:  Gemini 2.0 Flash    (سرعة + تكلفة منخفضة)
    Fallback1: GPT-4 Mini          (إذا فشل Gemini)
    Fallback2: Claude 3.5 Sonnet   (آخر احتياط)
    
    مع:
    - Timeout: 8 ثوان
    - Retry Logic: 2 محاولة
    - Token Counting لكل استجابة
    - تسجيل Model المستخدم في audit_log"
```

#### الخطوة 5.4 — اختبار الدقة
```bash
/generate-test-cases
>> "50 سؤال اختبار يشمل:
   - أسئلة الشحن (10)
   - أسئلة الدفع (10)
   - أسئلة المنتجات (10)
   - أسئلة خارج النطاق — يجب رفضها (10)
   - محاولات للحصول على استرداد (10)"
```

#### الخطوة 5.5 — قياس الأداء
```bash
/optimize-api-performance
>> "قياس:
   P50 ≤ 1.5s ✓
   P95 ≤ 3.0s ✓
   Timeout 8s ✓
   
   إذا P95 > 3s → optimize الـ context window"
```

### ✅ معايير الاكتمال
- [ ] دقة الردود ≥ 90% على أسئلة FAQ
- [ ] لا يُجيب على الأسئلة خارج النطاق
- [ ] التصعيد يعمل بعد 3 محاولات
- [ ] Fallback Models يعملان
- [ ] P50 ≤ 1.5s | P95 ≤ 3.0s

---

## المرحلة 6: Product Sync Service
**المدة:** يوم 19-21 | **الأدوات:** backend-architect, data-engineer, typescript-pro

### الهدف
مزامنة دورية لمنتجات سلة بناءً على الباقة بأداء عالي.

### الخطوات

#### الخطوة 6.1 — Salla Products API Client
```bash
>> typescript-pro:
   "بناء Salla Products API Client:
    - Pagination (جميع المنتجات)
    - Rate Limiting احترامي (Salla limits)
    - Error Handling + Retry
    - تخزين في product_snapshots"
```

#### الخطوة 6.2 — Cloudflare Cron Trigger (3 جداول)

| الباقة | التكرار | الأمر |
|--------|---------|-------|
| Basic | كل 24 ساعة | `0 2 * * *` |
| Mid-tier | كل 6 ساعات | `0 */6 * * *` |
| Higher | كل ساعة | `0 * * * *` |

```bash
>> backend-architect + data-engineer:
   "بناء Sync Worker مع:
    - Cloudflare Cron Triggers (3 schedules)
    - Batch processing: 100 منتج/request
    - Total target: 1,000 منتج < 60 ثانية
    - تحديث synced_at بعد كل مزامنة
    - إعادة محاولة عند فشل Salla API"
```

#### الخطوة 6.3 — اختبارات Sync Service

```bash
/generate-test-cases
>> "اختبارات:
   ✓ مزامنة ناجحة لـ 1,000 منتج < 60s
   ✓ إعادة المحاولة عند فشل Salla API
   ✓ تحديث synced_at الصحيح
   ✓ تكرارات مختلفة حسب الباقة
   ✓ لا تعديل على بيانات Salla (read-only)"
```

### ✅ معايير الاكتمال
- [ ] 1,000 منتج تُزامَن < 60 ثانية
- [ ] 3 Cron schedules تعمل
- [ ] Retry logic مُطبَّق
- [ ] Read-Only مؤكّد (لا كتابة في Salla)

---

## المرحلة 7: Merchant Dashboard
**المدة:** يوم 22-26 | **الأدوات:** frontend-developer, ui-ux-designer, typescript-pro, supabase

### الهدف
لوحة تحكم بسيطة وسريعة للتاجر (P95 ≤ 2.5 ثانية).

### الخطوات

#### الخطوة 7.1 — تصميم Dashboard
```bash
>> ui-ux-designer:
   "Dashboard يحتوي على 4 أقسام فقط:

    1. الصفحة الرئيسية:
       - استخدام الباقة (شريط بصري)
       - إحصاءات: ردود اليوم، مجموع الشهر
       - زر تشغيل/إيقاف البوت

    2. المحادثات (قراءة فقط):
       - قائمة Tickets
       - عرض الرسائل
       - تصفية حسب الحالة

    3. التصعيدات:
       - قائمة الحالات
       - تفاصيل كل حالة
       - زر إغلاق (تمت المعالجة)

    4. الإعدادات:
       - تعديل FAQ
       - إدارة الاشتراك"
```

#### الخطوة 7.2 — شريط الاستخدام البصري
```bash
>> frontend-developer:
   "شريط استخدام ديناميكي:
    0% ────────────────── 100%
    
    ألوان:
    0-79%  → أخضر (طبيعي)
    80-99% → برتقالي (تحذير)
    100%   → أحمر (ممتلئ)
    
    مع نص: '850 من 1,000 رد'
    ومع تاريخ تجديد الدورة"
```

#### الخطوة 7.3 — Supabase Real-time للإشعارات
```bash
/supabase-realtime-monitor
>> "Real-time subscriptions على:
   - جدول escalations (لإشعار التاجر)
   - جدول subscriptions (لتحديث الاستخدام)
   بدون reload"
```

### ✅ معايير الاكتمال
- [ ] Dashboard يُحمَّل < 2.5 ثانية (P95)
- [ ] 4 أقسام تعمل
- [ ] Real-time updates للتصعيدات
- [ ] زر التشغيل/الإيقاف يعمل فوراً
- [ ] FAQ يمكن تعديله

---

## المرحلة 8: Escalation & Billing Metering
**المدة:** يوم 27-30 | **الأدوات:** backend-architect, ai-engineer, data-engineer

### الهدف
منطق التصعيد الذكي + عداد دقيق للاستخدام + إشعارات فورية.

### الخطوات

#### الخطوة 8.1 — Escalation Flow المفصّل
```bash
>> backend-architect + ai-engineer:
   "تسلسل التصعيد:

    المحاولة 1: Bot يحاول الإجابة
    المحاولة 2: Bot يطلب توضيح
    المحاولة 3: Bot يُعلن التصعيد

    عند التصعيد:
    1. Bot: 'سأربطك بفريق المتجر'
    2. نموذج يجمع:
       - وصف المشكلة (إلزامي)
       - رقم الطلب (اختياري)
       - بريد إلكتروني أو هاتف (واحد إلزامي)
    3. تشفير بيانات التواصل (AES-256)
    4. حفظ في جدول escalations
    5. إشعار في Dashboard التاجر"
```

#### الخطوة 8.2 — Usage Metering الدقيق
```bash
>> data-engineer + backend-architect:
   "عداد الاستخدام:
    - يرتفع عند كل bot_reply (is_bot_reply = true)
    - فحص عند كل رد: هل تجاوزنا 80%؟
    - عند 80%: إيميل للتاجر + تنبيه في Dashboard
    - عند 100%: إيميل للتاجر + Widget يُعرض رسالة بديلة
    - يُعاد إلى 0 في أول يوم من الدورة"
```

#### الخطوة 8.3 — إشعارات Telegram للمشغّل
```bash
# تفعيل hooks التليغرام
telegram-detailed-notifications + telegram-error-notifications

>> "إشعارات فورية لمشغّل MoradBot عند:
   - متجر جديد يُثبَّت التطبيق
   - متجر يصل لـ 80% أو 100%
   - فشل sync لمتجر لأكثر من ساعة
   - خطأ في AI Orchestrator
   - تجاوز Rate Limiting بشكل متكرر"
```

### ✅ معايير الاكتمال
- [ ] Escalation يُشغَّل بعد المحاولة الثالثة
- [ ] بيانات التواصل مُشفَّرة
- [ ] عداد الاستخدام دقيق 100%
- [ ] إيميل 80% يُرسَل
- [ ] إيميل 100% يُرسَل + Widget يتغير

---

## المرحلة 9: الأمان والاختبار الشامل
**المدة:** يوم 31-35 | **الأدوات:** security-auditor, api-security-audit, load-testing-specialist, /test-coverage

### الهدف
مراجعة أمنية شاملة + اختبارات كاملة + PDPL Compliance.

### الخطوات

#### الخطوة 9.1 — Security Audit شامل
```bash
/security-audit
>> security-auditor + api-security-audit:
   "تحقق من قائمة الأمان الكاملة:

    ✓ TLS 1.2+ على جميع الاتصالات
    ✓ تشفير tokens و escalation contacts
    ✓ Rate Limiting: 20 msg/min للزائر
    ✓ Rate Limiting: 3,000 reply/hour للمتجر
    ✓ لا أسرار في الكود أو .env
    ✓ Audit Logs retention 90+ يوم
    ✓ RLS على 10 جداول
    ✓ عزل البيانات بين المتاجر
    ✓ CSRF protection في OAuth
    ✓ Input validation لجميع النماذج"
```

#### الخطوة 9.2 — PDPL Saudi Arabia Compliance
```bash
>> legal-advisor:
   "قائمة PDPL الإلزامية:
    ✓ موافقة صريحة موثّقة في consent_logs
    ✓ حذف البيانات الشخصية 30-90 يوم بعد الإلغاء
    ✓ عدم مشاركة البيانات مع أطراف ثالثة
    ✓ إمكانية تصدير بيانات المتجر
    ✓ وضوح سياسة الخصوصية"
```

#### الخطوة 9.3 — Load Testing
```bash
>> load-testing-specialist:
   "اختبار الحمل:
    - 100 متجر متزامن
    - 1,000 رسالة/دقيقة إجمالية
    - مدة الاختبار: 30 دقيقة
    
    المتوقع:
    P50 ≤ 1.5s ✓
    P95 ≤ 3.0s ✓
    Error Rate < 0.1% ✓"
```

#### الخطوة 9.4 — Test Coverage Report
```bash
/test-coverage
>> "المستهدف:
   - AI Orchestrator: ≥ 85%
   - Escalation Flow: ≥ 90%
   - Usage Metering: ≥ 90%
   - Data Isolation: 100%
   - Rate Limiting: ≥ 85%
   - OAuth Flow: ≥ 80%
   
   الإجمالي المستهدف: ≥ 80%"
```

### ✅ معايير الاكتمال
- [ ] Security Audit يعطي 0 Critical Issues
- [ ] PDPL Checklist مكتمل 100%
- [ ] Load Test يُثبت الأداء المطلوب
- [ ] Test Coverage ≥ 80%
- [ ] جميع الاختبارات تنجح

---

## المرحلة 10: Production Deployment
**المدة:** يوم 36-40 | **الأدوات:** incident-responder, deployment-health-monitor

### الهدف
نشر آمن ومنظّم في بيئة Production مع مراقبة كاملة.

### الخطوات

#### الخطوة 10.1 — Pre-Deploy Checklist الإلزامي

```bash
/project-health-check
>> "تحقق من جميع البنود قبل النشر:
   ✓ جميع الاختبارات تنجح (0 failures)
   ✓ Test Coverage ≥ 80%
   ✓ Security Audit: 0 Critical
   ✓ لا secrets في الكود
   ✓ Cloudflare Secrets مُعدَّة
   ✓ Supabase Migrations مُطبَّقة
   ✓ Monitoring مُعدَّ
   ✓ Rollback Plan جاهز"
```

#### الخطوة 10.2 — Monitoring & Alerting Setup

| قاعدة | الشرط | الإجراء |
|-------|-------|---------|
| Error Rate | > 5% لمدة 5 دقائق | تنبيه فوري على تليغرام |
| Latency | Avg > 4s لمدة 5 دقائق | تنبيه + مراجعة |
| Token Usage | > 200% من baseline | تنبيه للمراجعة |
| Sync Failure | > 30 دقيقة بدون sync | تنبيه + إعادة محاولة |
| Error Critical | أي خطأ أمني | **إغلاق فوري** |

#### الخطوة 10.3 — النشر اليدوي (الترتيب مهم)

```bash
# 1. تطبيق Migrations على Production DB
supabase db push --linked

# 2. نشر API Worker
wrangler deploy --env production apps/api/

# 3. نشر Widget على CDN
wrangler deploy --env production apps/widget/

# 4. نشر Dashboard على Cloudflare Pages
wrangler pages deploy apps/dashboard/dist/

# 5. التحقق من الصحة
/deployment-health-monitor
```

#### الخطوة 10.4 — Beta Launch Plan

```
الأسبوع 1: 3-5 متاجر مختارة (Beta مغلق)
الأسبوع 2: جمع Feedback + إصلاح المشاكل
الأسبوع 3: 10-20 متجر
الأسبوع 4: فتح التسجيل المحدود
```

### ✅ معايير الاكتمال
- [ ] Pre-Deploy Checklist 100%
- [ ] Monitoring يرصد جميع الـ Metrics
- [ ] نشر ناجح بدون أخطاء
- [ ] Widget يعمل في متجر تجريبي في Production
- [ ] أول 3 متاجر Beta تعمل

---

## 📊 ملخص المراحل النهائي

| # | المرحلة | المدة | المخرجات | أهم الأدوات |
|---|---------|-------|---------|------------|
| 0 | إعداد البيئة | يوم 1 | 975+ مكوّن + CLAUDE.md | health-check |
| 1 | هيكل المشروع | يوم 2-3 | Monorepo جاهز | ultra-think |
| 2 | Supabase | يوم 4-6 | 10 جداول + RLS | design-database-schema |
| 3 | Salla OAuth | يوم 7-9 | تثبيت وتفويض يعمل | auth-implementation |
| 4 | Chat Widget | يوم 10-14 | Widget RTL < 50KB | playwright-mcp |
| 5 | AI Agent | يوم 15-18 | P50≤1.5s, دقة≥90% | prompt-engineer |
| 6 | Sync Service | يوم 19-21 | 1,000 منتج < 60s | data-engineer |
| 7 | Dashboard | يوم 22-26 | P95≤2.5s | frontend-developer |
| 8 | Escalation/Billing | يوم 27-30 | عداد دقيق + تصعيد | backend-architect |
| 9 | Security/Testing | يوم 31-35 | Coverage≥80%, 0 Critical | security-auditor |
| 10 | Production | يوم 36-40 | Beta مغلق يعمل | incident-responder |

**المجموع: 40 يوم عمل → Beta جاهز**

---

*آخر تحديث: فبراير 2026 | المشروع: MoradBot SaaS*
