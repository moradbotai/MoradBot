> ⚠️ **SUPERSEDED** — هذه الوثيقة تعكس تخطيطاً قبل التنفيذ. معظم الأدوات المُدرجة (975 مكوناً) لم تُثبَّت.
> المرجع الصحيح للحالة الفعلية: `docs/claude/tools_report_v2.md`
> **لا تنفّذ أمر التثبيت الشامل الموجود في هذه الوثيقة.**

# 📦 تقرير الأدوات الشامل — MoradBot Claude Code Stack
**الإصدار:** 1.0.0 | **التاريخ:** فبراير 2026 | **المشروع:** MoradBot SaaS

---

## 📊 ملخص إحصائي

| الفئة | العدد | الحالة |
|-------|-------|--------|
| Plugins | 5 | ✅ جاهز للتثبيت |
| Skills Libraries | 2 مكتبة (860+ + مخصصة) | ✅ جاهز |
| MCP Servers | 20 | ✅ جاهز |
| Agents | 31 | ✅ جاهز |
| Commands | 35 | ✅ جاهز |
| Hooks | 16 | ✅ جاهز |
| Settings | 2 | ✅ جاهز |
| Skills (aitmpl) | 3 | ✅ جاهز |
| **الإجمالي** | **~975+ مكوّن** | **✅** |

---

## 1. 🔌 Plugins (5 إضافات)

> **التعريف:** إضافات تُثبَّت مباشرة على Claude.ai وتوسّع قدراته بشكل دائم عبر جميع المحادثات.

### 1.1 جدول المقارنة الشامل

| # | Plugin | الرابط | الوظيفة الأساسية | الأهمية لـ MoradBot | حالة التثبيت |
|---|--------|--------|-----------------|---------------------|--------------|
| 1 | **Superpowers** | claude.com/plugins/superpowers | يُضيف قدرات متقدمة من Anthropic الرسمية — Skills، أوامر معزّزة، Ultrathink | ⭐⭐⭐⭐⭐ أساسي | 🔜 |
| 2 | **Frontend Design** | claude.com/plugins/frontend-design | متخصص في بناء UI/UX احترافي، مكونات React، تصميم responsiv | ⭐⭐⭐⭐⭐ للـ Widget والـ Dashboard | 🔜 |
| 3 | **Ralph Loop** | claude.com/plugins/ralph-loop | أتمتة دورات التطوير، تكرار سريع، loop execution للمهام المتكررة | ⭐⭐⭐⭐ لتسريع التطوير | 🔜 |
| 4 | **Firecrawl** | claude.com/plugins/firecrawl | Web scraping ذكي وجلب محتوى الويب بدقة عالية | ⭐⭐⭐ مفيد لاحقاً لـ Salla | 🔜 |
| 5 | **Code Review** | claude.com/plugins/code-review | مراجعة كود تلقائية ومنهجية مع تقارير مفصّلة | ⭐⭐⭐⭐⭐ لضمان الجودة | 🔜 |

### 1.2 أولوية التثبيت

```
الأولوية القصوى  → Superpowers + Code Review + Frontend Design
الأولوية الثانوية → Ralph Loop
الأولوية المنخفضة → Firecrawl (مرحلة لاحقة)
```

---

## 2. 🧠 Skills Libraries (مكتبتان رئيسيتان)

> **التعريف:** مهارات قابلة لإعادة الاستخدام تُثبَّت في `.claude/skills/` وتُستدعى بأوامر `/skill-name`.

### 2.1 antigravity-awesome-skills

| الخاصية | القيمة |
|---------|--------|
| **الإصدار الحالي** | v5.4.0 |
| **عدد Skills** | 860+ |
| **النجوم على GitHub** | 10,400+ ⭐ |
| **المساهمون** | 2,000+ Fork |
| **الترخيص** | MIT |
| **أمر التثبيت** | `npx antigravity-awesome-skills --claude` |
| **مسار التثبيت** | `.claude/skills/` |

**أبرز الفئات والـ Skills المهمة لـ MoradBot:**

| الفئة | Skills الأساسية | الاستخدام في المشروع |
|-------|----------------|---------------------|
| **Full-Stack** | react-patterns, tailwind-mastery, api-design | Chat Widget + Dashboard |
| **Security** | auth-implementation-patterns, api-security-best-practices, backend-security-coder | PDPL Compliance |
| **Database** | supabase-best-practices, postgres-patterns | Schema + RLS |
| **AI/Agent** | llm-application-developer, memory-persistence, prompt-engineer | AI Orchestrator |
| **Testing** | jest-patterns, tdd-workflow, cc-skill-security-review | جميع المراحل |
| **DevOps** | environment-setup-guide, docker-expert, cloudflare-workers | نشر Cloudflare |
| **Documentation** | api-documenter, technical-writer | توثيق API |
| **Architecture** | clean-architecture, quality-coder | هيكل المشروع |

**طريقة الاستخدام:**
```bash
# استدعاء Skill مباشرة
>> /auth-implementation-patterns implement Salla OAuth

# استدعاء تلقائي بناءً على السياق
>> "help me secure the API endpoints"
# Claude يحمّل api-security-best-practices تلقائياً
```

### 2.2 aitmpl Skills (3 مخصصة)

| # | Skill | الوظيفة |
|---|-------|---------|
| 1 | `development/skill-creator` | إنشاء Skills جديدة مخصصة لـ MoradBot |
| 2 | `creative-design/theme-factory` | بناء ثيمات تصميم عربية للـ Widget |
| 3 | `productivity/file-organizer` | تنظيم هيكل ملفات المشروع |

---

## 3. 🔗 MCP Servers (20 خادم)

> **التعريف:** بروتوكول Model Context Protocol يربط Claude بالأنظمة الخارجية مباشرة.

### 3.1 الجدول الشامل

| # | MCP Server | المزود | الوظيفة | الأهمية لـ MoradBot | الفئة |
|---|-----------|--------|---------|---------------------|-------|
| 1 | **context7** | Upstash/Context7 | توثيق حي لـ 1,000+ مكتبة (Supabase, CF, React) | ⭐⭐⭐⭐⭐ حرجة | Dev Tools |
| 2 | **sequential-thinking** | Anthropic MCP | تفكير منهجي خطوة بخطوة للمسائل المعقدة | ⭐⭐⭐⭐⭐ | Dev Tools |
| 3 | **memory-integration** | claude-mem | ذاكرة دائمة عبر الجلسات لحفظ سياق المشروع | ⭐⭐⭐⭐⭐ | Integration |
| 4 | **supabase** | Supabase | تكامل مباشر مع قاعدة البيانات والـ Auth | ⭐⭐⭐⭐⭐ حرجة | Database |
| 5 | **postgresql-integration** | MCP | استعلامات SQL مباشرة وإدارة قاعدة البيانات | ⭐⭐⭐⭐⭐ | Database |
| 6 | **postgresql-documentation** | MCP | توثيق Schema تلقائي وـ ERD | ⭐⭐⭐⭐ | Database |
| 7 | **github-integration** | GitHub MCP | إدارة GitHub: PRs, Issues, Branches | ⭐⭐⭐⭐⭐ | Integration |
| 8 | **filesystem-access** | Anthropic MCP | الوصول الكامل لملفات المشروع | ⭐⭐⭐⭐⭐ أساسية | Dev Tools |
| 9 | **playwright-mcp-server** | Playwright | أتمتة المتصفح واختبار الـ Widget | ⭐⭐⭐⭐ | Browser |
| 10 | **playwright-mcp** | Playwright | نسخة بديلة محسّنة لأتمتة المتصفح | ⭐⭐⭐⭐ | Browser |
| 11 | **browser-use-mcp-server** | Community | أتمتة متصفح متقدمة وتفاعلية | ⭐⭐⭐ | Browser |
| 12 | **chrome-devtools** | Chrome | DevTools للتصحيح وتحليل الأداء | ⭐⭐⭐ | Browser |
| 13 | **web-fetch** | Community | جلب محتوى الويب وقراءة الصفحات | ⭐⭐⭐ | Web |
| 14 | **firecrawl** | Firecrawl.dev | Web scraping ذكي وهيكلي | ⭐⭐⭐ | Web |
| 15 | **deepgraph-nextjs** | DeepGraph | تحليل عميق لمشاريع Next.js | ⭐⭐⭐⭐ | Dev Tools |
| 16 | **deepgraph-react** | DeepGraph | تحليل مكونات React وتبعياتها | ⭐⭐⭐⭐ | Dev Tools |
| 17 | **deepgraph-typescript** | DeepGraph | تحليل TypeScript وتصحيح الأنواع | ⭐⭐⭐⭐⭐ | Dev Tools |
| 18 | **testsprite** | TestSprite | توليد اختبارات تلقائية من الكود | ⭐⭐⭐⭐ | Testing |
| 19 | **markitdown** | Microsoft | تحويل الملفات (PDF, DOCX) لـ Markdown | ⭐⭐⭐ | Utilities |
| 20 | **imagesorcery** | Community | معالجة ومعالجة الصور برمجياً | ⭐⭐ | Utilities |
| _(bonus)_ | **box** | Box.com | تكامل Box storage للملفات | ⭐ | Storage |

### 3.2 تصنيف حسب الأولوية

```
🔴 حرجة (لا يمكن العمل بدونها):
   supabase, postgresql-integration, filesystem-access,
   context7, sequential-thinking, github-integration

🟡 عالية الأهمية:
   memory-integration, playwright-mcp, deepgraph-typescript,
   testsprite, postgresql-documentation, deepgraph-react

🟢 متوسطة (مفيدة):
   web-fetch, chrome-devtools, deepgraph-nextjs,
   firecrawl, markitdown

⚪ منخفضة (مستقبلية):
   browser-use-mcp, imagesorcery, box
```

---

## 4. 🤖 Agents (31 وكيل)

> **التعريف:** شخصيات AI متخصصة مُحسَّنة لأدوار محددة، تُثبَّت في `.claude/agents/`.

### 4.1 فريق التطوير (4 وكلاء)

| # | Agent | الدور التفصيلي | مهام MoradBot |
|---|-------|---------------|---------------|
| 1 | `frontend-developer` | بناء UI بـ React/TypeScript، CSS المتقدم، Accessibility | Chat Widget + Dashboard |
| 2 | `backend-architect` | تصميم APIs، Microservices، قرارات البنية | Cloudflare Workers APIs |
| 3 | `ui-ux-designer` | تصميم تجربة المستخدم، Wire-frames، User Flows | UX عربية RTL |
| 4 | `ui-designer` | التصميم البصري، الألوان، التيبوغرافيا | هوية بصرية Widget |

### 4.2 لغات البرمجة (4 وكلاء)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 5 | `typescript-pro` | TypeScript متقدم، Generic Types، Decorators | جميع الكود الأساسي |
| 6 | `javascript-pro` | JavaScript ES2024، Browser APIs، Bundling | Chat Widget |
| 7 | `python-pro` | Python للأدوات والـ Scripts المساعدة | أدوات التطوير |
| 8 | `sql-pro` | PostgreSQL، Query Optimization، Window Functions | Supabase Queries |

### 4.3 AI والبيانات (5 وكلاء)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 9 | `ai-engineer` | LLM Integration، Prompting، RAG، Agents | AI Orchestrator |
| 10 | `data-engineer` | ETL، Data Pipelines، Scheduling | Salla Sync Service |
| 11 | `task-decomposition-expert` | تقسيم المهام المعقدة لخطوات قابلة للتنفيذ | التخطيط والتنفيذ |
| 12 | `prompt-engineer` | System Prompts، Few-shot، Chain-of-thought | Bot Prompts |
| 13 | `connection-agent` (Obsidian) | ربط المعرفة وتنظيم المعلومات | إدارة مستندات المشروع |

### 4.4 الأمان والجودة (4 وكلاء)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 14 | `security-auditor` | OWASP، Penetration Testing، Security Review | مراجعة أمنية شاملة |
| 15 | `api-security-audit` | API Authentication، Authorization، Rate Limiting | أمان Cloudflare API |
| 16 | `error-detective` | تشخيص الأخطاء المعقدة، Root Cause Analysis | تصحيح المشاكل |
| 17 | `dx-optimizer` | تحسين تجربة المطوّر، Tooling، Workflow | إعداد بيئة العمل |

### 4.5 قواعد البيانات (2 وكيلان)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 18 | `database-optimization` | Index Design، Query Planning، Partitioning | تحسين Supabase |
| 19 | `database-optimizer` | Query Rewriting، Execution Plans | استعلامات الإنتاج |

### 4.6 التوثيق (2 وكيلان)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 20 | `api-documenter` | OpenAPI، Swagger، API References | توثيق REST API |
| 21 | `technical-writer` | Technical Docs، Guides، READMEs | توثيق المشروع |

### 4.7 الأداء والاختبار (2 وكيلان)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 22 | `load-testing-specialist` | K6، Locust، Artillery، Load Patterns | اختبار الحمل |
| 23 | `web-vitals-optimizer` | Core Web Vitals، LCP، CLS، FID | أداء Widget |

### 4.8 الأعمال والتسويق (4 وكلاء)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 24 | `payment-integration` | Stripe، Billing Logic، Subscription Management | منطق الاشتراكات |
| 25 | `content-marketer` | Content Strategy، Arabic Copywriting | محتوى Landing Page |
| 26 | `seo-specialist` | Technical SEO، Arabic SEO، Core Web Vitals | تحسين محركات البحث |
| 27 | `legal-advisor` | Privacy Law، Compliance، Terms of Service | PDPL Saudi Arabia |

### 4.9 البنية والعمليات (4 وكلاء)

| # | Agent | التخصص | الاستخدام |
|---|-------|---------|-----------|
| 28 | `incident-responder` | Incident Response، Post-mortem، On-call | سياسة الإغلاق الفوري |
| 29 | `content-curator` (Obsidian) | تنظيم المعرفة والمحتوى | إدارة الوثائق |
| 30 | `competitive-intelligence-analyst` | تحليل المنافسين، Market Research | مراقبة سوق سلة |
| 31 | `context7` (documentation) | الوصول الفوري لتوثيق المكتبات | دعم التطوير |

---

## 5. ⚡ Commands (35 أمر)

> **التعريف:** أوامر مخصصة تُثبَّت في `.claude/commands/` وتُستدعى بـ `/command-name`.

### 5.1 أوامر التفكير والتخطيط (4 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 1 | `ultra-think` | تفكير عميق وتحليل متعدد الزوايا قبل القرارات الكبيرة | أي قرار معماري أو أمني |
| 2 | `architecture-scenario-explorer` | استكشاف سيناريوهات معمارية بديلة مع المقايضات | تصميم البنية |
| 3 | `prime` | تهيئة جلسة Claude Code بتحميل السياق الكامل | بداية كل جلسة |
| 4 | `resume` | استئناف العمل من حيث توقفت بدقة | بعد أي انقطاع |

### 5.2 أوامر تطوير الكود (5 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 5 | `refactor-code` | إعادة هيكلة كاملة مع الحفاظ على الوظائف | تحسين كود قديم |
| 6 | `code-review` | مراجعة شاملة: جودة، أمان، أداء، اتساق | قبل كل Merge |
| 7 | `debug-error` | تشخيص منهجي للأخطاء مع Root Cause Analysis | عند الأخطاء المعقدة |
| 8 | `all-tools` | تفعيل جميع الأدوات المتاحة في آن واحد | للمهام الشاملة |
| 9 | `directory-deep-dive` | تحليل عميق لهيكل أي مجلد أو ملف | فهم الكود الموروث |

### 5.3 أوامر قاعدة البيانات — Supabase (8 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 10 | `design-database-schema` | تصميم Schema كامل من متطلبات عمل | المرحلة 2 |
| 11 | `supabase-data-explorer` | استكشاف وتحليل البيانات الموجودة | الفهم والتصحيح |
| 12 | `supabase-schema-sync` | مزامنة Schema بين Environments | بعد كل Migration |
| 13 | `supabase-migration-assistant` | إنشاء وإدارة Migration Files بأمان | أي تغيير في Schema |
| 14 | `supabase-type-generator` | توليد TypeScript Types من Schema | بعد كل Schema change |
| 15 | `supabase-performance-optimizer` | تحليل وتحسين أداء الاستعلامات | عند بطء الاستجابة |
| 16 | `supabase-security-audit` | تدقيق أمني لـ RLS والـ Policies | قبل كل نشر |
| 17 | `supabase-realtime-monitor` | مراقبة Realtime Subscriptions | اختبار التكامل |

### 5.4 أوامر الاختبار (4 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 18 | `generate-test-cases` | توليد حالات اختبار شاملة من الكود | بعد كل Feature |
| 19 | `write-tests` | كتابة اختبارات Unit + Integration | التطوير الموجه بالاختبارات |
| 20 | `test-coverage` | تقرير تغطية تفصيلي مع النقاط العمياء | قبل الإطلاق |
| 21 | `nextjs-component-generator` | توليد مكونات Next.js متكاملة | Dashboard Components |

### 5.5 أوامر التوثيق (4 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 22 | `update-docs` | تحديث جميع وثائق المشروع تلقائياً | نهاية كل جلسة |
| 23 | `generate-api-documentation` | توليد OpenAPI / Swagger docs كاملة | بعد بناء API |
| 24 | `doc-api` | توثيق API سريع ومختصر | توثيق نقاط نهاية جديدة |
| 25 | `troubleshooting-guide` | إنشاء دليل استكشاف الأخطاء | قبل الإطلاق |

### 5.6 أوامر الأداء والأمان (6 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 26 | `performance-audit` | تدقيق شامل للأداء مع توصيات | قبل كل Release |
| 27 | `optimize-database-performance` | تحسين استعلامات قاعدة البيانات | عند بطء الاستجابة |
| 28 | `optimize-api-performance` | تحسين API Response Times | بعد Load Testing |
| 29 | `security-audit` | تدقيق أمني شامل: OWASP Top 10 | قبل كل نشر |
| 30 | `dependency-audit` | فحص المكتبات الخارجية للثغرات | أسبوعياً |
| 31 | `setup-docker-containers` | إعداد بيئة Docker للتطوير المحلي | المرحلة 0 |

### 5.7 أوامر إدارة المشروع (4 أوامر)

| # | Command | الوظيفة التفصيلية | متى تستخدمه |
|---|---------|-----------------|-------------|
| 32 | `project-health-check` | فحص صحة شامل للمشروع: كود، اختبارات، أمان | يومياً |
| 33 | `session-learning-capture` | تسجيل دروس وقرارات الجلسة | نهاية كل جلسة |
| 34 | `memory-spring-cleaning` | تنظيف ذاكرة الجلسات القديمة | شهرياً |
| 35 | `setup-development-environment` | إعداد بيئة التطوير من الصفر | المرحلة 0 |

---

## 6. 🪝 Hooks (16 خطاف)

> **التعريف:** أتمتة تُشغَّل تلقائياً عند أحداث محددة (قبل/بعد أوامر Claude Code).

### 6.1 الجدول الشامل

| # | Hook | يُشغَّل متى | الوظيفة التفصيلية | النوع |
|---|------|------------|-----------------|-------|
| 1 | `dependency-checker` | قبل كل Build | فحص المكتبات المُثبَّتة وتحديثاتها | pre-tool |
| 2 | `nextjs-code-quality-enforcer` | عند تعديل ملفات | إنفاذ معايير جودة Next.js (ESLint, Prettier) | change |
| 3 | `change-tracker` | بعد كل تعديل | تسجيل جميع التغييرات في سجل منظّم | post-tool |
| 4 | `security-scanner` | بعد كل تعديل | مسح أمني فوري للكود المُعدَّل | post-tool |
| 5 | `run-tests-after-changes` | post-tool | تشغيل الاختبارات المرتبطة بالتغيير تلقائياً | post-tool |
| 6 | `file-backup` | قبل التعديل | نسخ احتياطي للملف قبل أي تعديل | pre-tool |
| 7 | `test-runner` | عند الطلب | تشغيل مجموعة الاختبارات الكاملة | on-demand |
| 8 | `agents-md-loader` | بداية الجلسة | تحميل AGENTS.md وسياق الوكلاء تلقائياً | startup |
| 9 | `telegram-detailed-notifications` | عند الأحداث | إرسال إشعارات مفصّلة على تليغرام | event |
| 10 | `file-protection` | قبل الحذف/التعديل | حماية الملفات الحرجة من الحذف العرضي | pre-tool |
| 11 | `backup-before-edit` | pre-tool | نسخ احتياطي شامل قبل أي تعديل | pre-tool |
| 12 | `build-on-change` | بعد التعديل | إعادة بناء المشروع تلقائياً | post-tool |
| 13 | `deployment-health-monitor` | بعد النشر | التحقق من صحة النشر تلقائياً | post-deploy |
| 14 | `telegram-error-notifications` | عند الأخطاء | إشعار فوري على تليغرام عند أي خطأ | error |
| 15 | `simple-notifications` | عام | إشعارات مبسّطة للأحداث العادية | general |
| 16 | `performance-monitor` | مستمر | مراقبة أداء الجلسة ومقاييس الإنتاجية | continuous |

---

## 7. ⚙️ Settings (إعدادان)

| # | Setting | الوظيفة | السبب |
|---|---------|---------|-------|
| 1 | `mcp/enable-all-project-servers` | تفعيل جميع MCP Servers المُعرَّفة في المشروع تلقائياً | ضمان توفر جميع الأدوات |
| 2 | `telemetry/disable-telemetry` | إيقاف جمع بيانات الاستخدام | خصوصية بيانات المشروع |

---

## 8. 📊 أدوات المراقبة والصيانة

| # | الأداة | الأمر | الوظيفة التفصيلية |
|---|--------|-------|-----------------|
| 1 | **Analytics Dashboard** | `npx claude-code-templates@latest --analytics` | مراقبة جلسات Claude Code: استخدام الأدوات، الأداء، الإنتاجية |
| 2 | **Health Check** | `npx claude-code-templates@latest --health-check` | تشخيص إعداد Claude Code واكتشاف المشاكل |
| 3 | **Chat Monitor** | `npx claude-code-templates@latest --chats` | مراقبة ردود Claude وأنماط التفكير لحظياً |
| 4 | **Plugin Dashboard** | `npx claude-code-templates@latest --plugins` | إدارة بصرية للـ Plugins: تفعيل، تعطيل، حالة |

---

## 9. 📋 أمر التثبيت الكامل

```bash
# ===== الخطوة 1: تثبيت Skills الكبيرة =====
npx antigravity-awesome-skills --claude

# ===== الخطوة 2: تثبيت جميع مكونات aitmpl =====
npx claude-code-templates@latest \
  --agent development-team/frontend-developer,development-team/backend-architect,development-team/ui-ux-designer,programming-languages/python-pro,programming-languages/typescript-pro,programming-languages/javascript-pro,programming-languages/sql-pro,ai-specialists/task-decomposition-expert,development-tools/error-detective,data-ai/ai-engineer,data-ai/data-engineer,documentation/api-documenter,documentation/technical-writer,security/security-auditor,database/database-optimization,database/database-optimizer,security/api-security-audit,development-tools/dx-optimizer,business-marketing/content-marketer,business-marketing/payment-integration,obsidian-ops-team/connection-agent,obsidian-ops-team/content-curator,deep-research-team/competitive-intelligence-analyst,performance-testing/load-testing-specialist,performance-testing/web-vitals-optimizer,business-marketing/legal-advisor,devops-infrastructure/incident-responder,documentation/context7,ai-specialists/prompt-engineer,business-marketing/seo-specialist,development-team/ui-designer \
  --command "utilities/ultra-think,utilities/refactor-code,utilities/code-review,utilities/all-tools,documentation/update-docs,documentation/generate-api-documentation,setup/design-database-schema,database/supabase-data-explorer,database/supabase-schema-sync,documentation/doc-api,security/security-audit,database/supabase-performance-optimizer,setup/setup-docker-containers,setup/setup-development-environment,database/supabase-migration-assistant,database/supabase-type-generator,nextjs-vercel/nextjs-component-generator,utilities/prime,orchestration/resume,utilities/directory-deep-dive,utilities/debug-error,testing/generate-test-cases,database/supabase-security-audit,database/supabase-realtime-monitor,performance/optimize-database-performance,performance/optimize-api-performance,utilities/architecture-scenario-explorer,testing/write-tests,testing/test-coverage,performance/performance-audit,project-management/project-health-check,security/dependency-audit,team/memory-spring-cleaning,documentation/troubleshooting-guide,team/session-learning-capture" \
  --setting "mcp/enable-all-project-servers,telemetry/disable-telemetry" \
  --hook "automation/dependency-checker,development-tools/nextjs-code-quality-enforcer,development-tools/change-tracker,security/security-scanner,post-tool/run-tests-after-changes,development-tools/file-backup,testing/test-runner,automation/agents-md-loader,automation/telegram-detailed-notifications,security/file-protection,pre-tool/backup-before-edit,automation/build-on-change,automation/deployment-health-monitor,automation/telegram-error-notifications,automation/simple-notifications,performance/performance-monitor" \
  --mcp "devtools/context7,integration/memory-integration,browser_automation/playwright-mcp-server,web/web-fetch,integration/github-integration,database/postgresql-integration,database/supabase,devtools/chrome-devtools,devtools/box,browser_automation/browser-use-mcp-server,deepgraph/deepgraph-nextjs,deepgraph/deepgraph-react,deepgraph/deepgraph-typescript,devtools/firecrawl,devtools/imagesorcery,database/postgresql-documentation,devtools/testsprite,filesystem/filesystem-access,devtools/markitdown,browser_automation/playwright-mcp" \
  --skill "development/skill-creator,creative-design/theme-factory,productivity/file-organizer"

# ===== الخطوة 3: تثبيت MCP الإضافي =====
# sequential-thinking
npx -y @modelcontextprotocol/server-sequential-thinking

# claude-mem
# راجع: https://github.com/thedotmack/claude-mem

# ===== الخطوة 4: التحقق =====
npx claude-code-templates@latest --health-check
```

---

## 10. 🎯 مصفوفة الأولويات الإجمالية

| المكوّن | الأولوية | المرحلة الأولى لاستخدامه |
|---------|---------|--------------------------|
| context7 MCP | 🔴 حرج | المرحلة 0 |
| supabase MCP | 🔴 حرج | المرحلة 2 |
| sequential-thinking MCP | 🔴 حرج | المرحلة 0 |
| filesystem-access MCP | 🔴 حرج | المرحلة 0 |
| Superpowers Plugin | 🔴 حرج | قبل البدء |
| ultra-think Command | 🔴 حرج | المرحلة 1 |
| backend-architect Agent | 🔴 حرج | المرحلة 1 |
| typescript-pro Agent | 🔴 حرج | المرحلة 1 |
| security-scanner Hook | 🟡 عالي | المرحلة 0 |
| supabase-security-audit Command | 🟡 عالي | المرحلة 2 |
| memory-integration MCP | 🟡 عالي | المرحلة 0 |
| antigravity 860+ Skills | 🟡 عالي | كل المراحل |

---

*آخر تحديث: فبراير 2026 | المشروع: MoradBot SaaS*
