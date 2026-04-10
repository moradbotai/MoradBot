# مراد بوت — الميزات المستقبلية
**المصدر:** مُجمَّع من `879.md` | **التاريخ:** أبريل 2026 | **الإصدار:** 1.0

---

## نظرة عامة

هذه الوثيقة تجمع الميزات المستقبلية المخططة لمراد بوت بعد MVP، مع ملاحظات تقنية حول كيفية تطبيق كل ميزة باستخدام معمارية **LangGraph**.

**إجمالي الميزات:** 25 ميزة مُصنَّفة في 5 فئات

---

## الفئة أ — ذكاء اصطناعي متقدم (LangGraph يُنفّذها مباشرة)

### 1. كشف التعارضات في المعرفة
**الوصف:** وكيل خلفي يراقب قاعدة المعرفة ويكتشف التناقضات تلقائياً (مثل: سعر مختلف في صفحتين، شروط شحن متعارضة).
**التطبيق بـ LangGraph:** `conditional_edges` تُشغّل وكيل التدقيق بشكل دوري عبر Cron.
**المرحلة المقترحة:** Phase 6

### 2. وكيل نجاح العملاء (Customer Success Agent)
**الوصف:** إذا رد البوت بـ"لا أعرف" 3 مرات في موضوع معيّن، يرسل تنبيهاً للتاجر مع اقتراح لإضافة المعلومة لقاعدة المعرفة. يخلق "حلقة تعلم" بين المحادثات والتحسين.
**التطبيق بـ LangGraph:**
```
[Cron: كل ساعة]
  → [analyze_unknown_responses]
  → [group_by_topic]
  → [threshold_check: > 3 مرات؟]
  → [generate_knowledge_gap_report]
  → [send_alert_to_merchant] ← Resend email
```
**المرحلة المقترحة:** Phase 6

### 3. وكيل خدمة العملاء الاستباقي (Proactive CS Agent)
**الوصف:** لا ينتظر العميل ليسأل "أين طلبي؟". الوكيل يراقب شركة الشحن، وإذا لاحظ تأخيراً، يرسل رسالة اعتذار مع كود خصم قبل أن يغضب العميل.
**التطبيق بـ LangGraph:**
```
[Webhook: Salla → n8n → /api/webhook/order]
  → [monitor_shipping_agent]
  → [detect_delay: > X ساعة]
  → [generate_apology_message] ← بأسلوب التاجر
  → [send_to_customer]
```
**المرحلة المقترحة:** Phase 6

### 4. وكيل المبيعات الإضافية (Upselling Agent)
**الوصف:** يحلل سلة مشتريات العميل، وإذا وجد منتجاً يكمل ما اشتراه، يرسل عرضاً مخصصاً فوراً لرفع متوسط قيمة الطلب.
**التطبيق بـ LangGraph:** Tool-calling Agent يحلل السلة ويقترح منتجات من `product_snapshots`.
**المرحلة المقترحة:** Phase 7

### 5. GraphRAG (Microsoft)
**الوصف:** استخدام تقنية GraphRAG لبناء قاعدة معرفة ذكية تربط الكيانات والعلاقات (منتج → سياسة الشحن → مدينة) لردود أكثر دقة.
**التطبيق بـ LangGraph:**
```
[retrieve_knowledge] ← Node في الـ graph
  ├── Standard RAG: Supabase Vector (pgvector) ← المرحلة الأولى
  └── GraphRAG: Microsoft GraphRAG Index ← المرحلة الثانية
```
**ملاحظة:** ابدأ بـ RAG عادي → أضف GraphRAG عند الحاجة لدقة أعلى. الكود مفتوح المصدر لكن يتطلب LLM API.
**المرحلة المقترحة:** Phase 7

### 6. الرد الصوتي (Voice Responses)
**الوصف:** البوت يستقبل رسائل صوتية ويرد صوتياً.
**التطبيق بـ LangGraph:**
```
[receive_audio]
  → [STT: Whisper/ElevenLabs]
  → [MoradBot Chat Graph]
  → [TTS: ElevenLabs/Azure]
  → [stream_audio_response]
```
**المرحلة المقترحة:** Phase 7

### 7. توليد صور توضيحية
**الوصف:** البوت يُنشئ صوراً توضيحية للعميل إذا لم يفهم وصفاً نصياً.
**التطبيق بـ LangGraph:** Tool Node يستدعي DALL-E/Flux API.
**المرحلة المقترحة:** Phase 7

### 8. هندسة السياق (Context Engineering)
**الوصف:** إدارة ذكية لـ context window لتحسين جودة الردود مع تقليل التكلفة.
**التطبيق بـ LangGraph:** State management متقدم يتحكم في ما يُرسل للـ LLM.
**المرحلة المقترحة:** Phase 5 (تحسين مستمر)

---

## الفئة ب — تحسينات التجربة (LangGraph + مكونات أخرى)

### 9. المنتج يقود نفسه بنفسه (Product-Led Onboarding)
**الوصف:** بدلاً من لوحة تحكم معقدة، مساعد مراد يسأل التاجر: "أبدأ بالشحن أو المرتجعات؟" ويبني قالب FAQ مخصص تلقائياً.
**التطبيق بـ LangGraph:**
```python
@entrypoint(checkpointer=PostgresSaver(...))
async def onboarding_workflow(inputs):
    site_data = await analyze_merchant_site(inputs["store_url"])
    preferences = interrupt({"question": "من أين نبدأ؟"})
    faq_template = await build_faq_template(site_data, preferences)
    return {"faq_template": faq_template}
```
**المرحلة المقترحة:** Phase 6

### 10. معالج الإعداد (Onboarding Wizard)
**الوصف:** خطوات ترحيبية عند تسجيل التاجر: "1) اربط سلة → 2) أضف FAQ → 3) عيّن ساعات العمل".
**التطبيق بـ LangGraph:** Multi-step StateGraph مع checkpointing.
**المرحلة المقترحة:** Phase 6

### 11. سياق الصفحة (Page Context Awareness)
**الوصف:** البوت يعرف في أي صفحة العميل (منتج/سلة/سياسة) ويُكيّف ردوده.
**التطبيق بـ LangGraph:**
```javascript
// Widget يُرسل مع كل رسالة:
{ "message": "...", "context": { page_url, page_type, product_id } }
// يدخل الـ State مباشرة
```
**المرحلة المقترحة:** Phase 5 (مع Widget UI)

### 12. التعرف الذاتي التلقائي
**الوصف:** بمجرد إدخال التاجر رابط موقعه، الأداة تتعرف تلقائياً على بيانات المتجر وتملأ المعلومات.
**التطبيق:** Firecrawl + LangGraph retrieve node.
**المرحلة المقترحة:** Phase 6

### 13. تدريب مخصص لكل تاجر
**الوصف:** كل تاجر يُخصص system prompt للبوت الخاص به (لهجة، أسلوب، معلومات خاصة).
**التطبيق بـ LangGraph:** `configurable` في الـ graph يحقن `merchant_config`.
**المرحلة المقترحة:** Phase 5 (أساسي)

### 14. إشعار العميل عن المنتج
**الوصف:** إذا سأل العميل عن منتج غير متوفر، البوت يطلب بريده ويرسل إشعاراً عند توفره.
**التطبيق بـ LangGraph:** Tool Node يُرسل email عبر Resend عند تغيير حالة المخزون.
**المرحلة المقترحة:** Phase 6

### 15. اختبارات A/B
**الوصف:** اختبار نسختين من شخصية البوت (لهجة ودية vs. رسمية) وقياس أيهما يحقق مبيعات أعلى.
**ملاحظة مهمة:** يحتاج بيانات محادثات كافية (عشرات التجار + مئات المحادثات يومياً) قبل أن تعطي نتائج علمية.
**التطبيق بـ LangGraph:** `conditional_edge` يوزع بين prompt versions.
**المرحلة المقترحة:** Phase 8 (بعد جمع بيانات كافية)

---

## الفئة ج — التكاملات الخارجية

### 16. ربط WhatsApp
**الوصف:** البوت يرد على استفسارات عبر WhatsApp Business.
**التطبيق:** WhatsApp Business API يستقبل → LangGraph يعالج.
**المرحلة المقترحة:** Phase 7

### 17. ربط Discord
**الوصف:** تكامل مع Discord للدعم.
**التطبيق:** Discord Bot SDK → LangGraph.
**المرحلة المقترحة:** Phase 7

### 18. ربط البريد الإلكتروني
**الوصف:** خدمة العملاء على بريد المتجر.
**التطبيق بـ LangGraph:** Tool Node يقرأ/يرسل عبر Gmail API أو Resend.
**المرحلة المقترحة:** Phase 7

### 19. Webhooks (إشعارات تلقائية من سلة)
**الوصف:** بدلاً من سؤال سلة دورياً، سلة ترسل إشعاراً عند أي حدث (منتج جديد، طلب جديد).
**التطبيق:**
```
Salla Webhook → n8n يستقبل → يحدث Supabase → LangGraph يعالج
```
**المرحلة المقترحة:** Phase 6

### 20. مزامنة مع أنظمة خارجية
**الوصف:** مزامنة مع Notion أو أنظمة إدارة خارجية.
**التطبيق بـ LangGraph:** Tool Node يستدعي Notion API.
**المرحلة المقترحة:** Phase 8

---

## الفئة د — التحليلات والتقارير

### 21. تقارير حسب الباقة
**الوصف:** تقارير شهرية (الانطلاق) → أسبوعية (النمو) → يومية (المتمكّن).
**التطبيق:** Cron Jobs + Supabase queries + LangGraph وكيل يُلخّص.
**المرحلة المقترحة:** Phase 7

### 22. تحليل المنافسين
**الوصف:** رصد ما يفعله المنافسون وتحليل السوق.
**أدوات مقترحة:** Competely، Crayon، SEMrush، Ahrefs، SimilarWeb.
**التطبيق:** Firecrawl + PostHog (خارج نطاق LangGraph الأساسي).
**المرحلة المقترحة:** Phase 9 (بعد الاستقرار)

### 23. حاسبة ROI للتاجر
**الوصف:** يدخل التاجر عدد استفساراته/اليوم → يطلع عليه ساعات موفرة/شهر وقيمة الوقت.
**التطبيق:** صفحة ثابتة في Landing Page أو Dashboard.
**المرحلة المقترحة:** Phase 6 (Landing Page)

---

## الفئة هـ — البنية التحتية والأداء

### 24. وكلاء فرعيين متخصصين (Multi-Agent Architecture)
**الوصف:** بنية متعددة الوكلاء حيث كل وكيل متخصص في مهمة محددة.
**التطبيق بـ LangGraph:** Supervisor Pattern + Swarm Pattern (حزم جاهزة: `langgraph-supervisor`, `langgraph-swarm`).
**المرحلة المقترحة:** Phase 6

### 25. توزيع الأحمال
**الوصف:** توزيع الأحمال على عدة سيرفرات لتحقيق High Availability.
**التطبيق:** Cloudflare Load Balancer / K8s (خارج نطاق LangGraph).
**المرحلة المقترحة:** Phase 9

---

## مكتبة قوالب FAQ الجاهزة

قوالب جاهزة حسب المجال بصياغات عربية مناسبة + أسئلة شائعة "مُجرَّبة":
- **جمال وعناية:** تكوين المنتج، الحساسية، طريقة الاستخدام
- **أزياء:** المقاسات، سياسة الإرجاع، مدة التوصيل
- **مكملات غذائية:** تاريخ الانتهاء، الجرعة، التخزين
- **إكسسوارات إلكترونية:** الضمان، التوافق، التركيب

**المرحلة المقترحة:** Phase 5 (مع أول إعداد للتاجر)

---

## جلسة إعداد سريعة

**30 دقيقة** مع التاجر عند الاشتراك في باقة "النمو" فأعلى لضمان رؤية نتيجة خلال أسبوع:
1. مراجعة الأسئلة الشائعة الحالية
2. بناء أول 10 إجابات FAQ
3. اختبار البوت على سيناريوهات حقيقية

---

## ملاحظات تطبيق LangGraph

### جداول التطبيق المقترحة

```
Phase 5 — MVP AI (الأساس):
packages/ai-orchestrator/
├── src/graphs/chat_graph.py      ← StateGraph الأساسي
├── src/nodes/retrieve.py         ← FAQs + Products
├── src/nodes/classify.py         ← Intent classification
├── src/nodes/generate.py         ← OpenRouter → Gemini 2.0 Flash
├── src/nodes/validate.py         ← Prompt injection guard
├── src/nodes/escalate.py         ← Trigger escalation
├── src/state.py                  ← MoradBotState TypedDict
├── src/config.py                 ← Per-merchant configuration
└── src/server.py                 ← FastAPI endpoint

Phase 6 — Enhanced Agents:
├── src/agents/customer_success.py   ← مراقبة جودة
├── src/agents/proactive_cs.py       ← تتبع الشحن
└── src/agents/onboarding.py         ← Onboarding Wizard

Phase 7 — Advanced Intelligence:
├── src/agents/upselling.py          ← تحليل السلة
├── src/agents/voice_pipeline.py     ← STT → LLM → TTS
└── src/agents/graphrag_retriever.py ← Microsoft GraphRAG
```

### الميزات الـ 25 في سطر واحد

| # | الميزة | فئة LangGraph | المرحلة |
|---|--------|--------------|---------|
| 1 | كشف التعارضات | مثالية | 6 |
| 2 | وكيل نجاح العملاء | مثالية | 6 |
| 3 | وكيل استباقي | مثالية | 6 |
| 4 | وكيل Upselling | مثالية | 7 |
| 5 | GraphRAG | مثالية | 7 |
| 6 | رد صوتي | مثالية | 7 |
| 7 | توليد صور | مثالية | 7 |
| 8 | هندسة السياق | مثالية | 5 |
| 9 | Product-Led Onboarding | مثالية | 6 |
| 10 | Onboarding Wizard | مثالية | 6 |
| 11 | سياق الصفحة | مثالية | 5 |
| 12 | تعرف ذاتي | مثالية | 6 |
| 13 | تدريب مخصص | مثالية | 5 |
| 14 | إشعار المخزون | مثالية | 6 |
| 15 | A/B Testing | مثالية | 8 |
| 16 | WhatsApp | تكمّل | 7 |
| 17 | Discord | تكمّل | 7 |
| 18 | البريد الإلكتروني | تكمّل | 7 |
| 19 | Webhooks | تكمّل | 6 |
| 20 | Notion/خارجي | تكمّل | 8 |
| 21 | تقارير متدرجة | تكمّل | 7 |
| 22 | تحليل المنافسين | خارج LangGraph | 9 |
| 23 | حاسبة ROI | خارج LangGraph | 6 |
| 24 | Multi-Agent | مثالية | 6 |
| 25 | توزيع الأحمال | خارج LangGraph | 9 |
