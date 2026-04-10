# مراد بوت — رؤية LangGraph
**النوع:** مجلد استراتيجي | **التاريخ:** أبريل 2026 | **الحالة:** مرجع نشط

---

## الغرض

هذا المجلد يحتوي على نسخة محدَّثة من وثائق المشروع تعكس قرار اعتماد **LangGraph** كمحرك ذكاء اصطناعي لمراد بوت. الوثائق الأصلية في `docs_v2/` لم تُعدَّل — هذا المجلد هو النسخة الاستراتيجية المحدَّثة.

**القرار الجوهري:**
> مراد بوت = Hono (طبقة الخدمة) + LangGraph Python (طبقة الذكاء) + Supabase (طبقة البيانات)

---

## فهرس الوثائق

| # | الوثيقة | الوصف |
|---|---------|-------|
| [01](./01_langgraph_decision.md) | قرار LangGraph | التقرير الكامل: هل يمكن بناء مراد بوت بـ LangGraph؟ |
| [02](./02_project_overview.md) | نظرة عامة على المشروع | محدَّثة لتعكس Phase 5 = LangGraph AI Service |
| [03](./03_implementation_plan.md) | خطة التنفيذ | محدَّثة: Phase 5 يصف LangGraph service بالتفصيل |
| [04](./04_architecture.md) | قرارات المعمارية | محدَّثة: يشمل هجين Hono + LangGraph |
| [05](./05_infrastructure_and_costs.md) | التقنيات والتكاليف | محدَّثة: تشمل تكلفة Python service |
| [06](./06_business_requirements.md) | متطلبات الأعمال | نسخة حالية (بدون تعديل جوهري) |
| [07](./07_market_requirements.md) | متطلبات السوق | نسخة حالية (بدون تعديل جوهري) |
| [08](./08_development_standards.md) | دليل قواعد العمل | نسخة حالية (بدون تعديل جوهري) |
| [09](./09_future_features.md) | الميزات المستقبلية | مُجمَّعة من `879.md` مع ملاحظات LangGraph |

---

## القرار الاستراتيجي في سطر واحد

LangGraph ليس بديلاً عن أي شيء موجود — هو المحرك الذي يُحوّل مراد بوت من chatbot بسيط إلى نظام وكلاء ذكي يتعلم ويتطور.

### الخارطة المعمارية

```
Cloudflare Workers (Hono)    →    LangGraph Python Service    →    Supabase
  - HTTP routing                    - FAQ Agent (Phase 5)           - Business data
  - JWT Auth                        - CS Agent (Phase 6)            - RLS policies
  - Rate limiting                   - Upsell Agent (Phase 7)        - Checkpointing
  - FAQ CRUD                        - Voice Agent (Phase 7)         - Cross-thread memory
  - Stats/Tickets                   - Audit Agent (Phase 6)
```

### حالة المراحل (أبريل 2026)

| المرحلة | الحالة | ملاحظة |
|---------|--------|--------|
| Phase 1 | ✅ مكتملة | Turborepo، Biome، الحزم الأساسية |
| Phase 2 | ✅ مكتملة | DB: 14 جدول، 8 migrations، RLS كامل |
| Phase 3 | ✅ مكتملة | Hono API: 16 endpoint، middleware |
| Phase 3.5 | ✅ مكتملة | Landing page + waitlist + PostHog |
| Phase 4 | 🔜 التالية | Salla OAuth + Product Sync |
| Phase 5 | ⏳ معلّقة | **LangGraph AI Service + Widget UI** |
| Phase 6+ | ⏳ معلّقة | Enhanced Agents، Dashboard، Billing |
