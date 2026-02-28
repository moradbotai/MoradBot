# كيفية تكييف adk-samples لـ MoradBot

**المصدر الأصلي:** Google ADK Customer Service Agent (Apache 2.0)
**الهدف:** بناء `@moradbot/ai-orchestrator` — وحدة الذكاء الاصطناعي لـ MoradBot

---

## 1. الملفات المنسوخة

```
reference/original/
├── agent.ts                    # تعريف LlmAgent الرئيسي + تجميع الأدوات
├── config.ts                   # إعدادات النموذج والبيئة
├── prompts.ts                  # System prompts (GLOBAL_INSTRUCTION + INSTRUCTION)
├── tools/
│   ├── function_tools.ts       # Zod schemas + FunctionTool wrappers
│   └── tools.ts                # تنفيذ الأدوات الفعلي (mock implementations)
├── shared_libraries/
│   └── callbacks.ts            # Lifecycle hooks (beforeModel, beforeAgent, beforeTool, afterTool)
└── entities/
    └── customer.ts             # نموذج بيانات العميل مع toJson()
```

---

## 2. التعديلات المطلوبة لكل ملف

### `agent.ts` ← `src/agent.ts`

**المشكلة:** يستخدم `@google/adk` و `LlmAgent` — نحن لا نستخدم Google ADK.

**التعديلات:**

```typescript
// احذف: import { LlmAgent, InMemoryRunner } from "@google/adk"
// احذف: import { createUserContent, Part } from "@google/genai"

// أضف: OpenRouter HTTP client مباشرة
// أضف: استيراد Store context (store_id, faq_entries, product_snapshots)

// استبدل: 12 tool (cart, plants, CRM...) → 3 tools فقط:
//   - search_faq_answer
//   - search_products
//   - escalate_to_merchant

// استبدل: LlmAgent class → دالة createMoradBotAgent(storeContext: StoreContext)
// احتفظ بـ: نمط Callbacks (beforeAgent, beforeTool, afterTool, beforeModel)
```

---

### `config.ts` ← `src/config.ts`

**المشكلة:** يشير إلى Google Cloud/Vertex AI.

**التعديلات:**

```typescript
// احذف: CLOUD_PROJECT, CLOUD_LOCATION, GENAI_USE_VERTEXAI, GOOGLE_API_KEY

// أضف:
export class MoradBotConfig {
  model: string = "google/gemini-2.0-flash-001";  // OpenRouter model ID
  fallbackModels: string[] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet"
  ];
  openRouterBaseUrl: string = "https://openrouter.ai/api/v1";
  maxTokens: number = 500;             // ردود قصيرة للـ FAQ
  temperature: number = 0.3;           // أقل إبداعية، أكثر دقة
  timeoutMs: number = 8000;           // Chat Timeout = 8 ثوان (SRD)
  maxMessagesPerConversation: number = 20;
}
```

---

### `prompts.ts` ← `src/prompts.ts`

**المشكلة:** مكتوب لـ "Cymbal Home & Garden" باللغة الإنجليزية.

**التعديلات:**

```typescript
// احذف: Customer entity من GLOBAL_INSTRUCTION
// احذف: كل المحتوى الإنجليزي المتعلق بالنباتات والخصومات

// GLOBAL_INSTRUCTION الجديد يحتوي:
//   - اسم المتجر + بيانات FAQ (تُمرر ديناميكياً من Store context)
//   - قائمة المنتجات الحالية (من product_snapshots)

// INSTRUCTION الجديد يشمل قواعد MoradBot:
//   - الرد بالعربية فقط
//   - الإفصاح عن الذكاء الاصطناعي في أول رسالة
//   - التصعيد عند الفشل في الإجابة (بعد 2 محاولة)
//   - لا تخمين أسعار أو معلومات مخزون غير موجودة
//   - لا مناقشة منافسين أو موضوعات خارج نطاق المتجر
```

---

### `tools/function_tools.ts` ← `src/tools/function_tools.ts`

**المشكلة:** 12 أداة لا علاقة لها بـ MoradBot (cart، plants، CRM...).

**التعديلات — 3 أدوات فقط:**

```typescript
// احذف: جميع الأدوات الـ 12 الأصلية

// أداة 1: البحث في FAQ
const SearchFaqInput = z.object({
  query: z.string().describe("سؤال الزائر بالعربية"),
  category: z.enum(["shipping", "payment", "returns", "contact", "other"]).optional(),
});
export const searchFaqTool = new FunctionTool(searchFaq, {
  name: "search_faq_answer",
  description: "البحث عن إجابة لسؤال الزائر في قاعدة الأسئلة الشائعة للمتجر",
  inputSchema: SearchFaqInput,
});

// أداة 2: البحث في المنتجات
const SearchProductsInput = z.object({
  query: z.string().describe("اسم المنتج أو وصفه"),
  limit: z.number().default(5),
});
export const searchProductsTool = new FunctionTool(searchProducts, {...});

// أداة 3: التصعيد إلى التاجر
const EscalateInput = z.object({
  reason: z.enum(["no_answer_found", "complex_issue", "customer_request"]),
  summary: z.string().describe("ملخص المشكلة للتاجر"),
});
export const escalateTool = new FunctionTool(escalateToMerchant, {...});
```

---

### `tools/tools.ts` ← `src/tools/tools.ts`

**المشكلة:** يحتوي على mock implementations لـ Salesforce، cart، plants.

**التعديلات:**

```typescript
// احذف: جميع الدوال الـ 12 الأصلية

// أضف: implementations تقرأ من Store context (المُمرر في closure):
export function createTools(storeContext: StoreContext) {
  return {
    searchFaq: async (input) => {
      // بحث في faq_entries المحملة مسبقاً في context
      // يعيد: { found: boolean, answer?: string, category?: string }
    },
    searchProducts: async (input) => {
      // بحث في product_snapshots المحملة مسبقاً
      // يعيد: قائمة منتجات مطابقة (اسم، سعر، متاح)
    },
    escalateToMerchant: async (input) => {
      // يعيد: { escalated: true, ticket_id }
      // المعالجة الفعلية تتم في route handler
    }
  };
}
```

---

### `shared_libraries/callbacks.ts` ← `src/shared_libraries/callbacks.ts`

**المشكلة:** يستخدم `@google/adk` types + Rate limit مبني على RPM quota.

**التعديلات:**

```typescript
// احذف: import من "@google/adk"
// احذف: rateLimitCallback (Rate limiting يتم في Hono middleware)

// احتفظ بالنمط، عدّل الأنواع:
export interface CallbackContext {
  storeId: string;
  visitorId: string;
  ticketId: string;
  sessionState: Record<string, unknown>;
}

// beforeAgent: تسجيل بداية المحادثة في audit_log
// beforeModel: تحقق من reply_count < plan.bot_reply_limit
// beforeTool: تسجيل استخدام الأداة
// afterTool: تحديث session state بنتيجة الأداة
```

---

### `entities/customer.ts` ← لا يُستخدم، يُستبدل بـ 3 entities

```typescript
// احذف: Customer entity (يعتمد على Salesforce CRM)

// أضف في src/entities/:

// store.ts — بيانات المتجر في Session
export class StoreContext {
  storeId: string;
  storeName: string;
  planName: "basic" | "mid" | "premium";
  botReplyLimit: number;
  currentReplyCount: number;
  faqEntries: FaqEntry[];
  productSnapshots: ProductSnapshot[];
  toJson(): string { ... }
}

// session.ts — حالة المحادثة
export class ConversationSession {
  ticketId: string;
  visitorId: string;
  messages: Message[];
  isEscalated: boolean;
  consentGiven: boolean;
  toJson(): string { ... }
}
```

---

## 3. الأنماط المحفوظة من المصدر الأصلي

| النمط | يُحفظ | السبب |
|-------|-------|-------|
| Zod validation للأدوات | ✅ | Type safety + OpenRouter function calling |
| Callbacks system (4 hooks) | ✅ | Rate limiting، audit logging، state management |
| GLOBAL + INSTRUCTION prompts | ✅ | فصل السياق الديناميكي عن القواعد الثابتة |
| Mock implementations | ✅ | يسمح بالتطوير بدون Supabase |
| toJson() على entities | ✅ | حقن السياق في System Prompt |
| FunctionTool wrapper pattern | ✅ | متوافق مع OpenRouter function calling |

---

## 4. الهيكل النهائي المستهدف (`src/`)

```
src/
├── index.ts              # export: createMoradBotAgent, processMessage
├── agent.ts              # MoradBotAgent — OpenRouter integration
├── config.ts             # MoradBotConfig
├── prompts.ts            # buildSystemPrompt(storeContext) → string
├── tools/
│   ├── function_tools.ts # 3 FunctionTool definitions (Zod schemas)
│   └── tools.ts          # createTools(storeContext) → implementations
├── shared_libraries/
│   └── callbacks.ts      # 4 lifecycle callbacks (بدون ADK types)
└── entities/
    ├── store.ts           # StoreContext entity
    └── session.ts         # ConversationSession entity
```

---

## 5. خطة التنفيذ

**Phase 5** — بعد اكتمال Salla Client (Phase 4):

1. إنشاء `StoreContext` و `ConversationSession` entities
2. بناء `buildSystemPrompt()` مع حقن FAQ + Products
3. تنفيذ 3 أدوات مع mock بيانات
4. OpenRouter HTTP client مع fallback chain
5. تكامل مع route `/api/chat`
6. اختبارات Vitest

---

*آخر تحديث: 18 فبراير 2026*
