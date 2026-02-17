# تحليل المشروع المرجعي - Customer Service Agent

**المصدر:** `adk-samples/typescript/agents/customer_service`
**التاريخ:** 17 فبراير 2026
**الهدف:** استخلاص الأنماط والبنية المفيدة لـ MoradBot

---

## 1. بنية الـ Agent Architecture

### الهيكل العام
```
customer_service/
├── agent.ts              # نقطة الدخول الرئيسية
├── config.ts             # الإعدادات المركزية
├── prompts.ts            # تنظيم الـ System Prompts
├── entities/
│   └── customer.ts       # نماذج البيانات
├── tools/
│   ├── function_tools.ts # تعريفات الأدوات (Zod schemas)
│   └── tools.ts          # التنفيذ الفعلي للأدوات
└── shared_libraries/
    └── callbacks.ts      # Lifecycle callbacks
```

### النمط المستخدم
```typescript
export const rootAgent = new LlmAgent({
  model: config.agentSettings.model,
  name: config.agentSettings.name,
  instruction: COMBINED_INSTRUCTION,
  tools: [...toolsArray],
  beforeToolCallback,
  afterToolCallback,
  beforeAgentCallback,
  beforeModelCallback,
});
```

**الدروس المستفادة:**
- ✅ فصل واضح بين المكونات (separation of concerns)
- ✅ Agent يُعرَّف مرة واحدة ويُصدَّر
- ✅ كل المنطق موجود في callbacks وtools

---

## 2. نظام الـ Tools System

### طبقتين منفصلتين

#### الطبقة الأولى: التعريف (function_tools.ts)
```typescript
import { z } from "zod";
import { FunctionTool } from "@google/adk";

// 1. تعريف Input Schema بـ Zod
const AccessCartInformationInput = z.object({
  customerId: z.string().describe("The ID of the customer."),
});

// 2. إنشاء FunctionTool wrapper
export const accessCartInformationTool = new FunctionTool({
  name: "accessCartInformation",
  description: "Accesses cart information for a customer.",
  parameters: AccessCartInformationInput,
  execute: accessCartInformation, // من tools.ts
});
```

#### الطبقة الثانية: التنفيذ (tools.ts)
```typescript
export function accessCartInformation({
  customerId,
}: {
  customerId: string;
}): Cart {
  console.info(`Accessing cart information for customer ID: ${customerId}`);

  // تنفيذ المنطق الفعلي
  const mockCart: Cart = { ... };
  return mockCart;
}
```

**الفوائد:**
- ✅ Type safety كامل مع Zod
- ✅ وثائق واضحة في descriptions
- ✅ سهولة الاختبار (test implementation بشكل منفصل)
- ✅ إمكانية mock البيانات في مرحلة التطوير

**أمثلة الأدوات المهمة لـ MoradBot:**
1. `accessCartInformation` - الوصول للسلة ← **نحتاج نسخة لـ FAQ data**
2. `modifyCart` - تعديل السلة ← **نحتاج escalation**
3. `getProductRecommendations` - اقتراحات منتجات ← **البحث في Products**
4. `schedulePlantingService` - جدولة خدمة ← **ليس في MVP**

---

## 3. إدارة المحادثة والـ State Management

### Session State Pattern
```typescript
// في beforeAgentCallback
export function beforeAgent(callbackContext: CallbackContext): undefined {
  if (!callbackContext.state.has("customer_profile")) {
    const customer = Customer.getCustomer("123");
    if (customer) {
      callbackContext.state.set("customer_profile", customer.toJson());
    }
  }
}

// في beforeToolCallback
const [valid, err] = validateCustomerId({
  customerId: args["customer_id"],
  sessionState: toolContext.state,
});
```

**النمط:**
- `callbackContext.state.set(key, value)` - تخزين
- `callbackContext.state.get<T>(key)` - استرجاع
- `callbackContext.state.has(key)` - التحقق من وجود

**تطبيق على MoradBot:**
```typescript
// نحتاج تخزين:
state.set("store_id", storeId);           // معرف المتجر
state.set("faq_data", faqJson);           // بيانات FAQ
state.set("products_snapshot", products); // آخر sync للمنتجات
state.set("visitor_id", visitorId);       // معرف الزائر (cookie)
state.set("ticket_id", ticketId);         // معرف Ticket الحالي
state.set("clarification_count", 0);      // عداد محاولات التوضيح
```

---

## 4. Callbacks System - الأنماط الأربعة

### 1. beforeModelCallback - Rate Limiting
```typescript
export async function rateLimitCallback({
  context: callbackContext,
  request: llmRequest,
}): Promise<any> {
  const now = Date.now() / 1000;

  if (!callbackContext.state.has("timer_start")) {
    callbackContext.state.set("timer_start", now);
    callbackContext.state.set("request_count", 1);
    return undefined;
  }

  const requestCount = (callbackContext.state.get<number>("request_count") || 0) + 1;

  if (requestCount > RPM_QUOTA) {
    // Sleep or reject
  }
}
```

**الاستخدام في MoradBot:**
- تطبيق rate limiting للحماية من spam
- تتبع usage (bot replies count) للـ billing

### 2. beforeToolCallback - Validation & Preprocessing
```typescript
export function beforeTool({
  tool,
  args,
  context: toolContext,
}): Record<string, any> | undefined {
  // 1. Normalize inputs
  const lowercasedArgs = lowercaseValue(args);

  // 2. Validate
  if ("customer_id" in args) {
    const [valid, err] = validateCustomerId({
      customerId: args["customer_id"],
      sessionState: toolContext.state,
    });
    if (!valid && err) {
      return { error: err };
    }
  }

  // 3. Short-circuit logic
  if (tool.name === "sync_ask_for_approval") {
    if (amount <= 10) {
      return { status: "approved", message: "Auto-approved" };
    }
  }
}
```

**تطبيق على MoradBot:**
- التحقق من `store_id` في كل tool call
- normalization للنصوص العربية
- منع tool calls غير مصرح بها

### 3. afterToolCallback - Post-processing
```typescript
export function afterTool({
  tool,
  args,
  context,
  response: toolResponse,
}): Record<string, unknown> | undefined {
  if (tool.name === "approve_discount") {
    if (toolResponse?.status === "ok") {
      console.debug("Applying discount to the cart");
      // Actually apply changes
    }
  }
}
```

**تطبيق على MoradBot:**
- Logging لكل tool execution
- تحديث usage counters
- إضافة disclaimers للمنتجات الديناميكية

### 4. beforeAgentCallback - Initialization
```typescript
export function beforeAgent(callbackContext: CallbackContext): undefined {
  if (!callbackContext.state.has("customer_profile")) {
    const customer = Customer.getCustomer("123");
    callbackContext.state.set("customer_profile", customer.toJson());
  }
}
```

**تطبيق على MoradBot:**
- تحميل FAQ data
- تحميل Products snapshot
- تهيئة session memory

---

## 5. Entities & Data Models Pattern

### Customer Entity Structure
```typescript
export class Customer {
  // Properties
  customer_id: string;
  customer_first_name: string;
  email: string;
  phone_number: string;
  purchase_history: Purchase[];

  // Constructor
  constructor(data: Customer) {
    Object.assign(this, data);
  }

  // Methods
  toJson(): string {
    return JSON.stringify(this, null, 4);
  }

  static getCustomer(id: string): Customer | null {
    // Database lookup or mock
  }
}
```

**تطبيق على MoradBot Entities:**

#### Store Entity
```typescript
export class Store {
  store_id: string;
  salla_merchant_id: string;
  plan_tier: "basic" | "mid" | "premium";
  bot_enabled: boolean;
  usage_current_cycle: number;
  usage_limit: number;

  toJson(): string;
  static getStore(storeId: string): Store | null;
}
```

#### FAQ Entity
```typescript
export class FAQ {
  store_id: string;
  faqs: Array<{
    question_ar: string;
    answer_ar: string;
    category: string;
  }>;

  toJson(): string;
  static getFAQsForStore(storeId: string): FAQ | null;
}
```

#### Product Snapshot Entity
```typescript
export class ProductSnapshot {
  store_id: string;
  last_sync: Date;
  products: Array<{
    product_id: string;
    name_ar: string;
    price: number;
    available: boolean;
  }>;

  toJson(): string;
  static getProducts(storeId: string): ProductSnapshot | null;
}
```

---

## 6. Prompts Organization Pattern

### Two-Layer System
```typescript
// Layer 1: Global Context
export const GLOBAL_INSTRUCTION = `
The profile of the current customer is: ${defaultCustomerJson}
`;

// Layer 2: Agent Instructions
export const INSTRUCTION = `
You are "Project Pro," the primary AI assistant...

**Core Capabilities:**
1. Personalized Customer Assistance
2. Product Identification
3. Order Management
...

**Tools:**
* "access_cart_information": Retrieves cart contents
* "modify_cart": Updates cart
...

**Constraints:**
* Never mention internal tool names
* Always confirm before executing actions
`;

// Combined
const COMBINED_INSTRUCTION = `${GLOBAL_INSTRUCTION}\n\n${INSTRUCTION}`;
```

**تطبيق على MoradBot:**
```typescript
export const GLOBAL_INSTRUCTION = `
معلومات المتجر الحالي: ${storeProfileJson}
الأسئلة الشائعة للمتجر: ${faqDataJson}
آخر تحديث للمنتجات: ${lastSyncTime}
`;

export const INSTRUCTION = `
أنت مساعد AI للمتجر على منصة سلة.
مهمتك هي الإجابة على الأسئلة الشائعة فقط بناءً على بيانات المتجر.

**القدرات الأساسية:**
1. الإجابة على أسئلة توفر المنتجات
2. الإجابة على أسئلة الأسعار
3. الإجابة على أسئلة الشحن
4. الإجابة على أسئلة الدفع
5. الإجابة على أسئلة سياسة الإرجاع

**الأدوات المتاحة:**
* "search_products": البحث في المنتجات
* "get_faq_answer": الحصول على إجابة من FAQ
* "escalate_to_merchant": تصعيد السؤال للتاجر

**القيود:**
- لا يمكنك إلغاء طلبات
- لا يمكنك تعديل الأسعار
- لا يمكنك الوصول لبيانات الدفع
- يجب إضافة "حسب آخر تحديث..." للبيانات الديناميكية (السعر/التوفر)
- بعد 3 محاولات توضيح فاشلة، يجب التصعيد
`;
```

---

## 7. Configuration Pattern

### Centralized Config
```typescript
export class AgentModel {
  name: string = "customer_service_agent";
  model: string = "gemini-2.5-flash";
}

export class Config {
  agentSettings: AgentModel = new AgentModel();
  appName: string = "customer_service_app";

  CLOUD_PROJECT: string = getEnv("GOOGLE_CLOUD_PROJECT", "my_project");
  API_KEY: string = getEnv("GOOGLE_API_KEY", "");
}

// Singleton
export const config = new Config();
```

**تطبيق على MoradBot:**
```typescript
export class MoradBotConfig {
  agentSettings = {
    name: "moradbot_faq_assistant",
    model: "gemini-2.0-flash", // or from OpenRouter
  };

  // OpenRouter
  openRouterApiKey: string = getEnv("OPENROUTER_API_KEY", "");
  openRouterBaseUrl: string = "https://openrouter.ai/api/v1";

  // Supabase
  supabaseUrl: string = getEnv("SUPABASE_URL", "");
  supabaseKey: string = getEnv("SUPABASE_ANON_KEY", "");

  // Salla
  sallaClientId: string = getEnv("SALLA_CLIENT_ID", "");
  sallaClientSecret: string = getEnv("SALLA_CLIENT_SECRET", "");

  // Rate Limits
  rateLimits = {
    visitorPerMinute: 10,
    storePerMinute: 100,
  };
}
```

---

## 8. أنماط مهمة أخرى

### Mock Pattern للتطوير
```typescript
// MOCK API RESPONSE - Replace with actual API call
const mockCart: Cart = {
  items: [...],
  subtotal: 25.98,
};
return mockCart;
```

**الفائدة:**
- ✅ التطوير بدون backend جاهز
- ✅ اختبار سريع لـ Agent logic
- ✅ سهولة استبدال بـ real implementation لاحقاً

### Status Response Pattern
```typescript
export interface StatusResponse {
  status: string;
  message?: string;
  [key: string]: any;
}

// Usage
return { status: "success", message: "Cart updated" };
return { status: "rejected", message: "Discount too large" };
```

---

## 9. الدروس المستفادة لـ MoradBot

### ما يجب تطبيقه ✅
1. **فصل واضح بين المكونات** - config, prompts, tools, entities, callbacks
2. **Zod schemas للـ tools** - type safety كامل
3. **Session state للمحادثة** - تخزين السياق
4. **Callbacks للـ validation والـ rate limiting**
5. **Mock pattern** - للتطوير السريع
6. **Entity classes** - مع toJson() وstatic getters
7. **Two-layer prompts** - global context + instructions

### ما يجب تكييفه 🔄
1. **Google ADK → OpenRouter** - نستخدم OpenRouter بدل Vertex AI
2. **Customer entity → Store/Visitor entities** - نماذج مختلفة
3. **Cart tools → FAQ/Products tools** - أدوات مختلفة
4. **Rate limiting** - حسب visitor + store level
5. **Validation** - التحقق من store_id بدل customer_id

### ما لا نحتاجه ❌
1. **Video/Multimodal** - نصوص فقط في MVP
2. **Appointment scheduling** - ليس في نطاق MVP
3. **CRM integration** - ليس في MVP
4. **Discount management** - ليس في نطاق FAQ automation

---

## الخلاصة

المشروع المرجعي يوفر بنية ممتازة يمكن تكييفها لـ MoradBot:
- **Architecture** واضح وقابل للتوسع
- **Separation of concerns** محكم
- **Type safety** كامل مع Zod
- **State management** بسيط وفعال
- **Callbacks** للتحكم الدقيق في lifecycle

**الخطوة التالية:** تصميم قاعدة البيانات بناءً على هذه الأنماط.
