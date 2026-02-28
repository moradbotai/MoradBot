# Conversation Flow

يصف هذا المخطط مسار المحادثة الكاملة من أول رسالة حتى الإغلاق.

```mermaid
stateDiagram-v2
    [*] --> NEW_TICKET : User sends first message\n[POST /api/chat, no ticket_id]

    NEW_TICKET --> PROCESSING : ticket_id created\nFAQ + Products loaded from DB

    PROCESSING --> RESPONDING : AI found answer\n[confidence ≥ threshold]
    PROCESSING --> CLARIFICATION_1 : AI needs more info\n[missing context]

    RESPONDING --> RESOLVED : User satisfied\n[no follow-up]
    RESPONDING --> PROCESSING : User asks follow-up question\n[same ticket_id]

    CLARIFICATION_1 --> PROCESSING : User provides clarification
    CLARIFICATION_1 --> CLARIFICATION_2 : Still unclear\n[attempt_count = 2]

    CLARIFICATION_2 --> PROCESSING : User provides clarification
    CLARIFICATION_2 --> CLARIFICATION_3 : Still unclear\n[attempt_count = 3]

    CLARIFICATION_3 --> PROCESSING : User provides clarification
    CLARIFICATION_3 --> ESCALATING : 3rd attempt failed\n[escalate_to_merchant tool called]

    ESCALATING --> ESCALATED : Contact info collected\n[name + phone validated]

    ESCALATED --> MERCHANT_NOTIFIED : Real-time dashboard event\n[escalations.status = open]

    MERCHANT_NOTIFIED --> CLOSED : Merchant clicks Resolved\n[escalations.status = closed]

    RESOLVED --> CLOSED : Session timeout\n[no activity for 30 min]
    CLOSED --> [*]
```

---

## شرح الانتقالات (Conditions)

| الانتقال | الشرط |
|----------|-------|
| `NEW_TICKET → PROCESSING` | `ticket_id` غير موجود في الطلب → يُنشأ تلقائياً |
| `PROCESSING → RESPONDING` | الأداة `search_faq_answer` وجدت إجابة |
| `PROCESSING → CLARIFICATION_1` | الأداة لم تجد إجابة كافية |
| `CLARIFICATION_N → PROCESSING` | المستخدم أعاد الكتابة بتفاصيل إضافية |
| `CLARIFICATION_N → CLARIFICATION_N+1` | `attempt_count` يزيد، الذكاء الاصطناعي يطلب توضيحاً مرة أخرى |
| `CLARIFICATION_3 → ESCALATING` | `attempt_count = 3`، يُستدعى `escalate_to_merchant` tool |
| `RESPONDING → RESOLVED` | لا رسالة من المستخدم بعد رد البوت (timeout 30 دقيقة) |
| `RESPONDING → PROCESSING` | المستخدم يرسل رسالة متابعة بنفس الـ `ticket_id` |
| `ESCALATED → MERCHANT_NOTIFIED` | INSERT في `escalations` ينجح → Supabase Realtime يُطلق الحدث |
| `MERCHANT_NOTIFIED → CLOSED` | التاجر يضغط "تم الحل" في الداشبورد |

---

## بيانات المحادثة المخزنة

| الجدول | البيانات المحفوظة |
|--------|-----------------|
| `tickets` | ticket_id، store_id، visitor_id، status، created_at |
| `messages` | كل رسالة (زائر + بوت)، role، content، timestamps |
| `escalations` | بيانات التصعيد، contact info (مشفّرة)، status |
| `usage_tracking` | عدد الردود لكل متجر في الدورة الحالية |
