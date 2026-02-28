# Escalation Flow

يصف هذا المخطط عملية التصعيد بالتفصيل: من فشل الإجابة حتى إغلاق التاجر للحالة.

```mermaid
stateDiagram-v2
    [*] --> ATTEMPT_1 : AI tries to answer visitor question

    ATTEMPT_1 --> SUCCESS : Answer found in FAQ or Products
    ATTEMPT_1 --> ATTEMPT_2 : Needs clarification\n[ask_clarification called]

    ATTEMPT_2 --> SUCCESS : Answer found after clarification
    ATTEMPT_2 --> ATTEMPT_3 : Still needs clarification\n[attempt_count = 2]

    ATTEMPT_3 --> SUCCESS : Answer found after clarification
    ATTEMPT_3 --> TRIGGER_ESCALATION : Failed again\n[escalate_to_merchant tool called]\n[attempt_count = 3]

    TRIGGER_ESCALATION --> SHOW_FORM : Widget transitions to ESCALATING state\nContact form rendered

    SHOW_FORM --> COLLECTING : User starts filling form\n[name field focused]

    COLLECTING --> VALIDATE : User submits form\n[name + phone provided]

    VALIDATE --> COLLECTING : Validation failed\n[phone format invalid OR name empty]
    VALIDATE --> ENCRYPT : Validation passed\n[all required fields present]

    ENCRYPT --> STORE_DB : AES-256 encryption applied\nto phone number field

    STORE_DB --> NOTIFY_MERCHANT : INSERT INTO escalations succeeds\n[store_id, ticket_id, contact_info_encrypted]

    NOTIFY_MERCHANT --> DASHBOARD_UPDATE : Supabase Realtime broadcasts event\n[channel: store:{store_id}]
    NOTIFY_MERCHANT --> EMAIL_ALERT : Email sent to merchant\n[escalation details]

    DASHBOARD_UPDATE --> AWAITING_MERCHANT : Merchant sees new escalation\nin real-time dashboard

    AWAITING_MERCHANT --> MERCHANT_CLOSE : Merchant clicks Resolved\n[PATCH /api/escalations/:id]

    MERCHANT_CLOSE --> AUDIT_LOG : Log closure event\n[merchant_id, timestamp, action]

    AUDIT_LOG --> [*] : Complete

    SUCCESS --> [*]
```

---

## ملاحظات الأمان

### تشفير البيانات الحساسة
- رقم الهاتف يُشفَّر بـ **AES-256** قبل الحفظ في `escalations.contact_info`
- مفتاح التشفير يُخزَّن في **Cloudflare Secrets** فقط (لا يظهر في الكود)
- البيانات المشفرة غير قابلة للقراءة حتى مع الوصول المباشر لقاعدة البيانات

### Audit Logging
- كل حدث في عملية التصعيد يُسجَّل في `audit_logs` مع:
  - `store_id`، `actor_id`، `action`، `timestamp`
  - `ip_address` للمستخدم (مشفرة أو مجزأة وفق PDPL)
- السجلات محمية بـ RLS — لا يمكن للتاجر حذفها

### RLS Enforcement
- كل INSERT/SELECT على `escalations` يتطلب `store_id = auth.uid()`
- التاجر يرى فقط تصعيدات متجره الخاص
- Service Role فقط (للنظام) يمكنه الوصول عبر `createSupabaseAdmin()`

---

## شرح الانتقالات

| الانتقال | التفاصيل |
|----------|----------|
| `ATTEMPT_N → SUCCESS` | `search_faq_answer` أو `search_products` أعادت نتيجة |
| `ATTEMPT_3 → TRIGGER_ESCALATION` | `escalate_to_merchant` tool يُستدعى، يُرجع `{ escalated: true }` |
| `VALIDATE → COLLECTING` | الهاتف ليس بصيغة سعودية صحيحة، أو الاسم فارغ |
| `VALIDATE → ENCRYPT` | الحقول المطلوبة جميعها حاضرة وصحيحة |
| `NOTIFY_MERCHANT → DASHBOARD_UPDATE` | Supabase Realtime `INSERT` event على channel `store:{store_id}` |
| `AWAITING_MERCHANT → MERCHANT_CLOSE` | `PATCH /api/escalations/:id` بـ `{ status: "closed" }` |
