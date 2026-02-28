# Usage Metering Flow

يصف هذا المخطط منطق حساب الاستخدام، وإرسال التحذيرات، وقطع الخدمة عند الوصول للحد الأقصى.

```mermaid
flowchart TD
    A[Bot Reply Sent Successfully] --> B[Increment current_cycle_replies\nUPDATE usage_tracking\nSET replies = replies + 1]

    B --> C{Calculate Percentage\nreplies / bot_reply_limit × 100}

    C -->|"< 80%"| E[Continue Normal Operation]
    C -->|"≥ 80% AND < 100%"| F[Send Warning Email\nto merchant]
    C -->|"= 100%"| G[Send Limit Reached Email\nto merchant]

    F --> F2{Warning sent\nthis cycle already?}
    F2 -->|No| F3[Send email\nMark warning_sent = true]
    F2 -->|Yes| E
    F3 --> E

    G --> H[Set bot_disabled = true\nfor this store]

    H --> I[Widget API returns\nQUOTA_EXCEEDED error]
    I --> J[Widget transitions to\nQUOTA_EXCEEDED state]
    J --> K[Show Arabic message:\nتم الوصول إلى الحد الأقصى]

    E --> L[Continue Serving Visitors]

    style H fill:#ff6b6b,color:#fff
    style I fill:#ff6b6b,color:#fff
    style J fill:#ff6b6b,color:#fff
    style K fill:#ff6b6b,color:#fff
    style F fill:#ffd93d,color:#333
    style F3 fill:#ffd93d,color:#333
    style E fill:#6bcf7f,color:#333
    style L fill:#6bcf7f,color:#333
```

---

## خطط الاشتراك والحدود

| الخطة | `bot_reply_limit` | `sync_frequency_hours` | السعر |
|-------|-------------------|------------------------|-------|
| Basic | يُحدد بالخطة | 24 ساعة | — |
| Mid | يُحدد بالخطة | 6 ساعات | — |
| Premium | يُحدد بالخطة | 1 ساعة | — |

> القيم الفعلية مُعرَّفة في جدول `plans` في قاعدة البيانات.

---

## بيانات المتابعة في قاعدة البيانات

### جدول `usage_tracking`
```sql
-- كل صف يمثل دورة شهرية لمتجر واحد
store_id          UUID     -- ربط بالمتجر
cycle_start       DATE     -- بداية الدورة
cycle_end         DATE     -- نهاية الدورة
bot_replies_used  INTEGER  -- عدد الردود الفعلية
warning_sent      BOOLEAN  -- هل أُرسل تحذير الـ 80%؟
```

---

## منطق التحقق في Middleware (`beforeModel` Callback)

```typescript
// يُنفَّذ قبل كل طلب للـ AI
const usagePercent = (currentReplies / botReplyLimit) * 100;

if (usagePercent >= 100) {
  throw new QuotaExceededError(); // → Widget: QUOTA_EXCEEDED
}
// إذا < 100%، يستمر الطلب بشكل طبيعي
```

---

## دورة الفوترة (Billing Cycle)

- الدورة: **شهرية** تبدأ من تاريخ الاشتراك
- عند التجديد: `current_cycle_replies` يُعاد إلى صفر
- التجديد يدوي (يدفع التاجر) — لا تجديد تلقائي في MVP
