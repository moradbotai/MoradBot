# API Error Codes — MoradBot

آخر تحديث: 22 فبراير 2026

---

## نظام الترقيم

| النطاق | النوع |
|--------|-------|
| `4000–4099` | Authentication & Authorization |
| `4220–4229` | Validation |
| `4290–4299` | Rate Limiting |
| `4400–4499` | Business Logic |
| `5000–5099` | System & Infrastructure |

---

## 1. Authentication Errors (4000–4099)

| Code | HTTP | رسالة عربية | رسالة إنجليزية | Retry |
|------|------|-------------|----------------|-------|
| `4001` | 401 | معرف المتجر غير صحيح | Invalid store_id | No |
| `4002` | 401 | رمز الوصول مفقود | Missing authorization token | No |
| `4003` | 401 | انتهت صلاحية الرمز | Token expired | Yes — بعد refresh |
| `4004` | 403 | صلاحيات غير كافية | Insufficient permissions | No |
| `4005` | 401 | رمز Salla OAuth منتهي | Salla OAuth token expired | Yes — بعد إعادة ربط |

---

## 2. Rate Limiting (4290–4299)

| Code | HTTP | رسالة عربية | رسالة إنجليزية | Retry |
|------|------|-------------|----------------|-------|
| `4291` | 429 | تجاوزت الحد المسموح من الرسائل | Visitor rate limit exceeded | Yes — بعد 60s |
| `4292` | 429 | المتجر تجاوز الحد الساعي | Store hourly limit exceeded | Yes — بعد 1h |
| `4293` | 429 | انتهى الحد الشهري للاشتراك | Monthly quota exceeded | No — ترقية الخطة |

---

## 3. Validation Errors (4220–4229)

| Code | HTTP | رسالة عربية | رسالة إنجليزية | Retry |
|------|------|-------------|----------------|-------|
| `4221` | 400 | صيغة الرسالة غير صحيحة | Invalid message format | No |
| `4222` | 400 | الرسالة طويلة جداً (الحد: 1000 حرف) | Message too long (max 1000 chars) | No |
| `4223` | 400 | الحقول الإلزامية مفقودة | Required fields missing | No |
| `4224` | 400 | صيغة UUID غير صحيحة | Invalid UUID format | No |
| `4225` | 400 | URL الصفحة غير صالح | Invalid page URL | No |

---

## 4. Business Logic Errors (4400–4499)

| Code | HTTP | رسالة عربية | رسالة إنجليزية | Retry |
|------|------|-------------|----------------|-------|
| `4401` | 403 | البوت معطّل لهذا المتجر | Bot disabled for this store | No |
| `4402` | 402 | الاشتراك غير نشط | Subscription inactive or expired | No |
| `4403` | 404 | المحادثة غير موجودة | Ticket not found | No |
| `4404` | 409 | المحادثة مغلقة بالفعل | Ticket already closed | No |
| `4405` | 404 | سجل FAQ غير موجود | FAQ entry not found | No |
| `4406` | 409 | التصعيد موجود بالفعل | Escalation already exists | No |

---

## 5. System Errors (5000–5099)

| Code | HTTP | رسالة عربية | رسالة إنجليزية | Retry | تنبيه |
|------|------|-------------|----------------|-------|-------|
| `5001` | 503 | خدمة الذكاء الاصطناعي غير متاحة | AI service unavailable | Yes — بعد 5s | ⚠️ |
| `5002` | 500 | خطأ في الاتصال بقاعدة البيانات | Database connection failed | Yes — بعد 2s | 🚨 |
| `5003` | 504 | انتهت مهلة الاتصال بـ Salla | Salla API timeout | Yes — بعد 3s | ⚠️ |
| `5004` | 500 | فشل التشفير | Encryption operation failed | No | 🚨 Critical |
| `5005` | 500 | فشل التسجيل في audit log | Audit log write failed | No | ⚠️ |
| `5006` | 503 | خدمة البريد الإلكتروني غير متاحة | Email service unavailable | Yes — بعد 10s | ⚠️ |

---

## 6. تنسيق الاستجابة الموحد

### استجابة ناجحة

```json
{
  "success": true,
  "data": { }
}
```

### استجابة خطأ

```json
{
  "success": false,
  "error": {
    "code": 4291,
    "message": "تجاوزت الحد المسموح من الرسائل",
    "message_en": "Visitor rate limit exceeded",
    "retry_after": 60
  }
}
```

### استجابة خطأ validation (حقول متعددة)

```json
{
  "success": false,
  "error": {
    "code": 4223,
    "message": "الحقول الإلزامية مفقودة",
    "message_en": "Required fields missing",
    "fields": {
      "visitor_id": "مطلوب",
      "message": "مطلوب"
    }
  }
}
```

> **`retry_after`** يظهر فقط عند أخطاء 429. القيمة بالثواني.
> **`fields`** يظهر فقط عند أخطاء validation.

---

## 7. ربط الأخطاء بـ `errors.ts`

الكلاسات الموجودة في `apps/api/src/lib/errors.ts` تُعيَّن للـ codes هكذا:

| Class | Code افتراضي | HTTP |
|-------|-------------|------|
| `AuthenticationError` | 4002 | 401 |
| `AuthorizationError` | 4004 | 403 |
| `ValidationError` | 4223 | 400 |
| `NotFoundError` | 4403 | 404 |
| `RateLimitError` | 4291 | 429 |
| `DatabaseError` | 5002 | 500 |
| `AppError` (base) | 5000 | 500 |

عند رمي خطأ مع code محدد:

```typescript
throw new RateLimitError("Monthly quota exceeded", { code: 4293 });
throw new AuthenticationError("Token expired", { code: 4003 });
```

---

## 8. Widget Error Handling

```typescript
// apps/widget/src/lib/api.ts
function handleApiError(code: number): WidgetState {
  // Monthly quota — disable input permanently
  if (code === 4293) return "quota_exceeded";

  // Bot disabled — show contact info
  if (code === 4401) return "bot_disabled";

  // Subscription lapsed — treat as bot disabled
  if (code === 4402) return "bot_disabled";

  // System errors — show retry
  if (code >= 5000) return "error";

  // Rate limit — temporary, allow retry
  if (code === 4291 || code === 4292) return "rate_limited";

  // Default
  return "error";
}
```

---

## 9. Logging Guidelines

### Client Errors (4xxx) — `warn`

```typescript
logger.warn("client_error", {
  code: 4221,
  visitor_id: visitorId,
  store_id: storeId,
  path: req.path,
});
```

### System Errors (5xxx) — `error` + تنبيه

```typescript
logger.error("system_error", {
  code: 5001,
  store_id: storeId,
  error: err.message,
  stack: err.stack,
  alert: true,   // يُطلق تنبيهاً خارجياً
});
```

> **5002 و 5004** تُوقف العملية فوراً (إغلاق فوري حسب Rule 3 و Rule 4).

---

## 10. Retry Strategy

| الحالة | انتظار | محاولات قصوى |
|--------|--------|--------------|
| 4003 (token expired) | فوراً بعد refresh | 1 |
| 4291 (visitor limit) | 60s | 1 |
| 4292 (store limit) | 3600s | 1 |
| 5001 (AI unavailable) | 5s، 10s، 20s | 3 |
| 5002 (DB error) | 2s، 4s | 2 |
| 5003 (Salla timeout) | 3s، 6s | 2 |
| 5006 (email failed) | 10s | 2 |
| أخطاء 4xxx أخرى | — | لا retry |
