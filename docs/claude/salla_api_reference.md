# Salla Merchant API — مرجع تقني شامل

> **المصدر:** [docs.salla.dev](https://docs.salla.dev) — استُخلص بتاريخ 18 فبراير 2026
> **الغرض:** مرجع سريع لبناء MoradBot (تكامل سلة)

---

## الفهرس

1. [نظرة عامة — Overview](#1-نظرة-عامة)
2. [البدء — Getting Started](#2-البدء)
3. [المصادقة — OAuth 2.0](#3-مصادقة-oauth-20)
4. [واجهة المنتجات — GET /products](#4-واجهة-المنتجات)
5. [أكواد الأخطاء — Error Codes](#5-أكواد-الأخطاء)
6. [تحديد المعدل — Rate Limiting](#6-تحديد-المعدل)

---

## 1. نظرة عامة

### ما هي سلة؟
منصة تجارة إلكترونية تُمكّن المطورين من بناء تطبيقات وقوالب، وتتيح للتجار إدارة متاجرهم الإلكترونية.

### واجهات برمجية متاحة
| الواجهة | الوصف |
|---------|-------|
| **Merchant API** | RESTful endpoints للوصول الآمن إلى بيانات التاجر |
| App API | وظائف بوابة الشركاء |
| Shipping & Fulfillment API | إدارة الشحنات والطلبات |
| Twilight SDK | JavaScript SDK لواجهة المتجر |

### Base URL
```
https://api.salla.dev/admin/v2
```

### نسخة API الحالية
`v2` — جميع endpoints تبدأ بـ `/admin/v2/`

---

## 2. البدء

### متطلبات الدمج
1. حساب على [Salla Partners Portal](https://salla.partners/)
2. إنشاء App داخل البوابة للحصول على `client_id` و `client_secret`
3. تحديد Redirect URI (Callback URL)
4. اختيار الـ Scopes المطلوبة
5. تنفيذ OAuth 2.0 Flow لاستقبال `access_token`

### روابط مفيدة
| المورد | الرابط |
|--------|--------|
| Partners Portal | https://salla.partners/ |
| API Docs | https://docs.salla.dev/ |
| Demo Stores | https://salla.dev/blog/how-to-test-your-app-using-salla-demo-stores/ |
| Postman Collection | متاح من صفحة Authorization |
| دعم المطورين | support@salla.dev |
| مجتمع Telegram | https://t.me/salladev |

---

## 3. مصادقة OAuth 2.0

### نظرة عامة
Salla تستخدم OAuth 2.0 للسماح للتطبيقات بالوصول إلى بيانات التجار بشكل آمن دون الحاجة لكلمة المرور.

مدة Access Token: **2 أسابيع (14 يوم)**
مدة Refresh Token: **شهر واحد** (يُستخدم مرة واحدة فقط)

---

### Endpoints الأساسية

| Endpoint | URL | الوصف |
|----------|-----|-------|
| **Authorization** | `https://accounts.salla.sa/oauth2/auth` | بدء عملية الحصول على إذن التاجر |
| **Token** | `https://accounts.salla.sa/oauth2/token` | استبدال authorization code بـ access token |
| **Refresh Token** | `https://accounts.salla.sa/oauth2/token` | الحصول على access token جديد |
| **User Info** | `https://accounts.salla.sa/oauth2/user/info` | جلب بيانات التاجر بعد المصادقة |

---

### OAuth 2.0 Flow (Custom Mode)

#### الخطوة 1 — Authorization Request
وجّه التاجر إلى Authorization URL:

```
https://accounts.salla.sa/oauth2/auth
  ?client_id=your_client_id
  &response_type=code
  &redirect_uri=https://your-app.com/callback
  &scope=offline_access
  &state=random_csrf_value
```

**Query Parameters:**

| Parameter | الوصف | مثال |
|-----------|-------|-------|
| `client_id` | معرّف التطبيق من بوابة الشركاء | `1311508470xxx` |
| `response_type` | نوع الاستجابة المطلوبة | `code` |
| `redirect_uri` | URL إعادة التوجيه بعد الموافقة | `https://your-app.com/callback` |
| `scope` | الصلاحيات المطلوبة | `offline_access` أو `products.read` |
| `state` | قيمة عشوائية لحماية CSRF | `random_value` |

**الاستجابة — Callback URL:**
```
https://your-app.com/callback?code={code-value}&scope={app-scopes}+offline_access&state={state-value}
```

---

#### الخطوة 2 — Access Token Exchange
استبدل `code` بـ `access_token`:

```http
POST https://accounts.salla.sa/oauth2/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code={authorization_code}
&client_id={client_id}
&client_secret={client_secret}
&redirect_uri=https://your-app.com/callback
```

**استجابة ناجحة:**
```json
{
  "access_token": "eyJ...",
  "token_type": "Bearer",
  "expires_in": 1209600,
  "refresh_token": "def50200...",
  "scope": "offline_access products.read"
}
```

> ملاحظة: `expires_in` تُعاد كـ **Unix timestamp بالثواني** (مدة الصلاحية 14 يوم)

---

#### الخطوة 3 — استخدام Access Token
```http
GET https://api.salla.dev/admin/v2/products
Authorization: Bearer {access_token}
```

---

### Refresh Token

> **تحذير:** كل Refresh Token يُستخدم مرة واحدة فقط. استخدامه مرتين يُلغي كل الـ tokens ويجبر التاجر على إعادة تثبيت التطبيق.

```http
POST https://accounts.salla.sa/oauth2/token
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token
&refresh_token={refresh_token}
&client_id={client_id}
&client_secret={client_secret}
```

**ملاحظات مهمة للـ Refresh Token:**
- صالح لمدة **شهر واحد**
- يُصدر refresh token جديد مع كل refresh request
- يجب دائماً استخدام **آخر** refresh token
- يُنصح بتطبيق **mutex/locking** لتجنب استدعاءات متزامنة

---

### Scopes المتاحة (المهمة لـ MoradBot)

| Scope | الوصف |
|-------|-------|
| `products.read` | قراءة المنتجات (المطلوب لـ MoradBot) |
| `offline_access` | الحصول على refresh token |

> **MoradBot يحتاج فقط:** `products.read` + `offline_access`

---

### Easy Mode vs Custom Mode

| | Easy Mode | Custom Mode |
|--|-----------|-------------|
| **الاستخدام** | Apps في App Store | Testing & Development |
| **الآلية** | Salla يُعالج كل شيء تلقائياً | تنفيذ يدوي كامل |
| **الاستقبال** | Webhook (`app.store.authorize`) | Callback URL |
| **للنشر الرسمي** | ✅ مطلوب | ❌ للاختبار فقط |

في Easy Mode، تستقبل الـ Access Token عبر webhook event `app.store.authorize`.

---

## 4. واجهة المنتجات

### List Products

```http
GET https://api.salla.dev/admin/v2/products
Authorization: Bearer {access_token}
```

**Required Scope:** `products.read`

---

### Query Parameters

| Parameter | النوع | الوصف | القيمة الافتراضية |
|-----------|------|-------|-----------------|
| `page` | integer | رقم الصفحة | `1` |
| `per_page` | integer | عدد النتائج في الصفحة (الحد الأقصى: 60) | `15` |
| `keyword` | string | البحث في اسم المنتج | — |
| `status` | string | تصفية حسب الحالة (`sale`, `out`, etc.) | — |
| `category` | integer | تصفية حسب ID الفئة | — |
| `format` | string | `light` للاستجابة المختصرة | — |

**مثال:**
```bash
curl --request GET \
  'https://api.salla.dev/admin/v2/products?page=1&per_page=20&keyword=قميص&status=sale' \
  --header 'Authorization: Bearer <token>'
```

---

### Response Schema (Full Format)

```json
{
  "status": 200,
  "success": true,
  "data": [
    {
      "id": 1672932878,
      "sku": "23-4324432",
      "mpn": "58636897",
      "gtin": "45344432343",
      "type": "product",
      "name": "T-Shirt",
      "short_link_code": "qQvmmRb",
      "promotion": {
        "title": "ramadan promtotion",
        "sub_title": "special ramadan offers"
      },
      "thumbnail": "https://salla-dev.s3.eu-central-1.amazonaws.com/.../image.jpg",
      "main_image": "https://salla-dev.s3.eu-central-1.amazonaws.com/.../image.jpg",
      "urls": {
        "customer": "https://salla.sa/store-name/product-name/p{id}",
        "admin": "https://s.salla.sa/products/{id}"
      },
      "price": { "amount": 100, "currency": "SAR" },
      "taxed_price": { "amount": 100, "currency": "SAR" },
      "pre_tax_price": { "amount": 100, "currency": "SAR" },
      "tax": { "amount": 0, "currency": "SAR" },
      "regular_price": { "amount": 100, "currency": "SAR" },
      "sale_price": { "amount": 0, "currency": "SAR" },
      "cost_price": "35",
      "description": "وصف المنتج",
      "quantity": "50",
      "status": "sale",
      "is_available": true,
      "views": 0,
      "weight": 1,
      "weight_type": "kg",
      "with_tax": true,
      "url": "https://salla.sa/store-name/product-name/p{id}",
      "require_shipping": true,
      "sale_end": {},
      "sold_quantity": 0,
      "rating": { "total": 0, "count": 0, "rate": 0 },
      "max_items_per_user": 0,
      "maximum_quantity_per_order": 10,
      "hide_quantity": false,
      "unlimited_quantity": true,
      "notify_quantity": "10",
      "show_in_app": true,
      "managed_by_branches": false,
      "calories": "350",
      "channels": ["web", "app"],
      "metadata": {
        "title": "...",
        "description": "...",
        "url": "https://link"
      },
      "allow_attachments": false,
      "is_pinned": false,
      "updated_at": "2024-03-06 13:50:32",
      "options": [
        {
          "id": 782399854,
          "name": "Color",
          "type": "radio",
          "required": false,
          "values": [
            {
              "id": 422377234,
              "name": "Blue",
              "price": { "amount": 0, "currency": "SAR" }
            }
          ]
        }
      ],
      "skus": [
        {
          "id": 1936825372,
          "price": { "amount": 100, "currency": "SAR" },
          "stock_quantity": 12,
          "sku": "23-4324432",
          "weight": 12,
          "weight_type": "kg"
        }
      ],
      "categories": [
        {
          "id": 1032561074,
          "name": "الفساتين",
          "status": "active"
        }
      ],
      "tags": [],
      "images": [
        {
          "id": 1699133464,
          "url": "https://...",
          "main": false,
          "type": "image",
          "sort": 5
        }
      ]
    }
  ],
  "pagination": {
    "count": 20,
    "total": 90,
    "perPage": 20,
    "currentPage": 3,
    "totalPages": 5,
    "links": {
      "previous": "http://api.salla.dev/admin/v2/products?page=2",
      "next": "http://api.salla.dev/admin/v2/products?page=4"
    }
  }
}
```

---

### الحقول المهمة لـ MoradBot

| الحقل | النوع | الاستخدام في MoradBot |
|-------|------|----------------------|
| `id` | integer | معرّف المنتج |
| `name` | string | اسم المنتج (للعرض والبحث) |
| `description` | string | وصف المنتج لإجابة الأسئلة |
| `price.amount` | number | السعر الأساسي |
| `price.currency` | string | العملة (SAR) |
| `status` | string | `sale`=معروض، `out`=نفد المخزون |
| `is_available` | boolean | هل المنتج متاح للشراء |
| `quantity` | string/int | الكمية المتوفرة |
| `thumbnail` | string URL | صورة المنتج |
| `urls.customer` | string URL | رابط المنتج للعميل |
| `categories[].name` | string | تصنيف المنتج |

---

### Pagination Format

جميع responses المُصفّحة تتضمن:

```json
"pagination": {
  "count": 20,        // عدد النتائج في الصفحة الحالية
  "total": 90,        // إجمالي النتائج
  "perPage": 20,      // عدد النتائج لكل صفحة
  "currentPage": 3,   // الصفحة الحالية
  "totalPages": 5,    // إجمالي الصفحات
  "links": {
    "previous": "https://api.salla.dev/admin/v2/products?page=2",
    "next": "https://api.salla.dev/admin/v2/products?page=4"
  }
}
```

**الحد الأقصى لـ `per_page`: 60**

---

## 5. أكواد الأخطاء

### بنية الاستجابة الناجحة (2xx)

```json
{
  "status": 200,
  "success": true,
  "data": { ... }
}
```

### بنية الاستجابة الخاطئة (4xx / 5xx)

**خطأ واحد:**
```json
{
  "status": 422,
  "success": false,
  "error": {
    "code": "error",
    "message": "alert.invalid_fields",
    "fields": {
      "field_name": ["رسالة الخطأ"]
    }
  }
}
```

**أخطاء متعددة:**
```json
{
  "status": 422,
  "success": false,
  "error": {
    "code": "validation_failed",
    "message": "...",
    "fields": {
      "first_name": ["الاسم الاول للعميل مطلوب"],
      "email": ["البريد الإلكتروني مطلوب"]
    }
  }
}
```

---

### جدول أكواد HTTP

| Code | Slug | الاسم | المعنى |
|------|------|-------|--------|
| **200** | — | Success | الطلب نجح |
| **201** | — | Created | تم إنشاء المورد بنجاح |
| **202** | — | Accepted | تم حذف المورد بنجاح |
| **204** | — | No Content | نجح الطلب ولا يوجد محتوى للإرجاع |
| **400** | `bad_request` | Bad Request | معاملات أو حقول غير صحيحة |
| **401** | `unauthorized` | Unauthorized | خطأ في المصادقة |
| **403** | `forbidden` | Forbidden | الوصول مرفوض (تجاوز الأخطاء أو صلاحيات ناقصة) |
| **404** | `not_found` | Not Found | المورد غير موجود |
| **405** | `method_not_allowed` | Method Not Allowed | HTTP Method غير مسموح |
| **406** | `not_acceptable` | Not Acceptable | الصيغة المطلوبة غير مقبولة |
| **410** | `gone` | Gone | المورد لم يعد موجوداً |
| **422** | `validation_failed` | Unprocessable Entity | بيانات ناقصة أو غير صحيحة |
| **429** | `too_many_requests` | Too Many Requests | تجاوز حد المعدل (Rate Limit) |
| **500** | `server_error` | Internal Server Error | خطأ في الخادم |
| **503** | `service_unavailable` | Service Unavailable | الخادم غير متاح مؤقتاً |

---

### حالات 401 الشائعة

#### 1. مستخدم محذوف
```json
{
  "status": 401,
  "success": false,
  "error": {
    "code": "Unauthorized",
    "message": "The User is not exists."
  }
}
```

#### 2. مستخدم غير نشط
```json
{
  "status": 401,
  "success": false,
  "error": {
    "code": "Unauthorized",
    "message": "عفوا لا يمكنك تسجيل الدخول, حسابك غير مفعل"
  }
}
```

#### 3. إعادة استخدام Refresh Token
```json
{
  "error": "invalid_grant",
  "error_description": "The provided authorization grant ... or refresh token is invalid, expired, revoked..."
}
```
> **تحذير:** هذا يُلغي كل الـ tokens ويستلزم إعادة تثبيت التطبيق.

#### 4. Scope غير مسموح
```json
{
  "status": 401,
  "success": false,
  "error": {
    "code": "Unauthorized",
    "message": "The access token should have access to one of those scopes: products.read_write"
  }
}
```

#### 5. Access Token منتهي أو غير صحيح
```json
{
  "status": 401,
  "success": false,
  "error": {
    "code": "Unauthorized",
    "message": "The access token is invalid"
  }
}
```

---

## 6. تحديد المعدل

### خطط Rate Limit

Rate limits تعتمد على خطة اشتراك المتجر:

| الخطة | الحد الأقصى | مدة النافذة | معدل التسرب |
|-------|------------|-------------|------------|
| **Plus** | 120 طلب | 1 دقيقة | 1 طلب/ثانية |
| **Pro** | 360 طلب | 1 دقيقة | 1 طلب/ثانية |
| **Special** | 720 طلب | 1 دقيقة | 1 طلب/ثانية |

> الخوارزمية المستخدمة: **Leaky Bucket Algorithm**

---

### Rate Limit Headers

كل استجابة API تتضمن هذه الـ headers:

| Header | الوصف |
|--------|-------|
| `X-RateLimit-Limit` | الحد الأقصى لعدد الطلبات في الدقيقة |
| `X-RateLimit-Remaining` | عدد الطلبات المتبقية في النافذة الحالية |
| `X-RateLimit-Reset` | وقت إعادة تعيين النافذة (Unix timestamp) |
| `Retry-After` | الوقت المطلوب انتظاره (بالثواني) عند تجاوز الحد |

---

### حدود خاصة

| Endpoint | الحد |
|----------|------|
| Customers Endpoint | 500 طلب لكل 10 دقائق |

---

### التعامل مع 429 (Rate Limit Exceeded)

```json
{
  "status": 429,
  "success": false,
  "error": {
    "code": "too_many_requests",
    "message": "..."
  }
}
```

**الحل الموصى به:**
1. التحقق من `Retry-After` header
2. الانتظار المدة المحددة
3. تطبيق **Exponential Backoff** للمحاولات المتكررة
4. تخزين النتائج مؤقتاً (Caching) لتقليل الطلبات

---

## ملاحق

### قائمة Endpoints الكاملة للمنتجات

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `GET` | `/products` | قائمة المنتجات (المستخدم في MoradBot) |
| `POST` | `/products` | إنشاء منتج جديد |
| `GET` | `/products/{id}` | تفاصيل منتج |
| `PUT` | `/products/{id}` | تحديث منتج |
| `DELETE` | `/products/{id}` | حذف منتج |
| `POST` | `/products/{id}/status` | تغيير حالة المنتج |
| `GET` | `/products/sku/{sku}` | تفاصيل منتج بـ SKU |

> **MoradBot يستخدم فقط:** `GET /products` (قراءة فقط — القاعدة 2)

### روابط التوثيق المباشرة

| الموضوع | الرابط |
|---------|--------|
| Authorization | https://docs.salla.dev/421118m0 |
| List Products | https://docs.salla.dev/5394168e0 |
| Responses & Errors | https://docs.salla.dev/421123m0 |
| Pagination | https://docs.salla.dev/421124m0 |
| Rate Limiting | https://docs.salla.dev/421125m0 |
| Get Started | https://docs.salla.dev/421117m0 |
