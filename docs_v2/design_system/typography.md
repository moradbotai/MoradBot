# نظام الخطوط — مراد بوت
**الإصدار:** 1.2 | **الاتجاه:** RTL (عربي أولاً) | **التاريخ:** 2026-03-01

---

## الخط الرسمي المعتمد

**IBM Plex Sans Arabic** — خط عربي مفتوح المصدر من IBM، متوفر عبر Google Fonts.

### التحميل عبر Google Fonts

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@100;200;300;400;500;600;700&display=swap" rel="stylesheet">
```

أو عبر CSS:
```css
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@100;200;300;400;500;600;700&display=swap');
```

---

## خطوط النظام (Fallback)

```css
--font-primary: 'IBM Plex Sans Arabic', system-ui, sans-serif;
--font-mono:    'JetBrains Mono', 'Courier New', monospace;
```

> **ملاحظة:** IBM Plex Sans Arabic يُستخدم لجميع النصوص العربية (العناوين والجسم). JetBrains Mono للكود فقط.

---

## أوزان الخطوط المتاحة

| Weight | القيمة | الاستخدام |
|--------|--------|-----------|
| Thin | 100 | — |
| ExtraLight | 200 | — |
| Light | 300 | — |
| **Regular** | **400** | **النص العادي، الجسم** |
| **Medium** | **500** | **تمييز خفيف** |
| **SemiBold** | **600** | **العناوين الفرعية، Labels** |
| **Bold** | **700** | **العناوين الرئيسية** |

---

## مقاييس الخطوط

| Token | px | rem | الاستخدام |
|-------|-----|-----|-----------|
| `text-display` | 48px | 3rem | Hero text، عناوين ترحيبية |
| `text-h1` | 36px | 2.25rem | عنوان الصفحة الرئيسي |
| `text-h2` | 28px | 1.75rem | أقسام رئيسية |
| `text-h3` | 22px | 1.375rem | عناوين البطاقات |
| `text-h4` | 18px | 1.125rem | عناوين فرعية |
| `text-h5` | 16px | 1rem | عناوين صغيرة |
| `text-body-lg` | 18px | 1.125rem | نص كبير |
| `text-body` | 16px | 1rem | النص العادي |
| `text-body-sm` | 14px | 0.875rem | نص صغير |
| `text-caption` | 12px | 0.75rem | نصوص مساعدة |
| `text-overline` | 11px | 0.6875rem | Labels فوقية |
| `text-stat` | 32px | 2rem | أرقام الإحصائيات |

---

## التسلسل الهرمي للنصوص

```
Display (48px, Bold 700,    #020617)    → Hero sections
H1      (36px, Bold 700,    #020617)    → عنوان الصفحة
H2      (28px, SemiBold 600, #1E293B)   → أقسام رئيسية
H3      (22px, SemiBold 600, #1E293B)   → عناوين البطاقات
H4      (18px, Medium 500,   #1E293B)   → عناوين فرعية
body    (16px, Regular 400,  #1E293B)   → المحتوى
caption (12px, Regular 400,  #475569)   → نصوص مساعدة
```

---

## قواعد ثابتة

1. **لا تستخدم أكثر من 3 أحجام** خطوط في صفحة واحدة
2. **overline labels:** uppercase + letter-spacing: 0.5px دائماً
3. **النصوص العربية:** `text-align: right` دائماً
4. **الأرقام الإحصائية:** إنجليزية دائماً مع `direction: ltr`
5. **لا تستخدم Italic** في النصوص العربية
