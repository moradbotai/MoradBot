# نظام الخطوط — مراد بوت
**الإصدار:** 1.1 | **الاتجاه:** RTL (عربي أولاً) | **التاريخ:** 2026-03-01

---

## الخط الرسمي المعتمد

**TheYearofHandicrafts** — خط عربي رسمي للمشروع.

### المسار
```
assets/brand/fonts/Arabic_font/
├── WOFF2/
│   ├── TheYearofHandicrafts-Black.woff2
│   ├── TheYearofHandicrafts-Bold.woff2
│   ├── TheYearofHandicrafts-Medium.woff2
│   ├── TheYearofHandicrafts-Regular.woff2
│   └── TheYearofHandicrafts-SemiBold.woff2
└── (OTF — للتصميم فقط)
    ├── TheYearofHandicrafts-Black.otf
    ├── TheYearofHandicrafts-Bold.otf
    ├── TheYearofHandicrafts-Medium.otf
    ├── TheYearofHandicrafts-Regular.otf
    └── TheYearofHandicrafts-SemiBold.otf
```

### تعريف الـ @font-face في CSS
```css
@font-face {
  font-family: 'TheYearofHandicrafts';
  src: url('/fonts/TheYearofHandicrafts-Regular.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: 'TheYearofHandicrafts';
  src: url('/fonts/TheYearofHandicrafts-Medium.woff2') format('woff2');
  font-weight: 500;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: 'TheYearofHandicrafts';
  src: url('/fonts/TheYearofHandicrafts-SemiBold.woff2') format('woff2');
  font-weight: 600;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: 'TheYearofHandicrafts';
  src: url('/fonts/TheYearofHandicrafts-Bold.woff2') format('woff2');
  font-weight: 700;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: 'TheYearofHandicrafts';
  src: url('/fonts/TheYearofHandicrafts-Black.woff2') format('woff2');
  font-weight: 900;
  font-style: normal;
  font-display: swap;
}
```

---

## خطوط النظام (Fallback)

```css
--font-primary: 'TheYearofHandicrafts', 'Tajawal', system-ui, sans-serif;
--font-mono:    'JetBrains Mono', 'Courier New', monospace;
```

> **ملاحظة:** TheYearofHandicrafts يُستخدم لجميع النصوص العربية (العناوين والجسم). JetBrains Mono للكود فقط.

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

## أوزان الخطوط

| Token | Weight | الاستخدام |
|-------|--------|-----------|
| `weight-regular` | 400 | النص العادي، الجسم |
| `weight-medium` | 500 | تمييز خفيف |
| `weight-semibold` | 600 | العناوين الفرعية، الـ labels |
| `weight-bold` | 700 | العناوين الرئيسية |
| `weight-black` | 900 | Display text، أرقام Hero |

---

## ارتفاع السطر (Line Height)

| Token | القيمة | الاستخدام |
|-------|--------|-----------|
| `leading-tight` | 1.25 | العناوين الكبيرة |
| `leading-snug` | 1.35 | العناوين الفرعية |
| `leading-normal` | 1.5 | النص العادي |
| `leading-relaxed` | 1.65 | نص طويل، مقالات |
| `leading-loose` | 1.75 | نص صغير، captions |

> **ملاحظة للعربية:** استخدم `leading-relaxed` (1.65) كحد أدنى للنص العربي لضمان وضوح الحركات والتشكيل.

---

## التسلسل الهرمي للنصوص

```
Display (48px, Black 900, #020617)   → Hero sections
H1      (36px, Bold 700,  #020617)   → عنوان الصفحة
H2      (28px, SemiBold 600, #1E293B) → أقسام رئيسية
H3      (22px, SemiBold 600, #1E293B) → عناوين البطاقات
H4      (18px, Medium 500,  #1E293B) → عناوين فرعية
body    (16px, Regular 400, #1E293B) → المحتوى
caption (12px, Regular 400, #475569) → نصوص مساعدة
```

---

## قواعد ثابتة

1. **لا تستخدم أكثر من 3 أحجام** خطوط في صفحة واحدة
2. **overline labels:** uppercase + letter-spacing: 0.5px دائماً
3. **النصوص العربية:** `text-align: right` دائماً
4. **الأرقام الإحصائية:** `direction: ltr` للوضوح (أرقام إنجليزية فقط)
5. **لا تستخدم Italic** في النصوص العربية
