# إرشادات الشعار — مراد بوت
**الإصدار:** 1.0

---

## ⚠️ مطلوب منك: رفع ملفات الهوية

ارفع جميع ملفات الشعار إلى:
```
/Users/mohammedaljohani/Documents/Proj/moradbot/assets/brand/logo/
```

الهيكل المطلوب:
```
assets/brand/logo/
├── svg/
│   ├── logo-full.svg          # الشعار كامل (رمز + نص) — بخلفية
│   ├── logo-full-clear.svg    # الشعار كامل — بدون خلفية
│   ├── logo-mark.svg          # الرمز فقط — بخلفية
│   ├── logo-mark-clear.svg    # الرمز فقط — بدون خلفية
│   ├── logo-white.svg         # نسخة بيضاء (للخلفيات الداكنة)
│   └── logo-dark.svg          # نسخة داكنة (للخلفيات الفاتحة)
├── png/
│   ├── logo-full@1x.png       # 150px height
│   ├── logo-full@2x.png       # 300px height
│   ├── logo-full@3x.png       # 450px height
│   ├── logo-mark@1x.png       # 64px × 64px
│   ├── logo-mark@2x.png       # 128px × 128px
│   └── logo-mark@3x.png       # 256px × 256px
├── jpg/
│   ├── logo-full-on-white.jpg
│   ├── logo-full-on-warm.jpg  # على خلفية #F7F3EC
│   └── logo-mark-on-white.jpg
└── favicon/
    ├── favicon.ico
    ├── favicon-16x16.png
    ├── favicon-32x32.png
    ├── apple-touch-icon.png   # 180×180
    └── android-chrome-192x192.png
```

---

## قواعد استخدام الشعار

### 1. المساحة المحمية (Clear Space)
يجب أن تكون المساحة حول الشعار مساوية لـ **نصف ارتفاع حرف الشعار** على الأقل من جميع الجهات.

### 2. الأحجام الدنيا
| النسخة | الحد الأدنى |
|--------|------------|
| الشعار الكامل (رمز + نص) | 120px عرضاً |
| الرمز وحده | 24px × 24px |
| favicon | 16px × 16px |

### 3. الألوان المسموح بها

| الخلفية | نسخة الشعار المستخدمة |
|---------|----------------------|
| أبيض `#FFFFFF` | النسخة الداكنة (الأصلية) |
| أبيض دافئ `#F7F3EC` | النسخة الداكنة (الأصلية) |
| أزرق `#2281F8` | النسخة البيضاء `logo-white.svg` |
| داكن `#363636` أو أقتم | النسخة البيضاء `logo-white.svg` |
| بنفسجي `#9B51E0` | النسخة البيضاء `logo-white.svg` |

### 4. ممنوع

- ❌ تدوير الشعار
- ❌ تمديده أو تقليصه بشكل غير متناسب
- ❌ تغيير ألوانه خارج الأكواد المعتمدة
- ❌ إضافة ظلال أو تأثيرات
- ❌ وضعه على خلفية تقلل التباين

### 5. متى تستخدم كل نسخة

| النسخة | متى تستخدمها |
|--------|-------------|
| `logo-full` | Header الـ Dashboard، صفحات التسويق، الـ Widget |
| `logo-mark` | Favicon، App icon، مساحات صغيرة |
| `logo-white` | Footers داكنة، banners ملوّنة |

---

## الشعار في الكود

```html
<!-- SVG inline للأداء الأفضل -->
<img src="/assets/brand/logo/svg/logo-mark-clear.svg"
     alt="مراد بوت"
     width="32" height="32" />

<!-- للـ Widget (حجم صغير) -->
<img src="/assets/brand/logo/png/logo-mark@2x.png"
     alt="مراد بوت"
     width="24" height="24" />
```

---

## سيُحدَّث هذا الملف بعد رفع الهوية الرسمية
عند رفع الملفات، سيُضاف:
- وصف تفصيلي للرمز والنص
- ألوان الشعار الدقيقة
- إرشادات الـ animation (إن وُجد)
