# نظام الألوان — مراد بوت
**الإصدار:** 1.1 | **الوضع:** Light Mode | **التاريخ:** 2026-03-01

---

## الألوان المعتمدة رسمياً

| اللون | HEX | الاستخدام |
|-------|-----|-----------|
| **Primary** | `#2281F8` | الأزرار الرئيسية، الـ CTA الوحيد، الـ links |
| **Secondary** | `#9B51E0` | الـ data visualization، العناصر الثانوية |

> ⚠️ هذان اللونان ثابتان رسمياً — لا يُعدَّلان بأي حال.

---

## 1. Primary — الأزرق

| Token | HEX | الاستخدام |
|-------|-----|-----------|
| `primary-50` | `#EFF6FF` | خلفيات Highlight خفيفة |
| `primary-100` | `#DBEAFE` | Badge background |
| `primary-200` | `#BFDBFE` | Hover على elements محددة |
| `primary-300` | `#93C5FD` | – |
| `primary-400` | `#60A5FA` | Info color |
| **`primary-500`** | **`#2281F8`** | **BASE — الأزرق الرسمي** |
| `primary-600` | `#1A6FD4` | Hover على الأزرار الرئيسية |
| `primary-700` | `#155CB0` | Active state |
| `primary-800` | `#104A8C` | – |
| `primary-900` | `#0C3868` | – |

---

## 2. Secondary — البنفسجي

| Token | HEX | الاستخدام |
|-------|-----|-----------|
| `secondary-50` | `#F5F0FD` | خلفيات خفيفة |
| `secondary-100` | `#EDE0FB` | Badge background |
| `secondary-200` | `#D9C0F7` | – |
| `secondary-300` | `#C49BF1` | – |
| `secondary-400` | `#B071E8` | – |
| **`secondary-500`** | **`#9B51E0`** | **BASE — البنفسجي الرسمي** |
| `secondary-600` | `#7E3CB8` | Hover |
| `secondary-700` | `#632E91` | Active |
| `secondary-800` | `#4A226C` | – |
| `secondary-900` | `#321748` | – |

---

## 3. Neutral — Cool Slate

> تم اختيار هذا النظام المحايد لتناسقه مع اللون الأزرق الرئيسي. الدرجات تميل للبرود الخفيف (Cool-tinted) مما يخلق انسجاماً بصرياً مع Primary وSecondary.

| Token | HEX | الاستخدام |
|-------|-----|-----------|
| `neutral-50` | `#F8FAFC` | أفاتح درجة |
| `neutral-100` | `#F1F5F9` | **خلفية الصفحة (Page Background)** |
| `neutral-200` | `#E2E8F0` | Dividers، borders خفيفة |
| `neutral-300` | `#CBD5E1` | **حدود البطاقات والـ inputs** |
| `neutral-400` | `#94A3B8` | Placeholder text، icons غير نشطة |
| `neutral-500` | `#64748B` | – |
| `neutral-600` | `#475569` | **النص الثانوي** |
| `neutral-700` | `#334155` | – |
| `neutral-800` | `#1E293B` | **النص الرئيسي** |
| `neutral-900` | `#0F172A` | – |
| `neutral-black` | `#020617` | العناوين H1، الـ Active Nav pill |
| `neutral-white` | `#FFFFFF` | سطح البطاقات، خلفية الـ sidebar |

---

## 4. Semantic — الألوان الدلالية

### Success (النجاح)
| Token | HEX |
|-------|-----|
| `success-light` | `#D1FAE5` |
| `success` | `#10B981` |
| `success-dark` | `#059669` |

### Warning (التحذير)
| Token | HEX |
|-------|-----|
| `warning-light` | `#FEF3C7` |
| `warning` | `#F59E0B` |
| `warning-dark` | `#D97706` |

### Error (الخطأ)
| Token | HEX |
|-------|-----|
| `error-light` | `#FEE2E2` |
| `error` | `#EF4444` |
| `error-dark` | `#DC2626` |

### Info (المعلومات)
| Token | HEX |
|-------|-----|
| `info-light` | `#DBEAFE` |
| `info` | `#60A5FA` |
| `info-dark` | `#2281F8` |

---

## 5. Surface Aliases — الأسطح

| Token | HEX | الاستخدام |
|-------|-----|-----------|
| `surface-page` | `#F1F5F9` | خلفية الصفحة |
| `surface-card` | `#FFFFFF` | سطح البطاقات |
| `surface-sidebar` | `#FFFFFF` | الشريط الجانبي |
| `surface-input` | `#FFFFFF` | حقول الإدخال |
| `surface-hover` | `#E2E8F0` | Hover على عناصر Nav |
| `surface-active` | `#020617` | Active Nav pill |

---

## 6. Text Aliases — النصوص

| Token | HEX | الاستخدام |
|-------|-----|-----------|
| `text-primary` | `#1E293B` | النص الرئيسي |
| `text-secondary` | `#475569` | النص الثانوي |
| `text-disabled` | `#94A3B8` | عناصر معطلة |
| `text-heading` | `#020617` | العناوين |
| `text-on-dark` | `#FFFFFF` | نص على خلفيات داكنة |
| `text-on-primary` | `#FFFFFF` | نص على الأزرار الزرقاء |
| `text-link` | `#2281F8` | الروابط |
| `text-link-hover` | `#1A6FD4` | Hover على الروابط |

---

## 7. Border Aliases — الحدود

| Token | HEX | الاستخدام |
|-------|-----|-----------|
| `border-default` | `#E2E8F0` | حدود البطاقات |
| `border-input` | `#CBD5E1` | حدود الحقول |
| `border-focus` | `#2281F8` | Focus state |
| `border-error` | `#EF4444` | حقل به خطأ |
| `border-active` | `#020617` | عنصر نشط |

---

## 8. Data Visualization

| Token | HEX |
|-------|-----|
| `data-1` | `#2281F8` |
| `data-2` | `#9B51E0` |
| `data-3` | `#10B981` |
| `data-4` | `#F59E0B` |
| `data-5` | `#EF4444` |
| `data-neutral` | `#94A3B8` |

---

## قواعد الاستخدام

- **لا تضع Primary وSecondary** في نفس العنصر أو بجانب بعض مباشرةً
- **Primary** للـ CTA الرئيسي الواحد فقط في كل صفحة
- **Semantic colors** للحالات فقط — ليس للزينة
- **Neutral-100** هو لون خلفية الصفحة الوحيد المعتمد في Light Mode
