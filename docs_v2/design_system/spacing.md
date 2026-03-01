# نظام المسافات والأحجام — مراد بوت
**الإصدار:** 1.0 | **Base unit:** 4px

---

## 1. Spacing Scale

الوحدة الأساسية: **4px**
جميع المسافات مضاعفات من 4px.

| Token | px | rem | الاستخدام |
|-------|-----|-----|-----------|
| `space-0` | 0px | 0 | إلغاء المسافة |
| `space-1` | 4px | 0.25rem | مسافات داخلية صغيرة جداً (icon gap) |
| `space-2` | 8px | 0.5rem | padding داخل badges، gap بين icon والنص |
| `space-3` | 12px | 0.75rem | padding خفيف في chips، spacing بين عناصر Nav |
| `space-4` | 16px | 1rem | **الوحدة الأساسية** — padding البطاقات الصغيرة |
| `space-5` | 20px | 1.25rem | padding الحقول، gap بين عناصر Form |
| `space-6` | 24px | 1.5rem | padding البطاقات العادية، gap بين Cards |
| `space-8` | 32px | 2rem | padding البطاقات الكبيرة، gap بين Sections |
| `space-10` | 40px | 2.5rem | margin بين Sections في الصفحة |
| `space-12` | 48px | 3rem | padding الصفحات على Mobile |
| `space-16` | 64px | 4rem | padding الصفحات على Desktop |
| `space-20` | 80px | 5rem | spacing بين Sections الكبيرة |
| `space-24` | 96px | 6rem | Hero sections، Top padding |

---

## 2. Border Radius

| Token | px | الاستخدام |
|-------|-----|-----------|
| `radius-none` | 0px | لا يُستخدم عادةً |
| `radius-sm` | 4px | Badges، Tags صغيرة |
| `radius-md` | 8px | الأزرار الصغيرة، Inputs، Chips |
| `radius-lg` | 12px | **البطاقات (Cards)** — الاستخدام الرئيسي |
| `radius-xl` | 16px | Modals، Dropdowns كبيرة |
| `radius-2xl` | 24px | Sheets، Bottom sheets |
| `radius-full` | 9999px | الأزرار المستديرة، Avatars، الـ active Nav pill |

---

## 3. Shadows (Light Mode)

| Token | القيمة | الاستخدام |
|-------|--------|-----------|
| `shadow-none` | none | لا ظل |
| `shadow-xs` | `0 1px 2px rgba(0,0,0,0.04)` | عناصر داخل البطاقة |
| `shadow-sm` | `0 2px 4px rgba(0,0,0,0.05)` | Inputs، Dropdowns |
| `shadow-md` | `0 4px 12px rgba(0,0,0,0.06)` | **البطاقات الرئيسية** |
| `shadow-lg` | `0 8px 24px rgba(0,0,0,0.08)` | Modals، Floating panels |
| `shadow-xl` | `0 16px 40px rgba(0,0,0,0.10)` | Dialogs، Full-page overlays |

---

## 4. Borders

| Token | القيمة | الاستخدام |
|-------|--------|-----------|
| `border-none` | none | — |
| `border-thin` | `1px solid #E2DDD6` | حدود البطاقات الافتراضية |
| `border-medium` | `1px solid #C8C3BC` | حدود الـ Inputs |
| `border-focus` | `2px solid #2281F8` | Focus state للـ Inputs |
| `border-error` | `1px solid #EF4444` | خطأ في الحقل |

---

## 5. أحجام العناصر التفاعلية

### الأزرار
| الحجم | Height | Padding X | Font Size |
|-------|--------|-----------|-----------|
| sm | 32px | 12px | 13px |
| **md (default)** | **40px** | **16px** | **14px** |
| lg | 48px | 24px | 16px |
| xl | 56px | 32px | 18px |

### حقول الإدخال (Inputs)
| الحجم | Height | Padding X | Font Size |
|-------|--------|-----------|-----------|
| sm | 32px | 12px | 13px |
| **md (default)** | **40px** | **16px** | **14px** |
| lg | 48px | 16px | 16px |

### الأيقونات
| السياق | الحجم |
|--------|-------|
| داخل الأزرار الصغيرة | 16px |
| داخل الأزرار العادية | 18px |
| داخل Nav | 18px |
| Standalone icons | 20–24px |
| Hero/Feature icons | 32–48px |

---

## 6. Layout

### Container Widths
| السياق | Max Width |
|--------|-----------|
| Sidebar | 280px |
| Main content | fluid (يملأ المساحة) |
| Modal sm | 480px |
| Modal md | 640px |
| Modal lg | 800px |
| Widget | 380px width |

### Grid
- **Dashboard:** 12-column grid, gap: 24px
- **Mobile:** 4-column grid, gap: 16px
- **Widget:** Single column

### Breakpoints
| الاسم | Min Width | الاستخدام |
|-------|-----------|-----------|
| xs | 0px | Mobile small |
| sm | 480px | Mobile large |
| md | 768px | Tablet |
| lg | 1024px | Desktop small |
| xl | 1280px | Desktop |
| 2xl | 1536px | Wide screen |

---

## 7. Animation & Transitions

| Token | Duration | Easing | الاستخدام |
|-------|----------|--------|-----------|
| `duration-fast` | 100ms | ease-out | Hover states |
| `duration-normal` | 200ms | ease-in-out | Transitions العادية |
| `duration-slow` | 300ms | ease-in-out | Modals، Drawers |
| `duration-slower` | 500ms | ease-in-out | Page transitions |
