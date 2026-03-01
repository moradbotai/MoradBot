# قواعد التصميم — مراد بوت
**الإصدار:** 1.1 | **الفلسفة:** بسيط، عربي، احترافي | **التاريخ:** 2026-03-01

---

## الفلسفة العامة

مراد بوت أداة B2B للتجار السعوديين.
التصميم يجب أن يعكس: **الثقة، البساطة، الكفاءة**.

لا زخرفة. لا تعقيد. كل عنصر له وظيفة.

---

## 1. قواعد الألوان

### ✅ مسموح
- Primary `#2281F8` للأزرار الرئيسية وعنصر واحد فقط لكل صفحة
- Secondary `#9B51E0` للعناصر الثانوية والـ data visualization
- `#020617` للعناوين H1 والـ Nav pill النشط
- Semantic colors للحالات فقط (نجاح، تحذير، خطأ)

### ❌ ممنوع
- لا تضع اللونين الأساسيين (أزرق + بنفسجي) بجانب بعض في نفس العنصر
- لا تستخدم Primary لأكثر من CTA واحد في الصفحة
- لا تستخدم ألوان Semantic لأغراض تزيينية

---

## 2. قواعد الطباعة

### الخط الرسمي
**TheYearofHandicrafts** — الخط العربي الرسمي لجميع النصوص.

### التسلسل الهرمي
```
H1 (36px Bold 700, #020617)    → عنوان الصفحة
H2 (28px SemiBold 600, #1E293B) → أقسام رئيسية
H3 (22px SemiBold 600, #1E293B) → عناوين البطاقات
body (16px Regular 400, #1E293B) → المحتوى
caption (12px Regular 400, #475569) → نصوص مساعدة
```

### قواعد ثابتة
- **لا تستخدم أكثر من 3 أحجام خطوط في صفحة واحدة**
- **Labels دائماً uppercase + letter-spacing: 0.5px عند استخدام overline**
- النصوص العربية دائماً `text-align: right`
- **الأرقام (أرقام إحصاءات) يجب أن تكون إنجليزية دائماً** مع `direction: ltr`

---

## 3. قواعد البطاقات (Cards)

```
background: #FFFFFF
border: 1px solid #E2E8F0
border-radius: 12px
padding: 24px
shadow: 0 4px 12px rgba(0,0,0,0.06)
```

- كل بطاقة لها **هدف واحد** — لا تحشو معلومات غير ذات صلة
- Header + Content + (Actions اختياري)
- لا تضع أكثر من 3 metrics في بطاقة واحدة

---

## 4. قواعد الأزرار

### التسلسل
| النوع | اللون | متى تستخدمه |
|-------|-------|------------|
| Primary | `#2281F8` | الإجراء الأساسي الوحيد في الصفحة |
| Secondary | `border: 1px #2281F8, text: #2281F8` | إجراءات ثانوية |
| Ghost | `text: #1E293B, hover: #E2E8F0 bg` | إجراءات خفيفة |
| Danger | `#EF4444` | حذف، إلغاء، إجراءات خطيرة |
| Disabled | `#CBD5E1 bg, #94A3B8 text` | — |

### قاعدة
- **لا يوجد أكثر من زر Primary واحد في أي view**
- الأزرار المتجاورة: الأيمن = Primary، الأيسر = Secondary (RTL)

---

## 5. قواعد النماذج (Forms)

```
Input height: 40px
Border: 1px solid #CBD5E1
Border-radius: 8px
Padding: 0 16px
Focus border: 2px solid #2281F8
Error border: 1px solid #EF4444
```

- Label دائماً **فوق** الحقل (لا inline)
- Helper text أسفل الحقل، `#475569`، 12px
- Error message أسفل الحقل، `#EF4444`، 12px
- لا تستخدم placeholder بديلاً عن Label

---

## 6. قواعد Navigation

### Sidebar (Desktop)
```
width: 280px
background: #FFFFFF
border-right: 1px solid #E2E8F0
```

### Nav Item — حالة غير نشط
```
color: #475569
icon: #475569
hover background: #E2E8F0
border-radius: 8px
padding: 8px 12px
```

### Nav Item — حالة نشط
```
background: #020617
color: #FFFFFF
icon: #FFFFFF
border-radius: 8px (pill)
```

---

## 7. قواعد التصميم للـ Chat Widget

- **لا يتجاوز 380px عرضاً**
- **الرسائل في فقاعات:** المستخدم يمين `#2281F8 bg, #FFFFFF text`، البوت يسار `#F1F5F9 bg, #1E293B text`
- **Typing indicator:** 3 نقاط متحركة، لون `#475569`
- **AI Disclosure:** أول رسالة دائماً تحتوي على badge واضح "مساعد ذكاء اصطناعي"
- **لا يظهر على `/checkout/*`**
- **زر الإغلاق:** دائماً في الزاوية العلوية اليسرى (RTL)

---

## 8. قواعد Accessibility

- **تباين النص:** نسبة 4.5:1 كحد أدنى (WCAG AA)
- **Focus ring:** `2px solid #2281F8, offset: 2px` — يجب أن يكون مرئياً دائماً
- **لا تعتمد على اللون وحده** لنقل المعلومات (أضف icon أو نص)
- **Alt text** لجميع الصور الوظيفية
- **Keyboard navigation** لجميع العناصر التفاعلية

---

## 9. قواعد الـ Dashboard

### Layout
```
Sidebar (280px) | Main Content (fluid)
Main content padding: 32px
Content max-width: 1440px
Page background: #F1F5F9
```

### Stats Cards
- الرقم: 32px Bold 700، `#020617`، **إنجليزي دائماً**، `direction: ltr`
- Label: 12px Regular، `#475569`، uppercase
- Change indicator: 13px، أخضر للإيجابي، أحمر للسلبي

### Usage Bar (شريط الاستخدام)
```
0-79%:  background: #10B981 (أخضر)
80-99%: background: #F59E0B (برتقالي تحذير)
100%:   background: #EF4444 (أحمر)
```

---

## 10. قواعد الـ Empty States

عند عدم وجود بيانات:
- أيقونة كبيرة (48px)، لون `#94A3B8`
- عنوان: H3، `#1E293B`
- وصف: body-sm، `#475569`
- زر CTA (إن وجد): Primary

لا تترك الصفحة فارغة بدون توجيه.

---

## 11. قواعد الـ Loading States

- **Skeleton screens** بدلاً من spinners للمحتوى الكبير
- **Spinner** (دوائري) للأزرار فقط
- لون Skeleton: `#E2E8F0`، animated shimmer
- لا تُجمِّد الـ UI أثناء التحميل

---

## 12. قواعد RTL خاصة

- Padding/Margin: `padding-inline-start` بدلاً من `padding-right`
- Icons اتجاهية (أسهم، chevrons) تُعكس تلقائياً
- Form fields: الكتابة من اليمين
- Breadcrumbs: الصفحة الرئيسية يمين ← الصفحة الحالية يسار
- الجداول: العمود الأول يمين
- **الأرقام في الجداول والإحصاءات:** إنجليزية دائماً مع `direction: ltr`
