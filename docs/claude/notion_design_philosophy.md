# فلسفة Notion التصميمية — وثيقة مرجعية

> **المصدر:** مسح مباشر لـ 5 صفحات Notion بواسطة Firecrawl (branding format)
> الصفحات: homepage, product, pricing, about, blog
> **تاريخ التوثيق:** مارس 2026
> **الغرض:** مرجع تصميمي لصفحة هبوط مراد بوت

---

## 1. الفلسفة الجوهرية

### "الورقة البيضاء" (Blank Paper Philosophy)

تنطلق Notion من استعارة الورقة البيضاء كمبدأ تصميمي أساسي:

- الورقة البيضاء **مرنة**: تقبل أي محتوى بأي ترتيب
- الورقة البيضاء **لا تُقيّد**: لا تفرض هيكلاً مسبقاً على المستخدم
- الورقة البيضاء **بسيطة**: لا زخرفة، لا ضجيج بصري

**الجذور الفكرية:**
- Doug Engelbart: "تضخيم الذكاء البشري" (Augmenting Human Intellect, 1962)
- Alan Kay: الحاسوب كأداة لتوسيع الخيال البشري
- Ted Nelson: توسيع الأفكار بما يتخطى حدود الورق

### المبدأ الأساسي

> *"Design is not about how things look, but how things work."*

التصميم في Notion ليس جمالياً بالمعنى الزخرفي — بل **وظيفي بعمق**. كل قرار تصميمي يخدم الوضوح والاستخدام، لا المظهر.

### رؤية "البرمجيات الشخصية" (Software 3.0)

Notion يرى أن البرمجيات تطورت عبر ثلاثة أجيال:
1. **Desktop**: Word, Excel — أدوات فردية ثقيلة
2. **Cloud**: Figma, Airtable — تعاون سحابي
3. **Personalized**: **كل شخص يبني أداته الخاصة** ← هذا هو هدف Notion

---

## 2. نظام الألوان — مستخرج مباشرة من المسح

### لوحة الألوان الكاملة

| العنصر | الوصف | HEX |
|--------|-------|-----|
| `hero-bg-dark` | خلفية Hero الداكنة (Homepage) | `#02093A` |
| `dark-base` | الأساس الأغمق | `#01041B` |
| `cta-dark` | CTA على خلفية داكنة | `#455DD3` |
| `cta-light` | CTA على خلفية فاتحة | `#0075DE` |
| `btn-secondary-dark` | زر ثانوي على Dark | `#213183` |
| `page-bg` | خلفية الصفحة الرئيسية | `#FFFFFF` |
| `text-primary` | النص الأساسي | `#0D0D0D` |
| `text-muted` | النص الثانوي/الباهت | `#6B7280` (استنتاج) |
| `link` | روابط | `#1A73E8` |
| `input-bg` | خلفية الحقول | `#FFFFFF` |
| `input-border` | حدود الحقول | `#DDDDDD` |
| `input-text` | نص الحقول | `#000106` |

### قواعد استخدام الألوان

1. **الثنائية الأساسية**: Hero داكن (`#02093A`) ← → محتوى أبيض نقي (`#FFFFFF`)
2. **أزرق واحد فقط**: اللون الأزرق يُستخدم مرة واحدة كـ CTA في كل قسم
3. **لا تدرجات (No Gradients)** في أقسام المحتوى العادي
4. **اللون للدلالة** لا للزينة — كل لون له وظيفة معناها
5. **حياد في الباقي**: الرمادي الفاتح للبطاقات والحدود، لا أي لون آخر

### التطبيق العملي

```css
/* المناطق الداكنة (Hero, CTA sections) */
background: #02093A;
color: #FFFFFF;
--accent: #455DD3;

/* المناطق الفاتحة (محتوى عام) */
background: #FFFFFF;
color: #0D0D0D;
--accent: #0075DE;

/* البطاقات والخلفيات الثانوية */
background: #F9FAFB;
border: 1px solid #E5E7EB;
```

---

## 3. الطباعة (Typography)

### الخط الوحيد: Inter

Notion يستخدم **خطاً واحداً فقط** — Inter — في جميع الصفحات:

```css
font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
             Helvetica, "Apple Color Emoji", Arial, sans-serif,
             "Segoe UI Emoji", "Segoe UI Symbol";
```

### الهرمية الطباعية

| المستوى | الحجم | الوزن | الاستخدام |
|---------|-------|-------|-----------|
| Hero Display | **64px** | 700 (Bold) | العنوان الرئيسي للصفحة |
| H2 Section | ~40–48px | 700 | عناوين الأقسام الرئيسية |
| H3 Card | ~24px | 600 | عناوين البطاقات |
| Category Label | **12px** | 500 | تصنيفات فوق العناوين (UPPERCASE) |
| Body | **15–16px** | 400 | المحتوى العام |
| Small/Muted | 14px | 400 | النصوص المساعدة |
| Input | 14–16px | 400 | حقول النماذج |

### قواعد الطباعة

1. **خط واحد + أوزان مختلفة = تناسق كامل** (لا تنوع في الخطوط)
2. **الأوزان تُنشئ الهرمية** — لا الأحجام وحدها
3. **Category Labels**: `font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #6B7280`
4. **لا italic** في الواجهة الرئيسية
5. **Line height**: ~1.5 للـ body، ~1.1–1.2 للـ headings

---

## 4. المكوّنات — أبعاد وأنماط دقيقة

### الأزرار

```css
/* الزر الأساسي — على خلفية داكنة */
.btn-primary-dark {
  background: #455DD3;
  color: #FFFFFF;
  border-radius: 8px;
  box-shadow: none;
  font-weight: 500;
  padding: 10px 20px;
}

/* الزر الأساسي — على خلفية فاتحة */
.btn-primary-light {
  background: #0075DE;
  color: #FFFFFF;
  border-radius: 8px;
  box-shadow: none;
  font-weight: 500;
}

/* الزر الثانوي — على خلفية داكنة */
.btn-secondary-dark {
  background: #213183;
  color: #FFFFFF;
  border-radius: 8px;
  box-shadow: none;
}

/* الزر الثانوي — على خلفية فاتحة */
.btn-secondary-light {
  background: #FFFFFF;
  color: #000003;
  border: 1px solid #020834;
  border-radius: 8px;
  box-shadow: none;
}
```

### حقول الإدخال

```css
.input {
  background: #FFFFFF;
  color: #000106;
  border: 1px solid #DDDDDD;
  border-radius: 4px;  /* أصغر من الأزرار */
  box-shadow: none;
  padding: 10px 14px;
  font-size: 15px;
}

.input:focus {
  border-color: #0075DE;
  outline: none;
}
```

### نظام المسافات

```css
--space-xs:  4px;
--space-sm:  8px;
--space-md:  16px;
--space-lg:  24px;
--space-xl:  32px;
--space-2xl: 48px;
--space-3xl: 64px;
--space-4xl: 80px;  /* padding الأقسام */
--space-5xl: 120px; /* padding الأقسام الكبيرة */

--radius-sm:  4px;  /* inputs */
--radius-md:  8px;  /* buttons, cards */
--radius-lg:  12px; /* cards بعض */
--radius-full: 9999px; /* pills/badges */
```

---

## 5. أنماط التخطيط المُلاحظة

### 1. الثنائية Dual-Mode (الأهم)

```
┌──────────────────────────────────┐
│  DARK SECTION (#02093A)          │  ← Hero، يجذب الانتباه، يُثير الفضول
│  العنوان الكبير + CTA            │
├──────────────────────────────────┤
│  TRANSITION: Logo Strip          │  ← شعارات العملاء (dark أو light)
├──────────────────────────────────┤
│  LIGHT SECTION (#FFFFFF)         │  ← المحتوى، الشرح، التفاصيل
│  محتوى، بطاقات، ميزات            │
├──────────────────────────────────┤
│  DARK SECTION (#01041B)          │  ← CTA النهائي
└──────────────────────────────────┘
```

### 2. Bento Grid (للمزايا)

```
┌──────────────┬──────────────────┐
│   بطاقة 1   │                  │
│              │   بطاقة كبيرة   │
├──────────────┤     (2 صفوف)    │
│   بطاقة 2   │                  │
├──────────────┴──────────────────┤
│          بطاقة عريضة           │
└─────────────────────────────────┘
```
- كل بطاقة: label أعلى + عنوان + وصف + (صورة اختياري)
- Border: `1px solid #E5E7EB`
- Border-radius: `12px`
- Background: `#F9FAFB` أو `#FFFFFF`

### 3. شريط المنطقية المتحرك (Marquee)

```css
@keyframes marquee {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}
/* المحتوى مكرر مرتين للاستمرارية */
```

الإيقاع: بطيء ومستمر، لا يشتت، يُنقل أرقاماً دالة على الثقة.

### 4. شريط شعارات العملاء (Logo Strip)

- شعارات بالأبيض على خلفية داكنة
- أو شعارات رمادية على خلفية فاتحة
- أحجام طبيعية غير موحدة (not forced equal-size)
- "Trusted by teams at" أو "يثق بنا" فوق الشعارات

### 5. Pull Quotes (شهادات العملاء)

```
[شعار الشركة]
────────────────────────────────
"There's power in a single platform..."
  ← اقتباس طويل نسبياً، يُبرز الفائدة المحددة

Name — Role, Company
[Read full story →]
```

### 6. Category Labels (تصنيفات)

```html
<span class="label">NOTION HQ</span>
<h3>Introducing Custom Agents</h3>
```
```css
.label {
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #6B7280;
  margin-bottom: 8px;
  display: block;
}
```

### 7. قسم FAQ

```css
.faq-item {
  border-bottom: 1px solid #E5E7EB;
  padding: 20px 0;
}
/* فتح/إغلاق بـ JS بسيط — لا ألوان إضافية */
```

### 8. Statistics Grid

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Over 100M    │ #1 knowledge │ #1 AI search │ 62% Fortune  │
│ users        │ base (G2)    │ (G2)         │ 100          │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 6. الحركة والانتقالات

### فلسفة الحركة

> *"Motion serves understanding, not decoration."*

الحركة في Notion خافتة ومقصودة:

- **Fade-in عند التمرير**: `opacity: 0 → 1`, `transform: translateY(20px) → 0`
- **Marquee خطي**: CSS animation مستمرة وبطيئة
- **Hover states**: `opacity: 0.8`, أو `background` أفتح/أغمق قليلاً
- **Transitions**: `transition: all 150ms ease`

```css
/* نمط الدخول عند التمرير */
.fade-in {
  opacity: 0;
  transform: translateY(24px);
  transition: opacity 0.5s ease, transform 0.5s ease;
}
.fade-in.visible {
  opacity: 1;
  transform: translateY(0);
}
```

### ما يجب تجنبه في أسلوب Notion

| ❌ ممنوع | ✅ بديل |
|---------|--------|
| Parallax scrolling | Fade-in بسيط |
| Heavy particle effects | لا effects في الخلفية |
| Complex 3D transforms | Flat transitions |
| Bounce/Spring animations | Linear/Ease transitions |
| Multiple simultaneous animations | تسلسل منظم |

---

## 7. Navigation (القائمة)

### الهيكل
```
[شعار] ─── [Products ▼] [Solutions ▼] [Resources ▼] [Pricing] ─── [Log in] [Get Notion free]
```

### السلوك
- **Sticky**: تلتصق بالأعلى عند التمرير
- **Frosted glass** عند التمرير: `backdrop-filter: blur(12px); background: rgba(1,4,27,0.8)`
- **Height**: ~60–64px
- **Border bottom**: يظهر عند التمرير: `1px solid rgba(255,255,255,0.1)`

---

## 8. الشخصية والانطباع العام

من تحليل بيانات Firecrawl:

```json
"personality": {
  "tone": "modern",
  "energy": "medium",
  "targetAudience": "tech-savvy professionals"
}
```

### ما يجعل Notion مميزاً بصرياً

1. **الهدوء المقصود**: لا ضجيج، لا مبالغة
2. **الثقة بالمحتوى**: المنتج يتحدث عن نفسه — لا حاجة لإثارة مصطنعة
3. **الاتساق المطلق**: نفس الخط، نفس الألوان، نفس الحدود في كل مكان
4. **الـ Dark/Light Duality**: تناقض مقصود يُضفي عمقاً بصرياً
5. **Typography is the hero**: النص الكبير والنظيف هو عنصر التصميم الأقوى

---

## 9. CSS Design Tokens (مُستخرجة)

```css
:root {
  /* ألوان Notion الأصيلة */
  --notion-dark:        #01041B;
  --notion-dark-hero:   #02093A;
  --notion-dark-mid:    #213183;
  --notion-blue-dark:   #455DD3;
  --notion-blue-light:  #0075DE;
  --notion-white:       #FFFFFF;
  --notion-black:       #0D0D0D;
  --notion-link:        #1A73E8;
  --notion-input-border:#DDDDDD;

  /* طباعة */
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont,
               "Segoe UI", Helvetica, Arial, sans-serif;
  --text-hero:   clamp(40px, 5vw, 64px);
  --text-h2:     clamp(28px, 3.5vw, 48px);
  --text-h3:     clamp(18px, 2vw, 24px);
  --text-body:   16px;
  --text-sm:     14px;
  --text-label:  12px;

  /* مسافات */
  --space-4: 4px;
  --space-8: 8px;
  --space-16: 16px;
  --space-24: 24px;
  --space-32: 32px;
  --space-48: 48px;
  --space-64: 64px;
  --space-80: 80px;
  --space-120: 120px;

  /* حدود */
  --radius-input:  4px;
  --radius-btn:    8px;
  --radius-card:   12px;
  --radius-pill:   9999px;

  /* ظلال — لا ظلال في Notion */
  --shadow-none: none;
}
```

---

## 10. التطبيق على مراد بوت (خلاصة)

### التعديلات للعربية/RTL

| Notion (LTR) | مراد بوت (RTL) |
|--------------|---------------|
| Inter | IBM Plex Sans Arabic |
| `text-align: left` | `text-align: right` |
| `direction: ltr` | `direction: rtl` |
| `margin-left` | `margin-right` |

### لوحة الألوان المدمجة

```css
:root {
  /* من Notion */
  --hero-bg:      #01041B;
  --page-bg:      #FFFFFF;
  --card-bg:      #F9FAFB;
  --text-primary: #0D0D0D;
  --text-muted:   #6B7280;
  --border:       #E5E7EB;

  /* من MoradBot (يحل محل Notion's #455DD3) */
  --accent:       #2281F8;  /* MoradBot Primary */
  --accent-hover: #1A6FD4;

  /* مشترك */
  --font: "IBM Plex Sans Arabic", "Inter", sans-serif;
}
```

---

*آخر تحديث: مارس 2026 — موثّقة لـ مراد بوت Landing Page*
