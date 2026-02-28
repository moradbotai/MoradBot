# Environment Variables Reference — MoradBot

آخر تحديث: 22 فبراير 2026 — أضيف ENCRYPTION_KEY + RESEND

---

## 1. Development Environment

### Supabase

```env
SUPABASE_URL=https://qvujnhkfqwqfzkkweylk.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOi...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...   # للـ migrations فقط — لا تستخدمه في route handlers
```

### Salla OAuth (Development)

```env
SALLA_CLIENT_ID=xxxxx
SALLA_CLIENT_SECRET=xxxxx
SALLA_REDIRECT_URI=http://localhost:8787/auth/salla/callback
SALLA_AUTHORIZATION_URL=https://accounts.salla.sa/oauth2/authorize
SALLA_TOKEN_URL=https://accounts.salla.sa/oauth2/token
```

### Encryption (Phase 4)

```env
# توليد المفتاح: openssl rand -hex 32
ENCRYPTION_KEY=<64-char-hex-string>
```

يستخدم AES-256-GCM عبر Web Crypto API المدمجة في Cloudflare Workers — لا dependency خارجي.

### OpenRouter (Phase 5)

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL_PRIMARY=google/gemini-2.0-flash-exp:free
OPENROUTER_MODEL_FALLBACK_1=openai/gpt-4o-mini
OPENROUTER_MODEL_FALLBACK_2=anthropic/claude-3.5-sonnet
```

### Email Notifications — Resend (Phase 5)

```env
RESEND_API_KEY=re_...
RESEND_FROM_EMAIL=moradbot@yourdomain.com
```

يُستخدم لإشعار التاجر عند 80% و100% من حد الردود الشهري.

### Rate Limiting

```env
RATE_LIMIT_VISITOR_PER_MIN=20
RATE_LIMIT_STORE_PER_HOUR=3000
```

---

## 2. Production Environment

> **Rule 4 — لا استثناء:** جميع الأسرار في Cloudflare Secrets فقط. لا كود، لا DB، لا `.env` في production.

### Cloudflare Secrets (ترتيب التهيئة)

```bash
# Supabase
wrangler secret put SUPABASE_URL
wrangler secret put SUPABASE_ANON_KEY
wrangler secret put SUPABASE_SERVICE_ROLE_KEY

# Salla OAuth
wrangler secret put SALLA_CLIENT_ID
wrangler secret put SALLA_CLIENT_SECRET
wrangler secret put SALLA_REDIRECT_URI

# Encryption — أنشئ أولاً: openssl rand -hex 32
wrangler secret put ENCRYPTION_KEY

# OpenRouter (Phase 5)
wrangler secret put OPENROUTER_API_KEY

# Resend Email (Phase 5)
wrangler secret put RESEND_API_KEY
wrangler secret put RESEND_FROM_EMAIL
```

### التحقق من الـ Secrets المُعيَّنة

```bash
wrangler secret list
```

---

## 3. `env.ts` — TypeScript Interface الكاملة

```typescript
// apps/api/src/env.ts
export interface Env {
  // ── Supabase ─────────────────────────────────────
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;

  // ── Salla OAuth ───────────────────────────────────
  SALLA_CLIENT_ID: string;
  SALLA_CLIENT_SECRET: string;
  SALLA_REDIRECT_URI: string;

  // ── Encryption (Phase 4) ──────────────────────────
  ENCRYPTION_KEY: string;          // AES-256-GCM — 64 hex chars

  // ── OpenRouter (Phase 5) ─────────────────────────
  OPENROUTER_API_KEY: string;
  OPENROUTER_MODEL_PRIMARY: string;
  OPENROUTER_MODEL_FALLBACK_1: string;
  OPENROUTER_MODEL_FALLBACK_2: string;

  // ── Resend Email (Phase 5) ────────────────────────
  RESEND_API_KEY: string;
  RESEND_FROM_EMAIL: string;

  // ── Runtime ───────────────────────────────────────
  ENVIRONMENT: "development" | "production";

  // ── Cloudflare KV (Phase 4 — Rate Limiting) ───────
  RATE_LIMIT_KV: KVNamespace;
}
```

> `ENVIRONMENT` و `RATE_LIMIT_KV` تُعرَّف في `wrangler.toml` كـ `[vars]` وكـ KV binding — ليست secrets.

---

## 4. `wrangler.toml` — Non-Secret Bindings

```toml
[vars]
ENVIRONMENT = "production"
OPENROUTER_MODEL_PRIMARY = "google/gemini-2.0-flash-exp:free"
OPENROUTER_MODEL_FALLBACK_1 = "openai/gpt-4o-mini"
OPENROUTER_MODEL_FALLBACK_2 = "anthropic/claude-3.5-sonnet"

[[kv_namespaces]]
binding = "RATE_LIMIT_KV"
id = "<KV_NAMESPACE_ID>"           # من Cloudflare Dashboard
preview_id = "<KV_PREVIEW_ID>"    # للـ wrangler dev
```

---

## 5. Local Development — `.dev.vars`

Wrangler يقرأ `.dev.vars` تلقائياً عند `wrangler dev`. لا تُتتبع في git.

```env
# apps/api/.dev.vars  ← لا يُضاف لـ git
SUPABASE_URL=https://qvujnhkfqwqfzkkweylk.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SALLA_CLIENT_ID=
SALLA_CLIENT_SECRET=
SALLA_REDIRECT_URI=http://localhost:8787/auth/salla/callback
ENCRYPTION_KEY=<local-test-key-openssl-rand-hex-32>
OPENROUTER_API_KEY=
RESEND_API_KEY=
RESEND_FROM_EMAIL=test@localhost
```

---

## 6. كيفية الوصول في Runtime

### Cloudflare Worker

```typescript
// كل متغير يأتي من env binding — لا process.env في Workers
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = env.SUPABASE_URL;           // Cloudflare Secret
    const kv  = env.RATE_LIMIT_KV;          // KV Namespace binding
  },
};
```

### Hono Context (داخل route handlers)

```typescript
// Hono يمرر env عبر c.env
app.get("/example", (c) => {
  const supabase = createSupabaseClient(c.env, storeId);
  const key = c.env.ENCRYPTION_KEY;
});
```

> **ملاحظة:** `process.env` لا تعمل في Cloudflare Workers runtime. تعمل فقط في Vitest/Node عند الاختبار.

---

## 7. قواعد الأمان الصارمة

**ممنوع منعاً باتاً:**

- كتابة أي قيمة secret في الكود المصدري
- `git commit` لملفات `.env` أو `.dev.vars`
- تخزين secrets في قاعدة البيانات بنص صريح
- مشاركة secrets عبر أي قناة غير مشفرة

**المسموح:**

- `.env.example` بأسماء المتغيرات فقط (يُتتبع في git) ✅
- `.dev.vars` محلياً للتطوير (في `.gitignore`) ✅
- Cloudflare Secrets للـ production ✅
- `wrangler.toml` لـ non-secret vars فقط (ENVIRONMENT، model names) ✅

---

## 8. Checklist قبل كل Commit

- [ ] لا secrets في الكود المصدري
- [ ] `.env` و `.dev.vars` في `.gitignore`
- [ ] `.env.example` محدّث بأي متغير جديد (بدون قيم)
- [ ] لا قيم حقيقية في أي ملف مُتتبع بـ git
- [ ] `wrangler secret list` يُطابق قائمة القسم 2 كاملاً

---

## 9. إضافة متغير جديد — خطوات

1. أضف الاسم فقط في `.env.example`
2. أضف النوع في `apps/api/src/env.ts` ضمن `interface Env`
3. إذا كان secret: `wrangler secret put NEW_VAR`
4. إذا كان non-secret: أضفه في `[vars]` في `wrangler.toml`
5. للـ local: أضفه في `apps/api/.dev.vars`
