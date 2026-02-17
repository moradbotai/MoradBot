# المرحلة 3: API Foundation — خطة التنفيذ

**التاريخ:** 17 فبراير 2026
**الحالة:** 🚧 قيد التنفيذ
**المدة المتوقعة:** 4-5 ساعات

---

## الأهداف

بناء أساس API باستخدام Cloudflare Workers + Hono framework مع:
1. ✅ بنية Routing محكمة وموثقة
2. ✅ تكامل كامل مع Supabase (مع RLS)
3. ✅ Authentication middleware
4. ✅ Error handling patterns موحدة
5. ✅ Rate limiting
6. ✅ Audit logging
7. ✅ Type safety من الـ database إلى الـ response

---

## المخرجات المطلوبة

### 1. ملفات التكوين (Configuration Files)
- [ ] `apps/api/wrangler.toml` - إعدادات Cloudflare Worker
- [ ] `apps/api/.dev.vars.example` - متغيرات البيئة (template)
- [ ] `apps/api/src/env.ts` - TypeScript types للبيئة

### 2. بنية المشروع (Project Structure)
```
apps/api/src/
├── index.ts                  # Entry point
├── env.ts                    # Environment types
├── app.ts                    # Hono app initialization
├── routes/
│   ├── index.ts             # Route registry
│   ├── auth.ts              # Authentication routes
│   ├── chat.ts              # Chat endpoint
│   ├── faq.ts               # FAQ management
│   ├── stats.ts             # Analytics/stats
│   ├── tickets.ts           # Tickets list
│   └── escalations.ts       # Escalation management
├── middleware/
│   ├── auth.ts              # Authentication middleware
│   ├── rate-limit.ts        # Rate limiting
│   ├── audit.ts             # Audit logging
│   ├── error-handler.ts     # Error handling
│   └── cors.ts              # CORS configuration
├── lib/
│   ├── supabase.ts          # Supabase client factory
│   ├── errors.ts            # Custom error classes
│   ├── responses.ts         # Standard response helpers
│   └── validators.ts        # Zod schemas for validation
└── types/
    └── index.ts             # Additional types
```

### 3. Endpoints المطلوبة

#### A. Authentication Endpoints (`/auth`)
- [ ] `GET /auth/salla/start` - بدء OAuth flow
- [ ] `GET /auth/salla/callback` - استقبال OAuth code
- [ ] `POST /auth/salla/refresh` - تجديد access token
- [ ] `POST /auth/verify` - التحقق من session

#### B. Chat Endpoint (`/api`)
- [ ] `POST /api/chat` - استقبال رسالة من Widget

#### C. FAQ Management (`/api`)
- [ ] `GET /api/faq` - قراءة FAQ entries
- [ ] `POST /api/faq` - إضافة FAQ entry
- [ ] `PUT /api/faq/:id` - تحديث FAQ entry
- [ ] `DELETE /api/faq/:id` - حذف FAQ entry

#### D. Analytics (`/api`)
- [ ] `GET /api/stats` - إحصائيات Dashboard
- [ ] `GET /api/stats/usage` - استهلاك الـ bot replies

#### E. Tickets & Escalations (`/api`)
- [ ] `GET /api/tickets` - قائمة المحادثات
- [ ] `GET /api/tickets/:id` - تفاصيل محادثة
- [ ] `GET /api/escalations` - قائمة التصعيدات
- [ ] `PATCH /api/escalations/:id` - تحديث حالة تصعيد

---

## التقنيات والأدوات

### Runtime & Framework
- **Cloudflare Workers** - Serverless runtime
- **Hono v4** - Fast web framework for edge
- **TypeScript 5.7** - Type safety
- **Wrangler 3** - Development & deployment tool

### Database & Auth
- **Supabase Client** - Database access
- **@supabase/supabase-js** - Official JS client
- RLS policies (من المرحلة 2)

### Validation & Security
- **Zod** - Schema validation
- **@hono/zod-validator** - Zod integration for Hono
- Rate limiting (custom middleware)
- CORS configuration

### Error Handling
- Custom error classes
- Standardized error responses
- Error logging to Supabase audit_logs

---

## التنفيذ التفصيلي

### الخطوة 1: إعداد Cloudflare Worker (30 دقيقة)

#### 1.1 ملف `wrangler.toml`
```toml
name = "moradbot-api"
main = "src/index.ts"
compatibility_date = "2024-02-17"
node_compat = true

[vars]
ENVIRONMENT = "development"

# Secrets (via wrangler secret put)
# SUPABASE_URL
# SUPABASE_ANON_KEY
# SUPABASE_SERVICE_ROLE_KEY
# SALLA_CLIENT_ID
# SALLA_CLIENT_SECRET
```

#### 1.2 ملف `.dev.vars.example`
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SALLA_CLIENT_ID=your-salla-client-id
SALLA_CLIENT_SECRET=your-salla-client-secret
SALLA_REDIRECT_URI=http://localhost:8787/auth/salla/callback
```

#### 1.3 ملف `src/env.ts` - Environment Types
```typescript
export interface Env {
  // Supabase
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;

  // Salla OAuth
  SALLA_CLIENT_ID: string;
  SALLA_CLIENT_SECRET: string;
  SALLA_REDIRECT_URI: string;

  // Environment
  ENVIRONMENT: "development" | "production";
}
```

---

### الخطوة 2: Supabase Client Integration (30 دقيقة)

#### 2.1 ملف `src/lib/supabase.ts`
```typescript
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@moradbot/shared";
import type { Env } from "../env";

/**
 * Create Supabase client for authenticated merchant requests
 * Uses anon key + RLS (store_id in JWT)
 */
export function createSupabaseClient(env: Env, storeId: string) {
  const supabase = createClient<Database>(
    env.SUPABASE_URL,
    env.SUPABASE_ANON_KEY,
    {
      auth: {
        // Set store_id in JWT for RLS
        persistSession: false,
      },
      global: {
        headers: {
          "x-store-id": storeId,
        },
      },
    }
  );

  return supabase;
}

/**
 * Create Supabase admin client (bypasses RLS)
 * Use ONLY for system operations
 */
export function createSupabaseAdmin(env: Env) {
  return createClient<Database>(
    env.SUPABASE_URL,
    env.SUPABASE_SERVICE_ROLE_KEY,
    {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    }
  );
}
```

---

### الخطوة 3: Error Handling System (20 دقيقة)

#### 3.1 ملف `src/lib/errors.ts`
```typescript
export class AppError extends Error {
  constructor(
    public statusCode: number,
    public code: string,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = "AppError";
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(400, "VALIDATION_ERROR", message, details);
    this.name = "ValidationError";
  }
}

export class AuthenticationError extends AppError {
  constructor(message = "Authentication required") {
    super(401, "AUTHENTICATION_ERROR", message);
    this.name = "AuthenticationError";
  }
}

export class AuthorizationError extends AppError {
  constructor(message = "Insufficient permissions") {
    super(403, "AUTHORIZATION_ERROR", message);
    this.name = "AuthorizationError";
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string) {
    super(404, "NOT_FOUND", `${resource} not found`);
    this.name = "NotFoundError";
  }
}

export class RateLimitError extends AppError {
  constructor(message = "Rate limit exceeded") {
    super(429, "RATE_LIMIT_EXCEEDED", message);
    this.name = "RateLimitError";
  }
}

export class DatabaseError extends AppError {
  constructor(message: string, details?: unknown) {
    super(500, "DATABASE_ERROR", message, details);
    this.name = "DatabaseError";
  }
}
```

#### 3.2 ملف `src/lib/responses.ts`
```typescript
import type { Context } from "hono";

export interface SuccessResponse<T = unknown> {
  success: true;
  data: T;
  meta?: {
    timestamp: string;
    requestId?: string;
  };
}

export interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
  meta: {
    timestamp: string;
    requestId?: string;
  };
}

export function success<T>(
  c: Context,
  data: T,
  status = 200
): Response {
  const response: SuccessResponse<T> = {
    success: true,
    data,
    meta: {
      timestamp: new Date().toISOString(),
      requestId: c.req.header("x-request-id"),
    },
  };

  return c.json(response, status);
}

export function error(
  c: Context,
  code: string,
  message: string,
  status = 500,
  details?: unknown
): Response {
  const response: ErrorResponse = {
    success: false,
    error: {
      code,
      message,
      details,
    },
    meta: {
      timestamp: new Date().toISOString(),
      requestId: c.req.header("x-request-id"),
    },
  };

  return c.json(response, status);
}
```

---

### الخطوة 4: Middleware Implementation (45 دقيقة)

#### 4.1 ملف `src/middleware/error-handler.ts`
```typescript
import type { Context, Next } from "hono";
import { AppError } from "../lib/errors";
import { error as errorResponse } from "../lib/responses";

export async function errorHandler(c: Context, next: Next) {
  try {
    await next();
  } catch (err) {
    console.error("Error:", err);

    // Handle AppError instances
    if (err instanceof AppError) {
      return errorResponse(
        c,
        err.code,
        err.message,
        err.statusCode,
        err.details
      );
    }

    // Handle unknown errors
    return errorResponse(
      c,
      "INTERNAL_ERROR",
      "An unexpected error occurred",
      500
    );
  }
}
```

#### 4.2 ملف `src/middleware/auth.ts`
```typescript
import type { Context, Next } from "hono";
import type { Env } from "../env";
import { createSupabaseClient } from "../lib/supabase";
import { AuthenticationError } from "../lib/errors";

export async function requireAuth(c: Context<{ Bindings: Env }>, next: Next) {
  const authHeader = c.req.header("Authorization");

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new AuthenticationError("Missing or invalid authorization header");
  }

  const token = authHeader.slice(7); // Remove "Bearer "

  // Verify token with Supabase
  const supabase = createSupabaseClient(c.env, ""); // store_id will be extracted from token
  const { data: { user }, error } = await supabase.auth.getUser(token);

  if (error || !user) {
    throw new AuthenticationError("Invalid or expired token");
  }

  // Extract store_id from user metadata
  const storeId = user.id; // Assuming user.id IS the store_id

  if (!storeId) {
    throw new AuthenticationError("Store ID not found in token");
  }

  // Store in context for route handlers
  c.set("storeId", storeId);
  c.set("userId", user.id);

  await next();
}
```

#### 4.3 ملف `src/middleware/rate-limit.ts`
```typescript
import type { Context, Next } from "hono";
import { RateLimitError } from "../lib/errors";

// Simple in-memory rate limiter (for MVP)
// TODO: Use Cloudflare KV or Durable Objects for production
const requestCounts = new Map<string, { count: number; resetAt: number }>();

interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
  keyGenerator?: (c: Context) => string;
}

export function rateLimit(config: RateLimitConfig) {
  return async (c: Context, next: Next) => {
    const key = config.keyGenerator
      ? config.keyGenerator(c)
      : c.req.header("x-forwarded-for") || "unknown";

    const now = Date.now();
    const record = requestCounts.get(key);

    if (!record || now > record.resetAt) {
      // New window
      requestCounts.set(key, {
        count: 1,
        resetAt: now + config.windowMs,
      });
    } else if (record.count >= config.maxRequests) {
      // Rate limit exceeded
      throw new RateLimitError(
        `Rate limit exceeded. Try again in ${Math.ceil((record.resetAt - now) / 1000)}s`
      );
    } else {
      // Increment count
      record.count++;
    }

    await next();
  };
}

// Visitor rate limit: max 20 messages/min
export const visitorRateLimit = rateLimit({
  maxRequests: 20,
  windowMs: 60 * 1000, // 1 minute
  keyGenerator: (c) => {
    // Use visitor_id from request body or IP
    const visitorId = c.req.header("x-visitor-id");
    return `visitor:${visitorId || c.req.header("x-forwarded-for") || "unknown"}`;
  },
});

// Store rate limit: max 3000 bot replies/hour
export const storeRateLimit = rateLimit({
  maxRequests: 3000,
  windowMs: 60 * 60 * 1000, // 1 hour
  keyGenerator: (c) => {
    const storeId = c.get("storeId");
    return `store:${storeId}`;
  },
});
```

#### 4.4 ملف `src/middleware/audit.ts`
```typescript
import type { Context, Next } from "hono";
import type { Env } from "../env";
import { createSupabaseAdmin } from "../lib/supabase";

export async function auditLog(c: Context<{ Bindings: Env }>, next: Next) {
  const startTime = Date.now();

  await next();

  const duration = Date.now() - startTime;
  const storeId = c.get("storeId");
  const userId = c.get("userId");

  // Log to Supabase audit_logs (async, don't block response)
  const supabase = createSupabaseAdmin(c.env);

  supabase
    .from("audit_logs")
    .insert({
      store_id: storeId || null,
      actor_type: storeId ? "merchant" : "system",
      actor_id: userId || null,
      action: `${c.req.method} ${c.req.path}`,
      resource_type: "api_request",
      ip_address: c.req.header("x-forwarded-for") || null,
      user_agent: c.req.header("user-agent") || null,
      metadata: {
        duration_ms: duration,
        status: c.res.status,
      },
    })
    .then(({ error }) => {
      if (error) console.error("Audit log error:", error);
    });
}
```

#### 4.5 ملف `src/middleware/cors.ts`
```typescript
import { cors as honoCors } from "hono/cors";

export const cors = honoCors({
  origin: ["http://localhost:3000", "https://*.salla.sa"],
  credentials: true,
  allowMethods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
  allowHeaders: ["Content-Type", "Authorization", "x-visitor-id", "x-request-id"],
  exposeHeaders: ["x-request-id"],
  maxAge: 600,
});
```

---

### الخطوة 5: Routes Implementation (90 دقيقة)

#### 5.1 ملف `src/routes/auth.ts`
```typescript
import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { Env } from "../env";
import { success, error as errorResponse } from "../lib/responses";
import { createSupabaseAdmin } from "../lib/supabase";

const authRoutes = new Hono<{ Bindings: Env }>();

// GET /auth/salla/start - Start OAuth flow
authRoutes.get("/salla/start", async (c) => {
  const clientId = c.env.SALLA_CLIENT_ID;
  const redirectUri = c.env.SALLA_REDIRECT_URI;

  const authUrl = new URL("https://accounts.salla.sa/oauth2/authorize");
  authUrl.searchParams.set("client_id", clientId);
  authUrl.searchParams.set("redirect_uri", redirectUri);
  authUrl.searchParams.set("response_type", "code");
  authUrl.searchParams.set("scope", "offline_access");

  return c.redirect(authUrl.toString());
});

// GET /auth/salla/callback - Handle OAuth callback
const callbackSchema = z.object({
  code: z.string(),
  state: z.string().optional(),
});

authRoutes.get(
  "/salla/callback",
  zValidator("query", callbackSchema),
  async (c) => {
    const { code } = c.req.valid("query");

    // Exchange code for access token
    const tokenResponse = await fetch("https://accounts.salla.sa/oauth2/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        grant_type: "authorization_code",
        client_id: c.env.SALLA_CLIENT_ID,
        client_secret: c.env.SALLA_CLIENT_SECRET,
        redirect_uri: c.env.SALLA_REDIRECT_URI,
        code,
      }),
    });

    if (!tokenResponse.ok) {
      return errorResponse(c, "OAUTH_ERROR", "Failed to exchange authorization code", 400);
    }

    const tokens = await tokenResponse.json();

    // TODO: Store tokens in database and create store record
    // For now, return tokens to frontend
    return success(c, {
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
    });
  }
);

// POST /auth/salla/refresh - Refresh access token
const refreshSchema = z.object({
  refresh_token: z.string(),
});

authRoutes.post(
  "/salla/refresh",
  zValidator("json", refreshSchema),
  async (c) => {
    const { refresh_token } = c.req.valid("json");

    const tokenResponse = await fetch("https://accounts.salla.sa/oauth2/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        grant_type: "refresh_token",
        client_id: c.env.SALLA_CLIENT_ID,
        client_secret: c.env.SALLA_CLIENT_SECRET,
        refresh_token,
      }),
    });

    if (!tokenResponse.ok) {
      return errorResponse(c, "OAUTH_ERROR", "Failed to refresh token", 400);
    }

    const tokens = await tokenResponse.json();

    return success(c, {
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
    });
  }
);

export default authRoutes;
```

#### 5.2 ملف `src/routes/chat.ts`
```typescript
import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { Env } from "../env";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { visitorRateLimit, storeRateLimit } from "../middleware/rate-limit";

const chatRoutes = new Hono<{ Bindings: Env }>();

const chatMessageSchema = z.object({
  visitor_id: z.string().uuid(),
  ticket_id: z.string().uuid().optional(),
  message: z.string().min(1).max(1000),
  context: z.object({
    page_url: z.string().url().optional(),
    user_agent: z.string().optional(),
  }).optional(),
});

// POST /api/chat - Handle chat message
chatRoutes.post(
  "/",
  requireAuth,
  visitorRateLimit,
  storeRateLimit,
  zValidator("json", chatMessageSchema),
  async (c) => {
    const storeId = c.get("storeId");
    const { visitor_id, ticket_id, message, context } = c.req.valid("json");

    // TODO: Process message with AI orchestrator
    // For now, return mock response
    return success(c, {
      ticket_id: ticket_id || crypto.randomUUID(),
      message_id: crypto.randomUUID(),
      bot_response: "شكراً لك! سأساعدك في الإجابة على سؤالك.",
      needs_clarification: false,
      escalated: false,
    });
  }
);

export default chatRoutes;
```

#### 5.3 ملف `src/routes/faq.ts`
```typescript
import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { Env } from "../env";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";
import { NotFoundError } from "../lib/errors";

const faqRoutes = new Hono<{ Bindings: Env }>();

// GET /api/faq - List all FAQ entries
faqRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const supabase = createSupabaseClient(c.env, storeId);

  const { data, error } = await supabase
    .from("faq_entries")
    .select("*")
    .eq("store_id", storeId)
    .eq("is_active", true)
    .order("created_at", { ascending: false });

  if (error) {
    throw new Error(error.message);
  }

  return success(c, data);
});

const createFaqSchema = z.object({
  category: z.enum(["shipping", "payment", "returns", "contact", "other"]),
  question_ar: z.string().min(5).max(500),
  answer_ar: z.string().min(10).max(2000),
  tags: z.array(z.string()).optional(),
});

// POST /api/faq - Create new FAQ entry
faqRoutes.post(
  "/",
  requireAuth,
  zValidator("json", createFaqSchema),
  async (c) => {
    const storeId = c.get("storeId");
    const body = c.req.valid("json");
    const supabase = createSupabaseClient(c.env, storeId);

    const { data, error } = await supabase
      .from("faq_entries")
      .insert({
        store_id: storeId,
        ...body,
      })
      .select()
      .single();

    if (error) {
      throw new Error(error.message);
    }

    return success(c, data, 201);
  }
);

const updateFaqSchema = createFaqSchema.partial();

// PUT /api/faq/:id - Update FAQ entry
faqRoutes.put(
  "/:id",
  requireAuth,
  zValidator("json", updateFaqSchema),
  async (c) => {
    const storeId = c.get("storeId");
    const id = c.req.param("id");
    const body = c.req.valid("json");
    const supabase = createSupabaseClient(c.env, storeId);

    const { data, error } = await supabase
      .from("faq_entries")
      .update(body)
      .eq("faq_entry_id", id)
      .eq("store_id", storeId)
      .select()
      .single();

    if (error || !data) {
      throw new NotFoundError("FAQ entry");
    }

    return success(c, data);
  }
);

// DELETE /api/faq/:id - Delete FAQ entry (soft delete)
faqRoutes.delete("/:id", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const id = c.req.param("id");
  const supabase = createSupabaseClient(c.env, storeId);

  const { data, error } = await supabase
    .from("faq_entries")
    .update({ is_active: false })
    .eq("faq_entry_id", id)
    .eq("store_id", storeId)
    .select()
    .single();

  if (error || !data) {
    throw new NotFoundError("FAQ entry");
  }

  return success(c, { deleted: true });
});

export default faqRoutes;
```

#### 5.4 ملف `src/routes/stats.ts`
```typescript
import { Hono } from "hono";
import type { Env } from "../env";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";

const statsRoutes = new Hono<{ Bindings: Env }>();

// GET /api/stats - Dashboard statistics
statsRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const supabase = createSupabaseClient(c.env, storeId);

  // Get active subscription
  const { data: subscription } = await supabase
    .from("v_active_subscriptions")
    .select("*")
    .eq("store_id", storeId)
    .single();

  // Get ticket stats
  const { count: totalTickets } = await supabase
    .from("tickets")
    .select("*", { count: "exact", head: true })
    .eq("store_id", storeId);

  const { count: resolvedTickets } = await supabase
    .from("tickets")
    .select("*", { count: "exact", head: true })
    .eq("store_id", storeId)
    .eq("status", "resolved");

  const { count: escalatedTickets } = await supabase
    .from("tickets")
    .select("*", { count: "exact", head: true })
    .eq("store_id", storeId)
    .eq("status", "escalated");

  return success(c, {
    subscription: {
      plan_name: subscription?.plan_name,
      bot_reply_limit: subscription?.bot_reply_limit,
      current_cycle_usage: subscription?.current_cycle_usage,
      usage_percentage: subscription
        ? (subscription.current_cycle_usage / subscription.bot_reply_limit) * 100
        : 0,
    },
    tickets: {
      total: totalTickets || 0,
      resolved: resolvedTickets || 0,
      escalated: escalatedTickets || 0,
      resolution_rate: totalTickets
        ? ((resolvedTickets || 0) / totalTickets) * 100
        : 0,
    },
  });
});

// GET /api/stats/usage - Usage details
statsRoutes.get("/usage", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const supabase = createSupabaseClient(c.env, storeId);

  // Get usage events for current billing cycle
  const { data: subscription } = await supabase
    .from("v_active_subscriptions")
    .select("*")
    .eq("store_id", storeId)
    .single();

  if (!subscription) {
    return success(c, { events: [], total: 0 });
  }

  const { data: events } = await supabase
    .from("usage_events")
    .select("*")
    .eq("store_id", storeId)
    .gte("created_at", subscription.current_cycle_start)
    .lte("created_at", subscription.current_cycle_end)
    .order("created_at", { ascending: false });

  return success(c, {
    events: events || [],
    total: events?.length || 0,
    limit: subscription.bot_reply_limit,
    remaining: subscription.bot_reply_limit - (events?.length || 0),
  });
});

export default statsRoutes;
```

#### 5.5 ملف `src/routes/tickets.ts`
```typescript
import { Hono } from "hono";
import type { Env } from "../env";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";
import { NotFoundError } from "../lib/errors";

const ticketsRoutes = new Hono<{ Bindings: Env }>();

// GET /api/tickets - List tickets
ticketsRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const supabase = createSupabaseClient(c.env, storeId);

  const status = c.req.query("status"); // filter by status

  let query = supabase
    .from("tickets")
    .select(`
      *,
      visitor_sessions(visitor_id, first_visit, last_visit),
      messages(count)
    `)
    .eq("store_id", storeId);

  if (status) {
    query = query.eq("status", status);
  }

  const { data, error } = await query.order("created_at", { ascending: false });

  if (error) {
    throw new Error(error.message);
  }

  return success(c, data);
});

// GET /api/tickets/:id - Get ticket details with messages
ticketsRoutes.get("/:id", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const id = c.req.param("id");
  const supabase = createSupabaseClient(c.env, storeId);

  const { data: ticket, error: ticketError } = await supabase
    .from("tickets")
    .select(`
      *,
      visitor_sessions(*),
      messages(*),
      escalations(*)
    `)
    .eq("ticket_id", id)
    .eq("store_id", storeId)
    .single();

  if (ticketError || !ticket) {
    throw new NotFoundError("Ticket");
  }

  return success(c, ticket);
});

export default ticketsRoutes;
```

#### 5.6 ملف `src/routes/escalations.ts`
```typescript
import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { Env } from "../env";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";
import { NotFoundError } from "../lib/errors";

const escalationsRoutes = new Hono<{ Bindings: Env }>();

// GET /api/escalations - List pending escalations
escalationsRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId");
  const supabase = createSupabaseClient(c.env, storeId);

  const { data, error } = await supabase
    .from("v_pending_escalations")
    .select("*")
    .eq("store_id", storeId);

  if (error) {
    throw new Error(error.message);
  }

  return success(c, data);
});

const updateEscalationSchema = z.object({
  status: z.enum(["pending", "in_progress", "resolved"]),
  resolution_notes: z.string().optional(),
});

// PATCH /api/escalations/:id - Update escalation status
escalationsRoutes.patch(
  "/:id",
  requireAuth,
  zValidator("json", updateEscalationSchema),
  async (c) => {
    const storeId = c.get("storeId");
    const id = c.req.param("id");
    const body = c.req.valid("json");
    const supabase = createSupabaseClient(c.env, storeId);

    const updateData: any = {
      status: body.status,
    };

    if (body.status === "resolved") {
      updateData.resolved_at = new Date().toISOString();
      updateData.resolution_notes = body.resolution_notes;
    }

    const { data, error } = await supabase
      .from("escalations")
      .update(updateData)
      .eq("escalation_id", id)
      .eq("store_id", storeId)
      .select()
      .single();

    if (error || !data) {
      throw new NotFoundError("Escalation");
    }

    return success(c, data);
  }
);

export default escalationsRoutes;
```

#### 5.7 ملف `src/routes/index.ts` - Route Registry
```typescript
import { Hono } from "hono";
import type { Env } from "../env";
import authRoutes from "./auth";
import chatRoutes from "./chat";
import faqRoutes from "./faq";
import statsRoutes from "./stats";
import ticketsRoutes from "./tickets";
import escalationsRoutes from "./escalations";

export function registerRoutes(app: Hono<{ Bindings: Env }>) {
  // Health check
  app.get("/health", (c) => c.json({ status: "ok", timestamp: new Date().toISOString() }));

  // Auth routes (no /api prefix)
  app.route("/auth", authRoutes);

  // API routes
  app.route("/api/chat", chatRoutes);
  app.route("/api/faq", faqRoutes);
  app.route("/api/stats", statsRoutes);
  app.route("/api/tickets", ticketsRoutes);
  app.route("/api/escalations", escalationsRoutes);

  return app;
}
```

---

### الخطوة 6: Hono App Setup (20 دقيقة)

#### 6.1 ملف `src/app.ts`
```typescript
import { Hono } from "hono";
import type { Env } from "./env";
import { errorHandler } from "./middleware/error-handler";
import { cors } from "./middleware/cors";
import { auditLog } from "./middleware/audit";
import { registerRoutes } from "./routes";

export function createApp() {
  const app = new Hono<{ Bindings: Env }>();

  // Global middleware
  app.use("*", cors);
  app.use("*", errorHandler);
  app.use("*", auditLog);

  // Register routes
  registerRoutes(app);

  // 404 handler
  app.notFound((c) => {
    return c.json({
      success: false,
      error: {
        code: "NOT_FOUND",
        message: "Route not found",
      },
      meta: {
        timestamp: new Date().toISOString(),
      },
    }, 404);
  });

  return app;
}
```

#### 6.2 ملف `src/index.ts` - Entry Point
```typescript
import { createApp } from "./app";

const app = createApp();

export default app;
```

---

### الخطوة 7: Dependencies Installation (10 دقيقة)

إضافة dependencies المطلوبة لـ `apps/api/package.json`:

```json
{
  "dependencies": {
    "@moradbot/shared": "workspace:*",
    "@supabase/supabase-js": "^2.48.0",
    "hono": "^4.6.14",
    "@hono/zod-validator": "^0.4.1",
    "zod": "^3.24.1"
  }
}
```

---

## معايير النجاح

### Technical Success Criteria
- [ ] جميع الـ endpoints تعمل بدون أخطاء
- [ ] Type safety كاملة من database إلى response
- [ ] RLS policies تعمل بشكل صحيح (store isolation)
- [ ] Rate limiting يعمل على جميع الـ endpoints
- [ ] Audit logging يسجل جميع الطلبات
- [ ] Error handling موحد على جميع الـ routes
- [ ] CORS configured بشكل صحيح

### Code Quality Criteria
- [ ] TypeScript بدون errors
- [ ] Biome linting passes
- [ ] Build successful (`pnpm build`)
- [ ] No hardcoded secrets
- [ ] Proper error messages (Arabic for user-facing)

### Documentation Criteria
- [ ] جميع الـ endpoints موثقة
- [ ] Environment variables documented
- [ ] Deployment instructions created

---

## الخطوات التالية (المرحلة 4)

بعد اكتمال المرحلة 3:
- **المرحلة 4:** Salla Integration
  - OAuth implementation completion
  - Products API client
  - Sync service with cron
  - Token refresh mechanism

---

**نهاية خطة المرحلة 3**
