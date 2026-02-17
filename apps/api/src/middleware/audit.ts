import type { Context, Next } from "hono";
import type { HonoContext } from "../types";
import { createSupabaseAdmin } from "../lib/supabase";

export async function auditLog(c: Context<HonoContext>, next: Next) {
  const startTime = Date.now();

  await next();

  const duration = Date.now() - startTime;
  const storeId = c.get("storeId");
  const userId = c.get("userId");

  // Log to Supabase audit_logs (async, don't block response)
  const supabase = createSupabaseAdmin(c.env) as any;

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
    .then(({ error }: any) => {
      if (error) console.error("Audit log error:", error);
    });
}
