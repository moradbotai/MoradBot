import { Hono } from "hono";
import type { HonoContext } from "../types";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";

const statsRoutes = new Hono<HonoContext>();

// GET /api/stats - Dashboard statistics
statsRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId")!;
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
      plan_name: (subscription as any)?.plan_name,
      bot_reply_limit: (subscription as any)?.bot_reply_limit,
      current_cycle_usage: (subscription as any)?.current_cycle_usage,
      usage_percentage: subscription
        ? ((subscription as any).current_cycle_usage / (subscription as any).bot_reply_limit) * 100
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
  const storeId = c.get("storeId")!;
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
    .gte("created_at", (subscription as any).current_cycle_start)
    .lte("created_at", (subscription as any).current_cycle_end)
    .order("created_at", { ascending: false });

  return success(c, {
    events: events || [],
    total: events?.length || 0,
    limit: (subscription as any).bot_reply_limit,
    remaining: (subscription as any).bot_reply_limit - (events?.length || 0),
  });
});

export default statsRoutes;
