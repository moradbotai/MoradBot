import { Hono } from "hono";
import type { HonoContext } from "../types";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";
import { NotFoundError } from "../lib/errors";

const ticketsRoutes = new Hono<HonoContext>();

// GET /api/tickets - List tickets
ticketsRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId")!;
  const supabase = createSupabaseClient(c.env, storeId);

  const status = c.req.query("status");

  let query = supabase
    .from("tickets")
    .select(`
      *,
      visitor_sessions(visitor_id, first_visit, last_visit)
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
  const storeId = c.get("storeId")!;
  const id = c.req.param("id");
  const supabase = createSupabaseClient(c.env, storeId);

  const { data: ticket, error: ticketError } = await supabase
    .from("tickets")
    .select(`
      *,
      visitor_sessions(*),
      messages(*)
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
