import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { HonoContext } from "../types";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";
import { NotFoundError } from "../lib/errors";

const escalationsRoutes = new Hono<HonoContext>();

// GET /api/escalations - List pending escalations
escalationsRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId")!;
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
    const storeId = c.get("storeId")!;
    const id = c.req.param("id");
    const body = c.req.valid("json");
    const supabase = createSupabaseClient(c.env, storeId) as any;

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
