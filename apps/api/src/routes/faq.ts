import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { HonoContext } from "../types";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { createSupabaseClient } from "../lib/supabase";
import { NotFoundError } from "../lib/errors";

const faqRoutes = new Hono<HonoContext>();

// GET /api/faq - List all FAQ entries
faqRoutes.get("/", requireAuth, async (c) => {
  const storeId = c.get("storeId")!;
  const supabase = createSupabaseClient(c.env, storeId) as any;

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
    const storeId = c.get("storeId")!;
    const body = c.req.valid("json");
    const supabase = createSupabaseClient(c.env, storeId) as any;

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
    const storeId = c.get("storeId")!;
    const id = c.req.param("id");
    const body = c.req.valid("json");
    const supabase = createSupabaseClient(c.env, storeId) as any;

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
  const storeId = c.get("storeId")!;
  const id = c.req.param("id");
  const supabase = createSupabaseClient(c.env, storeId) as any;

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
