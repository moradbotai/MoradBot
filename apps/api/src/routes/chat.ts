import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { HonoContext } from "../types";
import { success } from "../lib/responses";
import { requireAuth } from "../middleware/auth";
import { visitorRateLimit, storeRateLimit } from "../middleware/rate-limit";

const chatRoutes = new Hono<HonoContext>();

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
    const { ticket_id } = c.req.valid("json");

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
