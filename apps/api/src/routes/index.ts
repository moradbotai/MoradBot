import { Hono } from "hono";
import type { HonoContext } from "../types";
import authRoutes from "./auth";
import chatRoutes from "./chat";
import faqRoutes from "./faq";
import statsRoutes from "./stats";
import ticketsRoutes from "./tickets";
import escalationsRoutes from "./escalations";

export function registerRoutes(app: Hono<HonoContext>) {
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
