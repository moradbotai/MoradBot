import { Hono } from "hono";
import type { HonoContext } from "./types";
import { errorHandler } from "./middleware/error-handler";
import { cors } from "./middleware/cors";
import { auditLog } from "./middleware/audit";
import { registerRoutes } from "./routes";

export function createApp() {
  const app = new Hono<HonoContext>();

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
