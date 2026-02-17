import { cors as honoCors } from "hono/cors";

export const cors = honoCors({
  origin: ["http://localhost:3000", "https://*.salla.sa"],
  credentials: true,
  allowMethods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
  allowHeaders: ["Content-Type", "Authorization", "x-visitor-id", "x-request-id"],
  exposeHeaders: ["x-request-id"],
  maxAge: 600,
});
