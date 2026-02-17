import type { Context, Next } from "hono";
import type { HonoContext } from "../types";
import { createSupabaseClient } from "../lib/supabase";
import { AuthenticationError } from "../lib/errors";

export async function requireAuth(c: Context<HonoContext>, next: Next) {
  const authHeader = c.req.header("Authorization");

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new AuthenticationError("Missing or invalid authorization header");
  }

  const token = authHeader.slice(7); // Remove "Bearer "

  // Verify token with Supabase
  const supabase = createSupabaseClient(c.env, "");
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
