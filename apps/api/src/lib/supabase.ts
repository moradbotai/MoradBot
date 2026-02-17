import { createClient } from "@supabase/supabase-js";
import type { Database } from "@moradbot/shared";
import type { Env } from "../env";

/**
 * Create Supabase client for authenticated merchant requests
 * Uses anon key + RLS (store_id in auth context)
 */
export function createSupabaseClient(env: Env, storeId: string) {
  const supabase = createClient<Database>(
    env.SUPABASE_URL,
    env.SUPABASE_ANON_KEY,
    {
      auth: {
        persistSession: false,
      },
      global: {
        headers: {
          "x-store-id": storeId,
        },
      },
    }
  );

  return supabase;
}

/**
 * Create Supabase admin client (bypasses RLS)
 * Use ONLY for system operations
 */
export function createSupabaseAdmin(env: Env) {
  return createClient<Database>(
    env.SUPABASE_URL,
    env.SUPABASE_SERVICE_ROLE_KEY,
    {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    }
  );
}
