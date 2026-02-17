/**
 * Environment variables type definition
 * All secrets are stored in Cloudflare Secrets
 */
export interface Env {
  // Supabase
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;

  // Salla OAuth
  SALLA_CLIENT_ID: string;
  SALLA_CLIENT_SECRET: string;
  SALLA_REDIRECT_URI: string;

  // Environment
  ENVIRONMENT: "development" | "production";
}
