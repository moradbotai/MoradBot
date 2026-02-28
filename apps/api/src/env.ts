/**
 * Cloudflare Workers environment bindings
 * Secrets → wrangler secret put
 * Non-secrets → wrangler.toml [vars]
 * Rule 4: No secrets in source code, DB, or .env in production
 */
export interface Env {
	// ── Supabase ─────────────────────────────────────────────
	SUPABASE_URL: string;
	SUPABASE_ANON_KEY: string;
	SUPABASE_SERVICE_ROLE_KEY: string;

	// ── Salla OAuth ───────────────────────────────────────────
	SALLA_CLIENT_ID: string;
	SALLA_CLIENT_SECRET: string;
	SALLA_REDIRECT_URI: string;

	// ── Encryption — AES-256-GCM (Phase 4) ───────────────────
	// Generate: openssl rand -hex 32
	ENCRYPTION_KEY: string;

	// ── OpenRouter (Phase 5) ──────────────────────────────────
	OPENROUTER_API_KEY: string;
	OPENROUTER_MODEL_PRIMARY: string;    // google/gemini-2.0-flash (باقات مدفوعة)
	OPENROUTER_MODEL_FREE_TIER: string;  // google/gemini-2.0-flash-exp:free (باقة مجانية)
	OPENROUTER_MODEL_FALLBACK_1: string; // openai/gpt-4o-mini

	// ── Email Notifications — Resend (Phase 5) ────────────────
	RESEND_API_KEY: string;
	RESEND_FROM_EMAIL: string;

	// ── Runtime (wrangler.toml [vars] — not secrets) ──────────
	ENVIRONMENT: "development" | "production";
	RATE_LIMIT_VISITOR_PER_MIN: string;
	RATE_LIMIT_STORE_PER_HOUR: string;

	// ── Cloudflare KV — Rate Limiting (Phase 4) ───────────────
	RATE_LIMIT_KV: KVNamespace;
}
