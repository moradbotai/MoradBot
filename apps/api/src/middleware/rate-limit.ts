import type { Context, Next } from "hono";
import { RateLimitError } from "../lib/errors";

// Simple in-memory rate limiter (for MVP)
// TODO: Use Cloudflare KV or Durable Objects for production
const requestCounts = new Map<string, { count: number; resetAt: number }>();

interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
  keyGenerator?: (c: Context) => string;
}

export function rateLimit(config: RateLimitConfig) {
  return async (c: Context, next: Next) => {
    const key = config.keyGenerator
      ? config.keyGenerator(c)
      : c.req.header("x-forwarded-for") || "unknown";

    const now = Date.now();
    const record = requestCounts.get(key);

    if (!record || now > record.resetAt) {
      // New window
      requestCounts.set(key, {
        count: 1,
        resetAt: now + config.windowMs,
      });
    } else if (record.count >= config.maxRequests) {
      // Rate limit exceeded
      throw new RateLimitError(
        `Rate limit exceeded. Try again in ${Math.ceil((record.resetAt - now) / 1000)}s`
      );
    } else {
      // Increment count
      record.count++;
    }

    await next();
  };
}

// Visitor rate limit: max 20 messages/min
export const visitorRateLimit = rateLimit({
  maxRequests: 20,
  windowMs: 60 * 1000, // 1 minute
  keyGenerator: (c) => {
    const visitorId = c.req.header("x-visitor-id");
    return `visitor:${visitorId || c.req.header("x-forwarded-for") || "unknown"}`;
  },
});

// Store rate limit: max 3000 bot replies/hour
export const storeRateLimit = rateLimit({
  maxRequests: 3000,
  windowMs: 60 * 60 * 1000, // 1 hour
  keyGenerator: (c) => {
    const storeId = c.get("storeId");
    return `store:${storeId}`;
  },
});
