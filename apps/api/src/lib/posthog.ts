import { PostHog } from "posthog-node";
import type { Env } from "../env";

/**
 * Create a per-request PostHog client configured for the Cloudflare Workers
 * serverless environment. flushAt=1 / flushInterval=0 ensures each captured
 * event is sent immediately without waiting for a batch window.
 */
export function createPostHogClient(env: Env): PostHog {
	return new PostHog(env.POSTHOG_API_KEY, {
		host: env.POSTHOG_HOST,
		flushAt: 1,
		flushInterval: 0,
		enableExceptionAutocapture: true,
	});
}
