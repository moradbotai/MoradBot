export default {
	async fetch(request, env) {
		const url = new URL(request.url);

		// PostHog Query API proxy — /phq → PostHog HogQL query endpoint
		// Uses POSTHOG_PERSONAL_KEY secret — key never exposed to browser
		if (url.pathname === "/phq") {
			if (request.method === "OPTIONS") {
				return new Response(null, {
					headers: {
						"Access-Control-Allow-Origin": "*",
						"Access-Control-Allow-Methods": "POST, OPTIONS",
						"Access-Control-Allow-Headers": "Content-Type",
					},
				});
			}
			const key = env.POSTHOG_PERSONAL_KEY;
			if (!key) {
				return new Response(JSON.stringify({ error: "POSTHOG_PERSONAL_KEY not configured" }), {
					status: 500,
					headers: { "Content-Type": "application/json" },
				});
			}
			const body = await request.text();
			const resp = await fetch("https://us.posthog.com/api/projects/368154/query/", {
				method: "POST",
				headers: {
					Authorization: `Bearer ${key}`,
					"Content-Type": "application/json",
				},
				body,
			});
			const data = await resp.text();
			return new Response(data, {
				status: resp.status,
				headers: { "Content-Type": "application/json" },
			});
		}

		// Reverse proxy for PostHog — /ingest/* → PostHog servers
		// /ingest/static/* → us-assets.i.posthog.com (JS bundle)
		// /ingest/...     → us.i.posthog.com (event ingestion)
		if (url.pathname.startsWith("/ingest")) {
			const isAsset = url.pathname.startsWith("/ingest/static/");
			const hostname = isAsset ? "us-assets.i.posthog.com" : "us.i.posthog.com";
			const newUrl = new URL(request.url);
			newUrl.hostname = hostname;
			newUrl.pathname = url.pathname.replace(/^\/ingest/, "");
			newUrl.port = "";
			return fetch(new Request(newUrl, request));
		}

		// All other requests → serve static assets
		return env.ASSETS.fetch(request);
	},
};
