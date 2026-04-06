export default {
	async fetch(request, env) {
		const url = new URL(request.url);

		// Reverse proxy for PostHog — /ingest/* → PostHog servers
		if (url.pathname.startsWith("/ingest")) {
			const hostname = "us.i.posthog.com";
			const newUrl = new URL(request.url);
			newUrl.hostname = hostname;
			newUrl.port = "";

			const newRequest = new Request(newUrl, request);
			return fetch(newRequest);
		}

		// All other requests → serve static assets
		return env.ASSETS.fetch(request);
	},
};
