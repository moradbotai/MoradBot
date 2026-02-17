/**
 * MoradBot API
 * Cloudflare Workers backend
 */

import { Hono } from "hono";

const app = new Hono();

app.get("/", (c) => {
	return c.json({ message: "MoradBot API v0.1.0" });
});

export default app;
