import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { HonoContext } from "../types";
import { success, error as errorResponse } from "../lib/responses";

const authRoutes = new Hono<HonoContext>();

// GET /auth/salla/start - Start OAuth flow
authRoutes.get("/salla/start", async (c) => {
  const clientId = c.env.SALLA_CLIENT_ID;
  const redirectUri = c.env.SALLA_REDIRECT_URI;

  const authUrl = new URL("https://accounts.salla.sa/oauth2/authorize");
  authUrl.searchParams.set("client_id", clientId);
  authUrl.searchParams.set("redirect_uri", redirectUri);
  authUrl.searchParams.set("response_type", "code");
  authUrl.searchParams.set("scope", "offline_access");

  return c.redirect(authUrl.toString());
});

// GET /auth/salla/callback - Handle OAuth callback
const callbackSchema = z.object({
  code: z.string(),
  state: z.string().optional(),
});

authRoutes.get(
  "/salla/callback",
  zValidator("query", callbackSchema),
  async (c) => {
    const { code } = c.req.valid("query");

    // Exchange code for access token
    const tokenResponse = await fetch("https://accounts.salla.sa/oauth2/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        grant_type: "authorization_code",
        client_id: c.env.SALLA_CLIENT_ID,
        client_secret: c.env.SALLA_CLIENT_SECRET,
        redirect_uri: c.env.SALLA_REDIRECT_URI,
        code,
      }),
    });

    if (!tokenResponse.ok) {
      return errorResponse(c, "OAUTH_ERROR", "Failed to exchange authorization code", 400);
    }

    const tokens = await tokenResponse.json() as any;

    // TODO: Store tokens in database and create store record
    // For now, return tokens to frontend
    return success(c, {
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
    });
  }
);

// POST /auth/salla/refresh - Refresh access token
const refreshSchema = z.object({
  refresh_token: z.string(),
});

authRoutes.post(
  "/salla/refresh",
  zValidator("json", refreshSchema),
  async (c) => {
    const { refresh_token } = c.req.valid("json");

    const tokenResponse = await fetch("https://accounts.salla.sa/oauth2/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        grant_type: "refresh_token",
        client_id: c.env.SALLA_CLIENT_ID,
        client_secret: c.env.SALLA_CLIENT_SECRET,
        refresh_token,
      }),
    });

    if (!tokenResponse.ok) {
      return errorResponse(c, "OAUTH_ERROR", "Failed to refresh token", 400);
    }

    const tokens = await tokenResponse.json() as any;

    return success(c, {
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      expires_in: tokens.expires_in,
    });
  }
);

export default authRoutes;
