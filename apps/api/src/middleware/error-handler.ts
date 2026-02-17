import type { Context, Next } from "hono";
import { AppError } from "../lib/errors";
import { error as errorResponse } from "../lib/responses";

export async function errorHandler(c: Context, next: Next): Promise<Response | void> {
  try {
    await next();
  } catch (err) {
    console.error("Error:", err);

    // Handle AppError instances
    if (err instanceof AppError) {
      return errorResponse(
        c,
        err.code,
        err.message,
        err.statusCode,
        err.details
      );
    }

    // Handle unknown errors
    return errorResponse(
      c,
      "INTERNAL_ERROR",
      "An unexpected error occurred",
      500
    );
  }
}
