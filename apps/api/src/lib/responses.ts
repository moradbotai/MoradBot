import type { Context } from "hono";

export interface SuccessResponse<T = unknown> {
  success: true;
  data: T;
  meta?: {
    timestamp: string;
    requestId?: string;
  };
}

export interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
  meta: {
    timestamp: string;
    requestId?: string;
  };
}

export function success<T>(
  c: Context,
  data: T,
  status: number = 200
): Response {
  const requestId = c.req.header("x-request-id");
  const response: SuccessResponse<T> = {
    success: true,
    data,
    meta: {
      timestamp: new Date().toISOString(),
      ...(requestId ? { requestId } : {}),
    },
  };

  return c.json(response, status as any);
}

export function error(
  c: Context,
  code: string,
  message: string,
  status: number = 500,
  details?: unknown
): Response {
  const requestId = c.req.header("x-request-id");
  const response: ErrorResponse = {
    success: false,
    error: {
      code,
      message,
      details,
    },
    meta: {
      timestamp: new Date().toISOString(),
      ...(requestId ? { requestId } : {}),
    },
  };

  return c.json(response, status as any);
}
