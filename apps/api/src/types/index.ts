import type { Env } from "../env";

export type Variables = {
  storeId?: string;
  userId?: string;
};

export type HonoContext = { Bindings: Env; Variables: Variables };
