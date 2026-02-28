-- MoradBot Seed Data
-- Run automatically by: supabase db reset
-- Applied to remote by: supabase db push --include-seed (or manually)

-- ── Plans ───────────────────────────────────────────────────
INSERT INTO plans (plan_name, plan_name_ar, bot_reply_limit, sync_frequency_hours, price_monthly_sar, features)
VALUES
  ('basic', 'الخطة الأساسية', 500, 24, 99.00, '{
    "support": "email",
    "branding": false,
    "analytics": "basic"
  }'::jsonb),
  ('mid', 'الخطة المتوسطة', 2000, 6, 299.00, '{
    "support": "priority_email",
    "branding": true,
    "analytics": "advanced"
  }'::jsonb),
  ('premium', 'الخطة المتقدمة', 10000, 1, 799.00, '{
    "support": "phone_email",
    "branding": true,
    "analytics": "premium",
    "dedicated_support": true
  }'::jsonb)
ON CONFLICT (plan_name) DO NOTHING;
