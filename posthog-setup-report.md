<wizard-report>
# PostHog post-wizard report

The wizard has completed a deep integration of PostHog analytics into the MoradBot API (Cloudflare Workers / Hono). The project uses `posthog-node` (v4.8.1) configured for the serverless edge environment with `flushAt: 1` and `flushInterval: 0` so events are dispatched immediately on each Worker invocation. This second run extended coverage to the remaining two untracked routes — `tickets.ts` and `stats.ts` — completing full analytics coverage across all 16 API endpoints.

**Files modified in this run:**

- `apps/api/src/routes/tickets.ts` — added `ticket_list_viewed` and `ticket_detail_viewed` events
- `apps/api/src/routes/stats.ts` — added `stats_dashboard_viewed` and `usage_stats_viewed` events
- `.env` — added `POSTHOG_API_KEY` and `POSTHOG_HOST` for local development

## Events

| Event | Description | File |
|-------|-------------|------|
| `oauth_flow_started` | Merchant initiates Salla OAuth — top of the activation funnel | `apps/api/src/routes/auth.ts` |
| `store_oauth_failed` | Salla OAuth code exchange failed — error monitoring | `apps/api/src/routes/auth.ts` |
| `store_oauth_completed` | Salla OAuth flow completed successfully | `apps/api/src/routes/auth.ts` |
| `store_token_refreshed` | Salla access token refreshed — ongoing engagement signal | `apps/api/src/routes/auth.ts` |
| `chat_message_received` | Visitor sent a message to the bot — core usage metric | `apps/api/src/routes/chat.ts` |
| `chat_escalated` | Conversation escalated to human — FAQ gap / churn indicator | `apps/api/src/routes/chat.ts` |
| `faq_entry_created` | Merchant created an FAQ entry — content setup progress | `apps/api/src/routes/faq.ts` |
| `faq_entry_updated` | Merchant updated an FAQ entry — content maintenance | `apps/api/src/routes/faq.ts` |
| `faq_entry_deleted` | Merchant soft-deleted an FAQ entry — content dissatisfaction signal | `apps/api/src/routes/faq.ts` |
| `escalation_status_updated` | Escalation status changed (pending/in_progress/resolved) | `apps/api/src/routes/escalations.ts` |
| `escalation_resolved` | Escalation marked resolved — support workflow completion | `apps/api/src/routes/escalations.ts` |
| `ticket_list_viewed` | Merchant viewed the ticket list — dashboard engagement signal | `apps/api/src/routes/tickets.ts` |
| `ticket_detail_viewed` | Merchant opened a specific ticket — support workflow depth indicator | `apps/api/src/routes/tickets.ts` |
| `stats_dashboard_viewed` | Merchant viewed the main stats dashboard — retention indicator | `apps/api/src/routes/stats.ts` |
| `usage_stats_viewed` | Merchant checked detailed usage stats — plan awareness / upgrade signal | `apps/api/src/routes/stats.ts` |
| `captureException` | Unhandled server errors (5xx) captured for error tracking | `apps/api/src/middleware/error-handler.ts` |

## Next steps

We've built insights and a dashboard to track user behavior across all the instrumented events:

- **Dashboard — Analytics basics:** https://us.posthog.com/project/368154/dashboard/1429363
- **Merchant Activation Funnel** (oauth_flow_started → store_oauth_completed): https://us.posthog.com/project/368154/insights/nZmv7N8n
- **Daily Chat Volume** (chat_message_received over time): https://us.posthog.com/project/368154/insights/AvK5SMOZ
- **Chat Escalation Rate** (chat_message_received → chat_escalated): https://us.posthog.com/project/368154/insights/bJaUB1cV
- **FAQ Management Activity** (created / updated / deleted trends): https://us.posthog.com/project/368154/insights/Ho2UtvF2
- **Escalation Resolution Funnel** (escalation_status_updated → escalation_resolved): https://us.posthog.com/project/368154/insights/7xlyzam3
- **Merchant Dashboard Engagement** (stats_dashboard_viewed + ticket_list_viewed + usage_stats_viewed): https://us.posthog.com/project/368154/insights/n38VYbo1
- **Ticket Drill-Down Rate** (ticket_list_viewed → ticket_detail_viewed): https://us.posthog.com/project/368154/insights/ywp69KHQ

### Agent skill

We've left an agent skill folder in your project at `.claude/skills/integration-javascript_node/`. You can use this context for further agent development when using Claude Code. This will help ensure the model provides the most up-to-date approaches for integrating PostHog.

</wizard-report>
