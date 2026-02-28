# MoradBot

> AI-Powered FAQ Automation for Salla Merchants

MoradBot is a B2B SaaS platform that provides 24/7 Arabic FAQ automation for Saudi e-commerce stores on the Salla platform.

## Project Status

**Phase:** Phase 4 (Salla Client) — starting
**Version:** 0.3.0
**Target Market:** Saudi Salla merchants (30-300 orders/month)

| Phase | Status | Description |
| ----- | ------ | ----------- |
| Phase 1 | ✅ Complete | Dev environment, Turborepo, Biome, base packages |
| Phase 2 | ✅ Complete | 12-table DB schema, 5 migrations, RLS policies, TypeScript types |
| Phase 3 | ✅ Complete | Hono API: 16 endpoints, middleware stack, error hierarchy, Supabase clients |
| Phase 4 | 🔜 Next | Salla Client Package (OAuth + `GET /products`) |
| Phase 5+ | ⏳ Pending | AI Orchestrator, Widget UI, Dashboard UI |

## Tech Stack

- **Language:** TypeScript
- **Monorepo:** Turborepo + pnpm
- **API:** Cloudflare Workers + Hono
- **Widget:** Preact + Vite
- **Dashboard:** Next.js 15 + React 19
- **Database:** Supabase (PostgreSQL)
- **AI Models:** OpenRouter (Gemini 2.0 Flash, GPT-4 Mini, Claude 3.5 Sonnet)
- **Integration:** Salla OAuth (Read-only Products API)
- **Code Quality:** Biome (linting + formatting)

## Project Structure

```
moradbot/
├── apps/
│   ├── api/          # Cloudflare Workers backend
│   ├── widget/       # Customer-facing chat widget (Preact)
│   └── dashboard/    # Merchant admin panel (Next.js)
├── packages/
│   ├── shared/           # Shared types and utilities
│   ├── ai-orchestrator/  # AI response orchestration
│   └── salla-client/     # Salla API client
├── supabase/
│   └── migrations/   # Database migrations
└── docs/             # Product documentation
```

## MVP Scope

**Included:**
- ✅ FAQ automation (5 questions: availability, price, shipping, payment, returns)
- ✅ Arabic text-only chat widget
- ✅ Periodic product sync (24h/6h/1h based on plan)
- ✅ Escalation workflow (3 clarification attempts)
- ✅ Merchant dashboard (view conversations, basic stats)
- ✅ Read-only Salla integration

**Excluded from MVP:**
- ❌ Order tracking
- ❌ WhatsApp integration
- ❌ Multi-language support
- ❌ Sales/upselling features
- ❌ Write permissions to Salla data

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- pnpm >= 8.0.0

### Installation

```bash
# Install dependencies
pnpm install

# Run development servers
pnpm dev

# Build all packages
pnpm build

# Run linting
pnpm lint

# Format code
pnpm format
```

### Development Workflow

Each workspace can be developed independently:

```bash
# API development
cd apps/api
pnpm dev

# Widget development
cd apps/widget
pnpm dev

# Dashboard development
cd apps/dashboard
pnpm dev
```

## Architecture Principles

1. **Zero-tolerance multi-tenant isolation** - No cross-store data leakage
2. **Read-only Salla integration** - No write permissions
3. **Single-agent MVP** - Designed to evolve to multi-agent
4. **Strict scope discipline** - No feature creep beyond MVP

## Documentation

Comprehensive product documentation is available in the `docs_v2/` folder (corrected, MVP-only):

- **MRD** (`mrd_v2.md`) - Market Requirements Document
- **BRD** (`brd_v2.md`) - Business Requirements Document
- **PRD** (`prd_v2.md`) - Product Requirements Document
- **SRD** (`srd_v2.md`) - System Requirements Document
- **Extended Architecture** (`extended_architecture_v2.md`) - Operational decisions
- **Marketing Strategy** (`marketing_strategy_moradbot.md`) - Pricing tiers, go-to-market plan
- **Tools & Costs** (`tools_and_costs.md`) - Infrastructure costs and break-even analysis

Claude-generated session docs and phase plans: `docs/claude/`

## Development

This project uses:
- **Turborepo** for build orchestration
- **pnpm workspaces** for dependency management
- **Biome** for linting and formatting
- **TypeScript** with strict mode enabled

## License

UNLICENSED - Proprietary software

## Author

Mohammed Aljohani
