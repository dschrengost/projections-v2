# Prefect Migration

This directory documents the migration from systemd timers to Prefect for workflow orchestration.

## Overview

We are incrementally migrating scheduled jobs from systemd timers to a self-hosted Prefect server, following the plan outlined in [`/docs/audit/MIGRATION_PLAN.md`](../audit/MIGRATION_PLAN.md).

## Quick Links

- **Prefect UI**: http://100.78.180.34:4200 (Tailscale)
- **Prefect Flows**: [`/prefect_flows/`](/prefect_flows/)
- **Deployment Config**: [`/prefect.yaml`](/prefect.yaml)
- **Manifests**: `/home/daniel/projections-data/manifests/`

## Current Status

| Job | Prefect | Schedule | systemd | Status |
|-----|---------|----------|---------|--------|
| live-score | ✅ `live-score/live-score` | `*/5 8-23 * * *` ET | ❌ disabled | **Migrated** |
| dk-salaries | ✅ `dk-salaries/dk-salaries` | `0 8 * * *` ET | ❌ disabled | **Migrated** |
| live-scrape | ✅ `live-scrape/live-scrape` | (manual only) | ❌ disabled | Integrated into live-score |

## Contents

- [SETUP.md](./SETUP.md) - How Prefect is configured
- [RUNBOOK.md](./RUNBOOK.md) - Common operations and troubleshooting
