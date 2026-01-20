#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# deploy_live.sh - Sync DEV → PROD for projections-v2
# ==============================================================================
# Usage: ./scripts/deploy/deploy_live.sh [--dry-run]
#
# This script syncs the development checkout to the production directory.
# Prefect runs exclusively from PROD; this is the only sanctioned way to
# update production code.
# ==============================================================================

DEV_REPO="/home/daniel/projects/projections-v2"
PROD_REPO="/home/daniel/prod/projections-v2"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[deploy] DRY RUN - no changes will be made"
fi

# --- Validation ---
if [[ ! -d "$DEV_REPO/.git" ]]; then
    echo "[deploy] ERROR: DEV_REPO is not a git repository: $DEV_REPO" >&2
    exit 1
fi

# Get git info from DEV before sync
DEV_SHA=$(git -C "$DEV_REPO" rev-parse --short HEAD 2>/dev/null || echo "unknown")
DEV_BRANCH=$(git -C "$DEV_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
DEV_DIRTY=$(git -C "$DEV_REPO" status --porcelain 2>/dev/null | head -c1)

if [[ -n "$DEV_DIRTY" ]]; then
    echo "[deploy] WARNING: DEV tree is dirty - uncommitted changes will be deployed"
    echo "[deploy] Consider committing first: git add -A && git commit -m 'wip'"
fi

# Create PROD directory if needed
mkdir -p "$PROD_REPO"

# --- Sync ---
echo "[deploy] Syncing $DEV_REPO → $PROD_REPO"
echo "[deploy] Source: $DEV_BRANCH @ $DEV_SHA"

rsync -av --delete $DRY_RUN \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.venv-user' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='.uv-cache' \
    --exclude='.mypy_cache' \
    --exclude='*.egg-info' \
    --exclude='node_modules' \
    --exclude='web/minutes-dashboard/dist' \
    --exclude='mlruns' \
    --exclude='runs' \
    --exclude='scratch' \
    --exclude='nohup.out' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    "$DEV_REPO/" "$PROD_REPO/"

if [[ -n "$DRY_RUN" ]]; then
    echo "[deploy] DRY RUN complete - no changes made"
    exit 0
fi

# --- Post-sync: deps ---
echo "[deploy] Running uv sync --frozen in PROD..."
cd "$PROD_REPO"
/home/daniel/.local/bin/uv sync --frozen

# --- Write deploy marker ---
DEPLOY_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$PROD_REPO/.deploy_info" <<EOF
{
  "deployed_at": "$DEPLOY_TS",
  "source_sha": "$DEV_SHA",
  "source_branch": "$DEV_BRANCH",
  "source_dirty": $([ -n "$DEV_DIRTY" ] && echo "true" || echo "false"),
  "source_repo": "$DEV_REPO"
}
EOF

# --- Print runtime stamp from PROD ---
echo ""
echo "[deploy] ============================================================"
echo "[deploy] DEPLOY COMPLETE"
echo "[deploy] ============================================================"
echo "[deploy] Deployed at: $DEPLOY_TS"
echo "[deploy] Source:      $DEV_BRANCH @ $DEV_SHA$([ -n "$DEV_DIRTY" ] && echo ' (dirty)')"
echo "[deploy] Target:      $PROD_REPO"
echo "[deploy] ============================================================"
echo ""
echo "[deploy] Runtime stamp from PROD:"
cd "$PROD_REPO"
/home/daniel/.local/bin/uv run python -c "
from projections.runtime_stamp import collect_runtime_stamp
stamp = collect_runtime_stamp(entrypoint='deploy')
print(stamp.to_pretty_block())
"

echo ""
echo "[deploy] SUCCESS - Prefect will now run from: $PROD_REPO"
