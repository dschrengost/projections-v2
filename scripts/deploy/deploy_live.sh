#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# deploy_live.sh - Sync DEV → PROD for projections-v2
# ==============================================================================
# Usage: ./scripts/deploy/deploy_live.sh [--dry-run] [--sync-pointers]
#
# This script syncs the development checkout to the production directory.
# Prefect runs exclusively from PROD; this is the only sanctioned way to
# update production code.
# ==============================================================================

DEV_REPO="/home/daniel/projects/projections-v2"
PROD_REPO="/home/daniel/prod/projections-v2"
DEFAULT_DATA_ROOT="/home/daniel/projections-data"

DRY_RUN=""
SYNC_POINTERS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
        --sync-pointers)
            SYNC_POINTERS=1
            ;;
        *)
            echo "[deploy] ERROR: unknown argument: $1" >&2
            echo "Usage: ./scripts/deploy/deploy_live.sh [--dry-run] [--sync-pointers]" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ -n "$DRY_RUN" ]]; then
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
if [[ "$SYNC_POINTERS" -eq 0 ]]; then
    echo "[deploy] Preserving PROD selector pointers (use --sync-pointers to overwrite):"
    echo "         - config/minutes_current_run.json"
    echo "         - config/rates_current_run.json"
fi

POINTER_EXCLUDES=()
if [[ "$SYNC_POINTERS" -eq 0 ]]; then
    POINTER_EXCLUDES+=(
        "--exclude=config/minutes_current_run.json"
        "--exclude=config/rates_current_run.json"
    )
fi

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
    "${POINTER_EXCLUDES[@]}" \
    "$DEV_REPO/" "$PROD_REPO/"

if [[ -n "$DRY_RUN" ]]; then
    echo "[deploy] DRY RUN complete - no changes made"
    exit 0
fi

# --- Runtime selector sync ---
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
RUNTIME_SELECTOR_DIR="$DATA_ROOT/control_plane/model_selectors"
mkdir -p "$RUNTIME_SELECTOR_DIR"

sync_selector() {
    local selector_file="$1"
    local src="$PROD_REPO/config/$selector_file"
    local dst="$RUNTIME_SELECTOR_DIR/$selector_file"

    if [[ ! -f "$src" ]]; then
        echo "[deploy] WARNING: selector source missing, skipping: $src"
        return
    fi

    if [[ "$SYNC_POINTERS" -eq 1 ]]; then
        cp "$src" "$dst"
        echo "[deploy] Synced runtime selector: $dst"
        return
    fi

    if [[ ! -f "$dst" ]]; then
        cp "$src" "$dst"
        echo "[deploy] Seeded runtime selector: $dst"
    fi
}

sync_selector "minutes_current_run.json"
sync_selector "rates_current_run.json"

# --- Post-sync: deps ---
echo "[deploy] Running uv sync --frozen in PROD..."
cd "$PROD_REPO"
/home/daniel/.local/bin/uv sync --frozen

# --- Post-sync: frontend build ---
FRONTEND_DIR="$PROD_REPO/web/minutes-dashboard"
if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    echo "[deploy] Building frontend in PROD..."
    cd "$FRONTEND_DIR"
    if [[ ! -d node_modules ]]; then
        echo "[deploy] Installing frontend deps with npm ci..."
        npm ci
    fi
    npm run build
    cd "$PROD_REPO"
else
    echo "[deploy] WARNING: frontend package.json not found at $FRONTEND_DIR"
fi

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
