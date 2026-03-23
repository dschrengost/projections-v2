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
RESTART_DASHBOARD=1

DASHBOARD_SERVICE="minutes-dashboard.service"
DASHBOARD_PORT="8501"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
        --sync-pointers)
            SYNC_POINTERS=1
            ;;
        --skip-dashboard-restart)
            RESTART_DASHBOARD=0
            ;;
        *)
            echo "[deploy] ERROR: unknown argument: $1" >&2
            echo "Usage: ./scripts/deploy/deploy_live.sh [--dry-run] [--sync-pointers] [--skip-dashboard-restart]" >&2
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
    echo "         - config/ownership_current_run.json"
fi

POINTER_EXCLUDES=()
if [[ "$SYNC_POINTERS" -eq 0 ]]; then
    POINTER_EXCLUDES+=(
        "--exclude=config/minutes_current_run.json"
        "--exclude=config/rates_current_run.json"
        "--exclude=config/ownership_current_run.json"
    )
fi

rsync -av --delete $DRY_RUN \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.venv311' \
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
    --exclude='/artifacts' \
    --exclude='/reports' \
    --exclude='/mlruns' \
    --exclude='/runs' \
    --exclude='/scratch' \
    --exclude='nohup.out' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    --exclude='.serena' \
    --exclude='Running' \
    "${POINTER_EXCLUDES[@]}" \
    "$DEV_REPO/" "$PROD_REPO/"

if [[ -n "$DRY_RUN" ]]; then
    echo "[deploy] DRY RUN complete - no changes made"
    exit 0
fi

_listener_pid_for_port() {
    local port="$1"
    ss -ltnpH "( sport = :${port} )" 2>/dev/null \
        | sed -nE 's/.*pid=([0-9]+).*/\1/p' \
        | head -n 1
}

_restart_dashboard_service_clean() {
    local service="$1"
    local port="$2"

    if ! systemctl --user cat "$service" >/dev/null 2>&1; then
        echo "[deploy] WARNING: $service not found in user systemd."
        if [[ -x "$DEV_REPO/scripts/deploy/install_minutes_dashboard_unit.sh" ]] && [[ "$service" == "$DASHBOARD_SERVICE" ]]; then
            echo "[deploy] Installing $service from repo template..."
            "$DEV_REPO/scripts/deploy/install_minutes_dashboard_unit.sh" >/dev/null
        else
            echo "[deploy] Skipping dashboard restart."
            return 0
        fi
    fi

    echo "[deploy] Restarting $service with orphan cleanup (port ${port})..."
    systemctl --user stop "$service" || true

    # Kill stray uvicorn processes that can survive outside the service cgroup.
    pkill -f "projections.api.minutes_api:create_app.*--port ${port}" || true
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi

    local listener_pid=""
    for _ in {1..30}; do
        listener_pid="$(_listener_pid_for_port "$port")"
        if [[ -z "$listener_pid" ]]; then
            break
        fi
        sleep 0.2
    done

    if [[ -n "$listener_pid" ]]; then
        echo "[deploy] ERROR: port ${port} still in use by PID ${listener_pid} after cleanup."
        ss -ltnpH "( sport = :${port} )" || true
        return 1
    fi

    systemctl --user reset-failed "$service" >/dev/null 2>&1 || true
    systemctl --user start "$service"

    local main_pid=""
    local healthy=0
    for _ in {1..40}; do
        main_pid="$(systemctl --user show -p MainPID --value "$service" 2>/dev/null || true)"
        listener_pid="$(_listener_pid_for_port "$port")"
        if systemctl --user is-active --quiet "$service" \
            && [[ -n "$main_pid" && "$main_pid" != "0" ]] \
            && [[ "$listener_pid" == "$main_pid" ]]; then
            healthy=1
            break
        fi
        sleep 0.25
    done

    if [[ "$healthy" -ne 1 ]]; then
        echo "[deploy] ERROR: $service failed to come up cleanly."
        systemctl --user status "$service" --no-pager -l | sed -n '1,30p' || true
        ss -ltnpH "( sport = :${port} )" || true
        return 1
    fi

    if command -v curl >/dev/null 2>&1; then
        if ! curl -fsS "http://127.0.0.1:${port}/api/version" >/dev/null; then
            echo "[deploy] ERROR: health check failed for /api/version on port ${port}."
            return 1
        fi
    fi

    echo "[deploy] $service healthy (pid=${main_pid}, port=${port})"
    return 0
}

# --- Runtime selector sync ---
DATA_ROOT="${PROJECTIONS_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
RUNTIME_SELECTOR_DIR="$DATA_ROOT/control_plane/model_selectors"
mkdir -p "$RUNTIME_SELECTOR_DIR"

sync_selector() {
    local selector_file="$1"
    local src_prod="$PROD_REPO/config/$selector_file"
    local src_dev="$DEV_REPO/config/$selector_file"
    local src="$src_prod"
    local dst="$RUNTIME_SELECTOR_DIR/$selector_file"

    # If we preserve PROD pointers, the config files may be excluded from rsync.
    # Seed from DEV only when the PROD file is missing (fresh bootstrap scenario).
    if [[ ! -f "$src_prod" ]]; then
        if [[ -f "$src_dev" ]]; then
            src="$src_dev"
            echo "[deploy] NOTE: prod selector missing; seeding from DEV: $src_dev"
        else
            echo "[deploy] WARNING: selector source missing, skipping: $src_prod"
            return
        fi
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
sync_selector "ownership_current_run.json"

# --- Post-sync: deps ---
echo "[deploy] Running uv sync --frozen in PROD..."
cd "$PROD_REPO"
/home/daniel/.local/bin/uv sync --frozen

# --- Post-sync: frontend build ---
FRONTEND_DIR="$PROD_REPO/web/minutes-dashboard"

_run_frontend_install() {
    local root_dir="$1"
    local cache_dir="$2"
    local had_lock=0
    local install_failed=0
    local -a install_cmd
    local -a cache_env=("--cache" "$cache_dir" "--prefer-offline")

    cd "$root_dir"
    if [[ -f package-lock.json ]]; then
        install_cmd=(npm ci)
        had_lock=1
    else
        install_cmd=(npm install)
    fi

    if [[ -n "${cache_dir}" ]]; then
        export NPM_CONFIG_CACHE="$cache_dir"
    else
        unset NPM_CONFIG_CACHE
    fi

    if [[ $had_lock -eq 1 ]]; then
        "${install_cmd[@]}" "${cache_env[@]}" --no-audit --no-fund || install_failed=$?
        if [[ $install_failed -ne 0 ]]; then
            npm install --no-audit --no-fund "${cache_env[@]}" || install_failed=$?
        fi
    else
        "${install_cmd[@]}" --no-audit --no-fund "${cache_env[@]}" || install_failed=$?
    fi

    return "$install_failed"
}

_run_frontend_build() {
    local root_dir="$1"
    local attempt="$2"
    local build_failed=0

    cd "$root_dir"
    NODE_OPTIONS="--max_old_space_size=4096" \
        npm run build >"/tmp/projections-deploy-prod-build-${attempt}.log" 2>&1 || build_failed=$?

    return "$build_failed"
}

_copy_or_build_frontend_from_dev() {
    local prod_dir="$1"
    local dev_dir="$2"

    if [[ ! -d "$dev_dir/dist" ]]; then
        return 1
    fi

    if [[ "$dev_dir/dist" -nt "$prod_dir/dist" ]]; then
        rsync -a --delete "$dev_dir/dist/" "$prod_dir/dist/"
        return $?
    fi

    rsync -a --delete "$dev_dir/dist/" "$prod_dir/dist/"
    return $?
}

if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    echo "[deploy] Building frontend in PROD..."
    cd "$PROD_REPO"

    NPM_INSTALL_OK=0
    for attempt in 1 2 3; do
        echo "[deploy] Installing frontend deps (attempt ${attempt}/3)..."
        TMP_NPM_CACHE=""
        TMP_NPM_CACHE=$(mktemp -d)
        if _run_frontend_install "$FRONTEND_DIR" "$TMP_NPM_CACHE"; then
            NPM_INSTALL_EXIT=0
        else
            NPM_INSTALL_EXIT=$?
        fi
        rm -rf "$TMP_NPM_CACHE"
        if [[ $NPM_INSTALL_EXIT -eq 0 ]]; then
            NPM_INSTALL_OK=1
            break
        fi
        echo "[deploy] Frontend install failed with exit code $NPM_INSTALL_EXIT"
        if [[ $attempt -lt 3 ]]; then
            echo "[deploy] Retrying frontend install with clean cache..."
        fi
    done

    if [[ $NPM_INSTALL_OK -ne 1 ]]; then
        echo "[deploy] WARNING: frontend dependency install failed in PROD after retries."
        echo "[deploy] Attempting fallback: copying a built dist from DEV if available."
        cd "$DEV_REPO/web/minutes-dashboard"
        DEV_NPM_LOG="/tmp/projections-deploy-dev-npm.log"
        DEV_BUILD_LOG="/tmp/projections-deploy-dev-build.log"
        if [[ -f package-lock.json ]]; then
            npm ci --no-audit --no-fund >"$DEV_NPM_LOG" 2>&1 || npm install --no-audit --no-fund >"$DEV_NPM_LOG" 2>&1
        else
            npm install --no-audit --no-fund >"$DEV_NPM_LOG" 2>&1
        fi
        if ! npm run build >"$DEV_BUILD_LOG" 2>&1; then
            echo "[deploy] WARNING: DEV frontend build failed, falling back to existing DEV dist.\n   logs:"
            echo "   npm: $DEV_NPM_LOG"
            echo "   build: $DEV_BUILD_LOG"
        fi

        if _copy_or_build_frontend_from_dev "$FRONTEND_DIR" "$DEV_REPO/web/minutes-dashboard"; then
            echo "[deploy] Reused DEV frontend dist for PROD fallback."
            cd "$PROD_REPO"
        else
            echo "[deploy] ERROR: fallback frontend copy failed. Check logs in /tmp/projections-deploy-dev-npm-*.log"
            exit 1
        fi
    else
        BUILD_OK=0
        for attempt in 1 2 3; do
            if _run_frontend_build "$FRONTEND_DIR" "$attempt"; then
                BUILD_OK=1
                break
            fi
            echo "[deploy] Frontend build failed on attempt $attempt."
            if [[ $attempt -lt 3 ]]; then
                echo "[deploy] Frontend build retrying (attempt $((attempt + 1))/3)..."
            fi
        done

        if [[ $BUILD_OK -ne 1 ]]; then
            echo "[deploy] WARNING: frontend build failed in PROD after retries."
            echo "[deploy] Attempting fallback: copying a built dist from DEV if available."
            cd "$DEV_REPO/web/minutes-dashboard"
            DEV_BUILD_LOG="/tmp/projections-deploy-dev-build.log"
            DEV_NPM_LOG="/tmp/projections-deploy-dev-npm.log"
            if [[ -f package-lock.json ]]; then
                npm ci --no-audit --no-fund >"$DEV_NPM_LOG" 2>&1 || npm install --no-audit --no-fund >"$DEV_NPM_LOG" 2>&1
            else
                npm install --no-audit --no-fund >"$DEV_NPM_LOG" 2>&1
            fi
            NODE_OPTIONS="--max_old_space_size=4096" npm run build >"$DEV_BUILD_LOG" 2>&1 || BUILD_OK=0

            if _copy_or_build_frontend_from_dev "$FRONTEND_DIR" "$DEV_REPO/web/minutes-dashboard"; then
                echo "[deploy] Reused DEV frontend dist for PROD fallback."
                BUILD_OK=1
            else
                echo "[deploy] ERROR: fallback frontend copy failed."
                echo "   npm: $DEV_NPM_LOG"
                echo "   build: $DEV_BUILD_LOG"
                echo "   prod build logs:"
                for failed_attempt in 1 2 3; do
                    echo "   /tmp/projections-deploy-prod-build-${failed_attempt}.log"
                done
                exit 1
            fi
        fi

        if [[ $BUILD_OK -ne 1 ]]; then
            echo "[deploy] ERROR: unable to produce a frontend dist for PROD."
            exit 1
        fi
    fi
    cd "$PROD_REPO"
else
    echo "[deploy] WARNING: frontend package.json not found at $FRONTEND_DIR"
fi

# --- Write deploy marker (before restarts so services report correct stamp) ---
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

if [[ "$RESTART_DASHBOARD" -eq 1 ]]; then
    _restart_dashboard_service_clean "$DASHBOARD_SERVICE" "$DASHBOARD_PORT"
else
    echo "[deploy] Skipping dashboard restart (--skip-dashboard-restart)"
fi

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
