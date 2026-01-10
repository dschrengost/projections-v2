from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from projections.paths import get_data_root, get_project_root

_DRAFTABLE_ID_RE = re.compile(r"\((\d+)\)")

EVAL_STATUS_FILENAME = "eval_status.json"
EVAL_LOG_FILENAME = "eval.log"
EVAL_LOCK_FILENAME = "eval.lock"
GLOBAL_EVAL_LOCK_FILENAME = ".eval_global.lock"

# If a job stays RUNNING longer than this and its PID is not alive, mark it FAILED.
DEFAULT_STALE_SECONDS = 30 * 60
# If the global lock exists, wait up to this long before giving up.
DEFAULT_GLOBAL_LOCK_WAIT_SECONDS = 30 * 60
# Consider a global lock stale if older than this and PID is not alive.
DEFAULT_GLOBAL_LOCK_STALE_SECONDS = 30 * 60


def _utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def _utc_iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    return ts.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _try_acquire_lock(lock_path: Path, *, payload: dict[str, Any] | None = None) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        if payload is None:
            payload = {"pid": os.getpid(), "created_at": _utc_iso(_utc_now())}
        os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
        return True
    except FileExistsError:
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _extract_draftable_id(value: str) -> int | None:
    if not value:
        return None
    match = _DRAFTABLE_ID_RE.search(value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _resolve_eval_dirs_from_manifest_path(manifest_path: Path, export_id: str) -> tuple[Path, Path]:
    exports_dir = manifest_path.parent
    contest_dir = exports_dir.parent
    eval_dir = contest_dir / "eval_pre" / f"export_{export_id}"
    status_path = eval_dir / EVAL_STATUS_FILENAME
    return eval_dir, status_path


def _ensure_status(
    *,
    status_path: Path,
    export_id: str,
    report_dir: Path,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    current = _read_json(status_path)
    if isinstance(current, dict) and current.get("export_id") == export_id:
        return current
    payload = {
        "status": "PENDING",
        "export_id": export_id,
        "created_at": _utc_iso(created_at or _utc_now()),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "return_code": None,
        "report_dir": str(report_dir.resolve()),
        "error_message": None,
        "warnings": [],
    }
    _write_json_atomic(status_path, payload)
    return payload


def _maybe_mark_stale_running(status_path: Path, *, stale_seconds: int = DEFAULT_STALE_SECONDS) -> dict[str, Any]:
    status = _read_json(status_path) or {}
    if not isinstance(status, dict):
        return {}
    if status.get("status") != "RUNNING":
        return status

    started_at_raw = status.get("started_at")
    pid_raw = status.get("pid")
    try:
        started_at = datetime.fromisoformat(str(started_at_raw).replace("Z", "+00:00"))
    except Exception:
        return status
    try:
        pid = int(pid_raw) if pid_raw is not None else 0
    except Exception:
        pid = 0

    age = _utc_now() - started_at.astimezone(UTC)
    if age.total_seconds() <= float(stale_seconds):
        return status
    if pid and _pid_is_running(pid):
        # Long-running; keep status RUNNING but add a warning for UI visibility.
        warnings = status.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
        msg = f"running longer than {int(stale_seconds)}s"
        if msg not in warnings:
            warnings.append(msg)
            status["warnings"] = warnings
            _write_json_atomic(status_path, status)
        return status

    status["status"] = "FAILED"
    status["finished_at"] = _utc_iso(_utc_now())
    status["return_code"] = None
    status["error_message"] = "stale job (server restart?)"
    _write_json_atomic(status_path, status)
    return status


def _resolve_base_worlds_matrix_path(
    *,
    game_date: str,
    base_worlds_path: str | None,
    base_worlds_run_id: str | None,
    data_root: Path,
) -> Path:
    if base_worlds_path:
        p = Path(base_worlds_path).expanduser().resolve()
        if p.is_dir():
            candidate = p / "worlds_matrix.parquet"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Missing worlds_matrix.parquet under {p}")
        if p.exists():
            return p
        raise FileNotFoundError(f"Base worlds path not found: {p}")

    base_dir = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"
    if not base_dir.exists():
        raise FileNotFoundError(f"No sim_v2 worlds directory found for {game_date}: {base_dir}")

    def _pick_run_dir(root_dir: Path, run_id: str | None) -> Path:
        if run_id:
            candidate = root_dir / f"run={run_id}"
            if candidate.exists():
                return candidate

        pointer = root_dir / "latest_run.json"
        if pointer.exists():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            latest = payload.get("run_id") if isinstance(payload, dict) else None
            if latest:
                candidate = root_dir / f"run={latest}"
                if candidate.exists():
                    return candidate

        run_dirs = sorted(
            [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
            reverse=True,
        )
        if run_dirs:
            return run_dirs[0]
        return root_dir

    run_dir = _pick_run_dir(base_dir, base_worlds_run_id)
    matrix_path = run_dir / "worlds_matrix.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing worlds_matrix.parquet at {matrix_path}")
    return matrix_path


def _convert_export_csv_to_eval_lineups(
    *,
    export_csv_path: Path,
    eval_lineups_path: Path,
    game_date: str,
    draft_group_id: int,
    site: str,
) -> dict[str, Any]:
    if site.lower() != "dk":
        raise ValueError(f"Only dk export conversion is supported (site={site!r})")

    # Local imports to avoid pulling FastAPI router modules during import.
    from projections.api.optimizer_api import DK_NBA_SLOTS, _load_dk_nba_draftable_ids_by_player
    from projections.api.optimizer_service import build_player_pool

    pool = build_player_pool(game_date=game_date, draft_group_id=int(draft_group_id), site="dk")
    internal_to_dk: dict[str, int] = {}
    for p in pool:
        pid = str(p.get("player_id") or "").strip()
        if not pid:
            continue
        dk_id_raw = p.get("dk_id")
        if dk_id_raw is None or dk_id_raw == "":
            continue
        try:
            internal_to_dk[pid] = int(dk_id_raw)
        except Exception:
            continue

    dk_to_internal = {dk_id: pid for pid, dk_id in internal_to_dk.items()}
    draftable_ids_by_player, _dk_names = _load_dk_nba_draftable_ids_by_player(int(draft_group_id))

    draftable_to_internal: dict[int, str] = {}
    for dk_id, slot_map in draftable_ids_by_player.items():
        internal_id = dk_to_internal.get(int(dk_id))
        if not internal_id:
            continue
        for draftable_id in slot_map.values():
            if not draftable_id:
                continue
            draftable_to_internal.setdefault(int(draftable_id), internal_id)

    rows_out: list[dict[str, Any]] = []
    errors: list[str] = []
    with export_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            entry_id = str(row.get("Entry ID") or "").strip()
            contest_id = str(row.get("Contest ID") or "").strip()

            internal_ids: list[str] = []
            missing_slots: list[str] = []
            for slot in DK_NBA_SLOTS:
                raw = str(row.get(slot) or "").strip()
                did = _extract_draftable_id(raw)
                if did is None:
                    missing_slots.append(slot)
                    continue
                internal = draftable_to_internal.get(int(did))
                if not internal:
                    missing_slots.append(slot)
                    continue
                internal_ids.append(str(internal))

            if missing_slots:
                errors.append(
                    f"row={row_idx + 2} entry_id={entry_id or 'n/a'} contest_id={contest_id or 'n/a'} missing_or_unmapped_slots={missing_slots}"
                )
                continue

            out_row: dict[str, Any] = {
                "lineup_id": entry_id or f"row-{row_idx + 1}",
                "entry_id": entry_id,
                "contest_id": contest_id,
                "draft_group_id": int(draft_group_id),
            }
            for i, pid in enumerate(internal_ids, start=1):
                out_row[f"p{i}_id"] = pid
            rows_out.append(out_row)

    eval_lineups_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_lineups_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["lineup_id", "entry_id", "contest_id", "draft_group_id"] + [f"p{i}_id" for i in range(1, 9)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    meta = {
        "export_rows": int(len(rows_out) + len(errors)),
        "eval_lineups_written": int(len(rows_out)),
        "eval_lineups_dropped": int(len(errors)),
        "errors": errors[:50],
        "errors_truncated": len(errors) > 50,
    }
    if errors:
        raise ValueError(f"Failed to convert DK export to eval lineups; first_error={errors[0]}")
    return meta


def run_eval_job_from_manifest(
    *,
    manifest_path: Path,
    global_lock_wait_seconds: int = DEFAULT_GLOBAL_LOCK_WAIT_SECONDS,
    global_lock_stale_seconds: int = DEFAULT_GLOBAL_LOCK_STALE_SECONDS,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest JSON: {manifest_path}")

    export_id = str(manifest.get("export_id") or "").strip()
    if not export_id:
        raise ValueError("Manifest missing export_id")

    eval_dir, status_path = _resolve_eval_dirs_from_manifest_path(manifest_path, export_id)
    report_dir = eval_dir
    status = _ensure_status(status_path=status_path, export_id=export_id, report_dir=report_dir)
    _maybe_mark_stale_running(status_path, stale_seconds=stale_seconds)

    per_export_lock = eval_dir / EVAL_LOCK_FILENAME
    if not _try_acquire_lock(per_export_lock, payload={"pid": os.getpid(), "export_id": export_id, "created_at": _utc_iso(_utc_now())}):
        return 0

    data_root = get_data_root()
    global_lock = data_root / "contests" / GLOBAL_EVAL_LOCK_FILENAME
    acquired_global = False
    try:
        # Wait for global lock to enforce one-at-a-time evaluation.
        started_wait = _utc_now()
        while True:
            acquired_global = _try_acquire_lock(
                global_lock,
                payload={
                    "pid": os.getpid(),
                    "export_id": export_id,
                    "created_at": _utc_iso(_utc_now()),
                },
            )
            if acquired_global:
                break

            # If lock seems stale, clean it up.
            try:
                st = global_lock.stat()
                mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC)
            except Exception:
                mtime = None
            lock_payload = _read_json(global_lock) or {}
            lock_pid = None
            try:
                lock_pid = int(lock_payload.get("pid")) if isinstance(lock_payload, dict) else None
            except Exception:
                lock_pid = None
            age = (_utc_now() - mtime) if mtime is not None else timedelta(seconds=0)
            if mtime is not None and age.total_seconds() > float(global_lock_stale_seconds) and (not lock_pid or not _pid_is_running(lock_pid)):
                _release_lock(global_lock)
                continue

            waited = (_utc_now() - started_wait).total_seconds()
            if waited > float(global_lock_wait_seconds):
                status = _read_json(status_path) or status
                if isinstance(status, dict):
                    status["status"] = "FAILED"
                    status["finished_at"] = _utc_iso(_utc_now())
                    status["error_message"] = "queued too long waiting for global eval lock"
                    _write_json_atomic(status_path, status)
                return 1

            time.sleep(2.0)

        # Convert lineups to eval format (p1_id..p8_id).
        export_csv_path_raw = manifest.get("export_csv_path")
        if not export_csv_path_raw:
            raise ValueError("Manifest missing export_csv_path")
        export_csv_path = Path(str(export_csv_path_raw)).expanduser().resolve()

        site = str(manifest.get("site") or "dk")
        game_date = str(manifest.get("game_date") or "").strip()
        if not game_date:
            raise ValueError("Manifest missing game_date")

        dg_raw = manifest.get("draft_group_id")
        if dg_raw is None:
            raise ValueError("Manifest missing draft_group_id")
        draft_group_id = int(dg_raw)

        eval_lineups_path = eval_dir / "eval_lineups.csv"
        conv_meta = _convert_export_csv_to_eval_lineups(
            export_csv_path=export_csv_path,
            eval_lineups_path=eval_lineups_path,
            game_date=game_date,
            draft_group_id=draft_group_id,
            site=site,
        )
        manifest["eval_lineups_path"] = str(eval_lineups_path.resolve())
        manifest["eval_lineups_meta"] = conv_meta
        _write_json_atomic(manifest_path, manifest)

        base_worlds_matrix_path = _resolve_base_worlds_matrix_path(
            game_date=game_date,
            base_worlds_path=manifest.get("base_worlds_path"),
            base_worlds_run_id=manifest.get("base_worlds_run_id"),
            data_root=data_root,
        )

        seed = int(manifest.get("eval_seed") or 123)
        train_frac = float(manifest.get("train_frac") or 0.7)
        k_runtime = int(manifest.get("k_runtime_holdouts") or 3)
        num_worlds_runtime = int(manifest.get("num_worlds_runtime") or 10000)

        log_path = eval_dir / EVAL_LOG_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-u",
            str(get_project_root() / "scripts" / "lineups" / "eval_portfolio_holdout_runtime.py"),
            "--date",
            game_date,
            "--site",
            site,
            "--lineups-path",
            str(eval_lineups_path),
            "--base-worlds-path",
            str(base_worlds_matrix_path),
            "--train-frac",
            str(train_frac),
            "--seed",
            str(seed),
            "--k-runtime-holdouts",
            str(k_runtime),
            "--num-worlds-runtime",
            str(num_worlds_runtime),
            "--output-root",
            str(report_dir),
            "--data-root",
            str(data_root),
        ]

        # Update status -> RUNNING before spawning.
        status = _read_json(status_path) or status
        if not isinstance(status, dict):
            status = {}
        status.update(
            {
                "status": "RUNNING",
                "started_at": _utc_iso(_utc_now()),
                "finished_at": None,
                "pid": None,
                "return_code": None,
                "error_message": None,
            }
        )
        _write_json_atomic(status_path, status)

        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"[eval_runner] cmd={cmd}\n")
            log_f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(get_project_root()),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status = _read_json(status_path) or status
            if isinstance(status, dict):
                status["pid"] = int(proc.pid)
                _write_json_atomic(status_path, status)
            rc = proc.wait()

        status = _read_json(status_path) or status
        if not isinstance(status, dict):
            status = {}
        status["return_code"] = int(rc)
        status["finished_at"] = _utc_iso(_utc_now())
        if rc == 0:
            status["status"] = "DONE"
        else:
            status["status"] = "FAILED"
            status["error_message"] = f"eval exited with return_code={rc}"
        _write_json_atomic(status_path, status)
        return int(rc)
    except Exception as exc:
        status = _read_json(status_path) or status
        if isinstance(status, dict):
            status["status"] = "FAILED"
            status["finished_at"] = _utc_iso(_utc_now())
            status["error_message"] = str(exc)
            _write_json_atomic(status_path, status)
        return 1
    finally:
        if acquired_global:
            _release_lock(global_lock)
        _release_lock(per_export_lock)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run runtime holdout evaluation job from an export manifest.")
    p.add_argument("--manifest", required=True, help="Path to export_<id>_manifest.json")
    p.add_argument("--global-lock-wait-seconds", type=int, default=DEFAULT_GLOBAL_LOCK_WAIT_SECONDS)
    p.add_argument("--global-lock-stale-seconds", type=int, default=DEFAULT_GLOBAL_LOCK_STALE_SECONDS)
    p.add_argument("--stale-seconds", type=int, default=DEFAULT_STALE_SECONDS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_eval_job_from_manifest(
        manifest_path=Path(args.manifest),
        global_lock_wait_seconds=int(args.global_lock_wait_seconds),
        global_lock_stale_seconds=int(args.global_lock_stale_seconds),
        stale_seconds=int(args.stale_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
