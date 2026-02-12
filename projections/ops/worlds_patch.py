from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, date as date_cls, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from projections import paths
from projections.artifacts.unified_projections import load_summary, resolve_unified_run_dir
from projections.cli.finalize_projections import finalize_projections
from projections.pipeline.status import JobStatus, write_status

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _generate_run_id(data_root: Path, *, slate_day: date_cls) -> str:
    """Generate a unique run_id in canonical format (YYYYMMDDTHHMMSSZ)."""
    base_dir = data_root / "artifacts" / "projections" / slate_day.isoformat()
    ts0 = datetime.now(tz=UTC).replace(microsecond=0)
    for offset in range(0, 120):
        ts = ts0 + timedelta(seconds=offset)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        if not (base_dir / f"run={run_id}").exists():
            return run_id
    # Fallback: include minutes if collisions are extreme
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _patch_sim_projections(
    base: pd.DataFrame,
    patch: pd.DataFrame,
    *,
    key_cols: tuple[str, ...],
) -> pd.DataFrame:
    if base.empty:
        return patch.copy()
    if patch.empty:
        return base

    base_work = base.copy()
    patch_work = patch.copy()

    key_tmp = []
    for key in key_cols:
        if key not in base_work.columns or key not in patch_work.columns:
            raise ValueError(f"Missing key column {key!r} for sim projections patch.")
        tmp = f"_ops_{key}"
        base_work[tmp] = base_work[key].astype(str)
        patch_work[tmp] = patch_work[key].astype(str)
        key_tmp.append(tmp)

    patch_cols = [c for c in patch_work.columns if c not in key_cols and c not in key_tmp]

    patch_keys = patch_work[key_tmp].drop_duplicates().copy()
    base_keys = base_work[key_tmp].drop_duplicates().copy()

    patch_key_tuples = set(map(tuple, patch_keys.to_numpy()))
    base_key_tuples = set(map(tuple, base_keys.to_numpy()))
    new_key_tuples = patch_key_tuples - base_key_tuples

    for col in patch_cols:
        if col not in base_work.columns:
            base_work[col] = pd.NA

    idx = base_work.set_index(key_tmp).index
    patch_idx = patch_work.set_index(key_tmp)

    for col in patch_cols:
        s = patch_idx[col]
        base_work[col] = base_work[col].where(~idx.isin(s.index), idx.map(s))

    if new_key_tuples:
        new_mask = patch_work[key_tmp].apply(tuple, axis=1).isin(new_key_tuples)
        append = patch_work.loc[new_mask].copy()
        for col in base_work.columns:
            if col not in append.columns:
                append[col] = pd.NA
        append = append[base_work.columns]
        base_work = pd.concat([base_work, append], ignore_index=True)

    return base_work.drop(columns=key_tmp, errors="ignore")


def patch_worlds_matrix_for_game(
    *,
    game_date: date_cls,
    game_id: int,
    base_projections_run_id: str | None = None,
    data_root: Path | None = None,
    pin_projections_run: bool = False,
    minutes_override_mode: str = "legacy",
    override_infeasible: str = "error",
    status_run_ts: str | None = None,
) -> dict[str, Any]:
    """Re-run sim for a single game and patch the slate worlds matrix + projections."""
    root = data_root or paths.data_path()
    run_ts = status_run_ts or _utc_now_iso()
    minutes_override_mode_eff = str(minutes_override_mode).strip().lower()
    if minutes_override_mode_eff not in {"legacy", "v2"}:
        raise ValueError(
            f"minutes_override_mode must be one of legacy|v2, got {minutes_override_mode!r}"
        )
    override_infeasible_eff = str(override_infeasible).strip().lower()
    if override_infeasible_eff not in {"error", "relax", "ignore"}:
        raise ValueError(
            f"override_infeasible must be one of error|relax|ignore, got {override_infeasible!r}"
        )

    write_status(
        JobStatus(
            job_name="ops_patch_worlds_matrix_game",
            stage="ops",
            target_date=game_date.isoformat(),
            run_ts=run_ts,
            status="running",
            rows_written=0,
            expected_rows=None,
            message=f"patching game_id={game_id}",
        )
    )

    try:
        projections_run_dir, ctx = resolve_unified_run_dir(root, game_date, run_id=base_projections_run_id)
        if projections_run_dir is None:
            raise FileNotFoundError(f"No unified projections found for {game_date.isoformat()}.")

        summary = load_summary(projections_run_dir) or {}

        minutes_run_id = summary.get("minutes_run_id")
        rates_run_id = summary.get("rates_run_id")
        ownership_run_id = summary.get("ownership_run_id") or ctx.resolved_run_id
        draft_group_id = summary.get("draft_group_id")
        sim_profile = summary.get("sim_profile") or "sim_v3"

        base_sim_run_id = summary.get("sim_run_id") or ctx.resolved_run_id
        if not base_sim_run_id:
            raise FileNotFoundError("Base sim_run_id missing; cannot patch worlds.")

        sim_day_dir = root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date.isoformat()}"
        base_sim_dir = sim_day_dir / f"run={base_sim_run_id}"
        base_matrix_path = base_sim_dir / "worlds_matrix.parquet"
        base_proj_path = base_sim_dir / "projections.parquet"
        if not base_matrix_path.exists():
            raise FileNotFoundError(f"Base worlds_matrix.parquet missing: {base_matrix_path}")
        if not base_proj_path.exists():
            raise FileNotFoundError(f"Base sim projections.parquet missing: {base_proj_path}")

        base_matrix = pd.read_parquet(base_matrix_path)
        base_sim_proj = pd.read_parquet(base_proj_path)
        n_worlds = int(base_matrix.shape[0])

        # Run a partial sim (this game only) into a temp output root.
        with tempfile.TemporaryDirectory(prefix="ops-sim-game-") as tmp:
            tmp_root = Path(tmp) / "worlds_fpts_v2"
            tmp_run_id = "ops_partial"
            generate_worlds_main(
                start_date=game_date.isoformat(),
                end_date=game_date.isoformat(),
                n_worlds=n_worlds,
                profile=sim_profile,
                data_root=root,
                output_root=tmp_root,
                sim_run_id=tmp_run_id,
                rates_run_id=str(rates_run_id) if rates_run_id else None,
                minutes_run_id=str(minutes_run_id) if minutes_run_id else None,
                seed=None,
                min_play_prob=None,
                game_id=int(game_id),
                minutes_override_mode=minutes_override_mode_eff,
                override_infeasible=override_infeasible_eff,
            )

            partial_dir = tmp_root / f"game_date={game_date.isoformat()}" / f"run={tmp_run_id}"
            partial_matrix_path = partial_dir / "worlds_matrix.parquet"
            partial_proj_path = partial_dir / "projections.parquet"
            if not partial_matrix_path.exists() or not partial_proj_path.exists():
                raise FileNotFoundError(f"Partial sim outputs missing under {partial_dir}")

            partial_matrix = pd.read_parquet(partial_matrix_path)
            partial_proj = pd.read_parquet(partial_proj_path)

        if int(partial_matrix.shape[0]) != n_worlds:
            raise ValueError(
                f"Partial sim worlds mismatch: base={n_worlds} partial={partial_matrix.shape[0]}"
            )

        # Patch worlds matrix columns (player_id strings as columns).
        patched_matrix = base_matrix.copy()
        for col in partial_matrix.columns:
            patched_matrix[str(col)] = partial_matrix[col].to_numpy(copy=False)

        # Patch sim projections rows.
        patched_sim_proj = _patch_sim_projections(
            base_sim_proj,
            partial_proj,
            key_cols=("player_id", "game_id"),
        )

        # Persist as a new sim run dir and advance sim latest pointer.
        new_run_id = _generate_run_id(root, slate_day=game_date)
        new_sim_dir = sim_day_dir / f"run={new_run_id}"
        new_sim_dir.mkdir(parents=True, exist_ok=True)
        patched_sim_proj.to_parquet(new_sim_dir / "projections.parquet", index=False)
        patched_matrix.to_parquet(new_sim_dir / "worlds_matrix.parquet", index=False)
        _atomic_write_json(
            sim_day_dir / "latest_run.json",
            {"run_id": new_run_id, "updated_at": _utc_now_iso(), "source": "ops_patch_game"},
        )

        if not isinstance(draft_group_id, str) or not draft_group_id:
            raise ValueError(
                "Missing draft_group_id in unified projections summary; cannot finalize projections."
            )

        out_path = finalize_projections(
            game_date=game_date,
            projections_run_id=new_run_id,
            draft_group_id=draft_group_id,
            data_root=root,
            minutes_run_id=str(minutes_run_id) if minutes_run_id else None,
            sim_run_id=new_run_id,
            ownership_run_id=str(ownership_run_id) if ownership_run_id else None,
            merge_locked_games=True,
        )
        if out_path is None:
            raise RuntimeError("Finalize projections failed (no output written).")

        if pin_projections_run:
            pinned_path = root / "artifacts" / "projections" / game_date.isoformat() / "pinned_run.json"
            _atomic_write_json(pinned_path, {"run_id": new_run_id, "updated_at": _utc_now_iso(), "source": "ops_patch_game"})

        write_status(
            JobStatus(
                job_name="ops_patch_worlds_matrix_game",
                stage="ops",
                target_date=game_date.isoformat(),
                run_ts=run_ts,
                status="success",
                rows_written=int(len(patched_sim_proj)),
                expected_rows=int(len(patched_sim_proj)),
                message=f"patched game_id={game_id} projections_run_id={new_run_id}",
            )
        )

        return {
            "status": "success",
            "date": game_date.isoformat(),
            "game_id": int(game_id),
            "run_ts": run_ts,
            "projections_run_id": new_run_id,
            "sim_run_id": new_run_id,
            "sim_profile": sim_profile,
            "n_worlds": n_worlds,
            "output_path": str(out_path),
            "minutes_override_mode": minutes_override_mode_eff,
            "override_infeasible": override_infeasible_eff,
        }
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="ops_patch_worlds_matrix_game",
                stage="ops",
                target_date=game_date.isoformat(),
                run_ts=run_ts,
                status="error",
                rows_written=0,
                expected_rows=None,
                message=str(exc),
            )
        )
        raise
