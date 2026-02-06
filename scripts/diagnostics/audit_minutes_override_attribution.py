"""Audit minutes override attribution end-to-end (ops overrides → reconcile → sim).

Computes per-player minutes vectors for a slate date:
  m0: baseline minutes from minutes.parquet (scorer output)
  m1: pre-reconcile minutes (ops overrides applied; team reconcile OFF)
  m2: post-reconcile minutes (ops overrides applied; team reconcile ON; matches effective_minutes.parquet logic)
  m3: sim mean minutes (prefer mean over minutes_matrix.parquet worlds; fallback to sim projections.parquet)

Then emits per (game_id, team_id):
  - Team totals at each stage (and deviations from 240)
  - Reconcile diagnostics (cap/lock infeasibility)
  - Top movers at each stage: (m1-m0), (m2-m1), (m3-m2)
  - Highlight cases where sim seems to “undo” effective-layer intent.

Usage:
  uv run python -m scripts.diagnostics.audit_minutes_override_attribution --date 2026-01-29 --run-id 20260129T204959Z

Optional:
  --overrides-file /path/to/overrides.json  (apply overrides from an explicit file, without mutating your data_root)
  --game-id 123 --game-id 456              (filter)
  --sim-run-id <id>                        (if sim outputs are from a different run)
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.minutes.reconcile import TeamReconcileDiagnostics, reconcile_team_minutes
from projections.ops.overrides import overrides_path
from projections.ops.overrides import apply_overrides_to_minutes_df as apply_ops_overrides

app = typer.Typer(add_completion=False)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _coerce_id_str(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype="string")
    out = series.astype("string", copy=False).fillna("")
    numeric = pd.to_numeric(series, errors="coerce")
    int_like = numeric.notna() & (numeric % 1 == 0)
    if int_like.any():
        out = out.where(~int_like, numeric.where(int_like).astype("Int64").astype("string"))
    return out.str.replace(r"\.0$", "", regex=True)


def _resolve_run_dir(base_dir: Path, *, run_id: str | None, required_file: str) -> tuple[Path, str | None]:
    if run_id:
        run_dir = base_dir / f"run={run_id}"
        if not (run_dir / required_file).exists():
            raise FileNotFoundError(f"Missing {required_file} under {run_dir}")
        return run_dir, run_id

    latest = _read_json(base_dir / "latest_run.json")
    latest_id = str(latest.get("run_id")) if latest and latest.get("run_id") else None
    if latest_id and (base_dir / f"run={latest_id}" / required_file).exists():
        return base_dir / f"run={latest_id}", latest_id

    run_dirs = sorted([p for p in base_dir.glob("run=*") if p.is_dir()], reverse=True)
    for candidate in run_dirs:
        if (candidate / required_file).exists():
            resolved = candidate.name.split("=", 1)[1] if candidate.name.startswith("run=") else None
            return candidate, resolved

    raise FileNotFoundError(f"No {required_file} found under {base_dir}")


def _load_ops_overrides_payload(
    *,
    game_date: date_cls,
    data_root: Path,
    overrides_file: Path | None,
) -> tuple[dict[str, Any] | None, Path]:
    if overrides_file is not None:
        payload = _read_json(overrides_file)
        if payload is None:
            raise ValueError(f"Failed to read overrides JSON: {overrides_file}")
        return payload, overrides_file

    canonical = overrides_path(game_date, data_root=data_root)
    payload = _read_json(canonical)
    return payload, canonical


def _iter_override_items(payload: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    if not payload:
        return []
    raw = payload.get("overrides", [])
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _override_fields_map(payload: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _iter_override_items(payload):
        gid = item.get("game_id")
        pid = item.get("player_id")
        if gid is None or pid is None:
            continue
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        out[(str(gid), str(pid))] = fields
    return out


def _minutes_col_for_attribution(df: pd.DataFrame) -> str:
    # Use the same center semantics as ops effective layer for minutes_v1 frames:
    # minutes_final is derived from minutes_p50_cond when effective_minutes absent.
    if "minutes_p50_cond" in df.columns:
        return "minutes_p50_cond"
    if "minutes_p50" in df.columns:
        return "minutes_p50"
    if "minutes_final" in df.columns:
        return "minutes_final"
    raise ValueError("Missing minutes center column (expected minutes_p50_cond/minutes_p50/minutes_final)")


@dataclass(frozen=True)
class TeamStageSummary:
    game_id: str
    team_id: str
    sum_m0: float
    sum_m1: float
    sum_m2: float
    sum_m3: float | None
    dev_m0: float
    dev_m1: float
    dev_m2: float
    dev_m3: float | None
    reconcile_diag: dict[str, Any] | None


def _top_movers(
    df_team: pd.DataFrame,
    *,
    delta_col: str,
    n: int,
    min_abs: float = 0.25,
) -> dict[str, list[dict[str, Any]]]:
    d = pd.to_numeric(df_team[delta_col], errors="coerce").fillna(0.0).astype(float)
    base = df_team.copy()
    base[delta_col] = d
    base["abs_delta"] = d.abs()
    base = base.loc[base["abs_delta"] >= float(min_abs)].copy()
    if base.empty:
        return {"gainers": [], "donors": []}

    extra_cols = [
        c
        for c in (
            "minutes_lock_eff",
            "minutes_target_eff",
            "minutes_target",
            "minutes_lock",
            "ops_depth_role",
            "status",
        )
        if c in base.columns and c != delta_col
    ]
    cols = ["player_id", "player_name", delta_col, *extra_cols]
    gainers = (
        base.loc[base[delta_col] > 0, cols]
        .sort_values(delta_col, ascending=False)
        .head(int(max(n, 1)))
        .to_dict(orient="records")
    )
    donors = (
        base.loc[base[delta_col] < 0, cols]
        .sort_values(delta_col, ascending=True)
        .head(int(max(n, 1)))
        .to_dict(orient="records")
    )
    for rec in [*gainers, *donors]:
        # Normalize schema for downstream renderers.
        rec["delta"] = float(rec.pop(delta_col))
        if "minutes_lock_eff" in rec:
            rec["minutes_lock_eff"] = bool(rec["minutes_lock_eff"])
        if "minutes_lock" in rec and rec["minutes_lock"] is not None and rec["minutes_lock"] is not pd.NA:
            rec["minutes_lock"] = bool(rec["minutes_lock"])
        for k in ("minutes_target_eff", "minutes_target"):
            if k in rec and rec[k] is not None and rec[k] is not pd.NA and not (isinstance(rec[k], float) and np.isnan(rec[k])):
                rec[k] = float(rec[k])
    return {"gainers": gainers, "donors": donors}


def _compute_reconcile_diagnostics(
    df_pre: pd.DataFrame,
    *,
    minutes_col: str,
    locked_mask: pd.Series,
    target_team_minutes: float = 240.0,
) -> dict[tuple[str, str], TeamReconcileDiagnostics]:
    required = {"game_id", "team_id", "status", "play_prob", minutes_col}
    if not required <= set(df_pre.columns):
        missing = sorted(required - set(df_pre.columns))
        raise ValueError(f"pre-reconcile df missing required columns: {missing}")

    work = df_pre.copy()
    work["game_id"] = _coerce_id_str(work["game_id"])
    work["team_id"] = _coerce_id_str(work["team_id"])
    status_lower = work["status"].astype("string").fillna("").str.strip().str.lower()

    out: dict[tuple[str, str], TeamReconcileDiagnostics] = {}
    for (gid, tid), idxs in work.groupby(["game_id", "team_id"], sort=False).groups.items():
        idx = pd.Index(idxs)
        g = work.loc[idx]
        g_out = status_lower.loc[idx].eq("out")
        g_locked = locked_mask.reindex(idx).fillna(False).astype(bool) | g_out

        _reconciled, diag = reconcile_team_minutes(
            g,
            float(target_team_minutes),
            minutes_col=minutes_col,
            cap_col=None,
            weight_col=None,
            state_col=None,
            locked_mask=g_locked,
            default_cap=48.0,
            max_passes=5,
            eps=1e-6,
        )
        out[(str(gid), str(tid))] = diag
    return out


def _load_sim_minutes_mean(
    *,
    data_root: Path,
    game_date: str,
    sim_run_id: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_dir = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={game_date}"
    run_dir, resolved = _resolve_run_dir(base_dir, run_id=sim_run_id, required_file="projections.parquet")

    meta: dict[str, Any] = {"sim_run_id": resolved, "sim_run_dir": str(run_dir)}

    proj = pd.read_parquet(run_dir / "projections.parquet")
    if proj.empty:
        return proj, {**meta, "minutes_source": "missing"}

    proj = proj.copy()
    for c in ("game_id", "team_id", "player_id"):
        if c in proj.columns:
            proj[c] = _coerce_id_str(proj[c])

    # Prefer mean derived from minutes worlds (unconditional DNP=0).
    mean_col = "minutes_sim_mean_uncond" if "minutes_sim_mean_uncond" in proj.columns else "minutes_sim_mean"
    if mean_col not in proj.columns:
        raise ValueError("sim projections missing minutes_sim_mean[_uncond]")

    # If the optional minutes_matrix exists, compute mean from it and use that as the source of truth.
    minutes_matrix_path = run_dir / "minutes_matrix.parquet"
    if minutes_matrix_path.exists():
        mm = pd.read_parquet(minutes_matrix_path)
        # Columns are player_id strings.
        mm.columns = [str(c) for c in mm.columns]
        minutes_mean = mm.mean(axis=0).astype(float)
        mean_from_matrix = pd.DataFrame({"player_id": minutes_mean.index.astype(str), "minutes_sim_mean_matrix": minutes_mean.to_numpy()})
        mean_from_matrix["player_id"] = _coerce_id_str(mean_from_matrix["player_id"])
        proj = proj.merge(mean_from_matrix, on="player_id", how="left")
        # Use matrix mean when present (should be extremely close to projections mean).
        proj["minutes_m3"] = proj["minutes_sim_mean_matrix"].where(proj["minutes_sim_mean_matrix"].notna(), proj[mean_col])
        meta["minutes_source"] = "minutes_matrix.parquet"
        meta["minutes_matrix_path"] = str(minutes_matrix_path)
        meta["minutes_mean_col_fallback"] = mean_col
    else:
        proj["minutes_m3"] = pd.to_numeric(proj[mean_col], errors="coerce").fillna(0.0).astype(float)
        meta["minutes_source"] = "projections.parquet"
        meta["minutes_mean_col"] = mean_col

    return proj[["game_id", "team_id", "player_id", "minutes_m3"]].copy(), meta


def compute_attribution(
    *,
    baseline_minutes: pd.DataFrame,
    game_date: date_cls,
    data_root: Path,
    overrides_payload: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute m0/m1/m2 and reconcile diagnostics (no sim)."""
    base = baseline_minutes.copy()
    for c in ("game_id", "team_id", "player_id"):
        base[c] = _coerce_id_str(base[c])
    if "player_name" not in base.columns:
        base["player_name"] = ""

    minutes_col = _minutes_col_for_attribution(base)
    base["m0"] = pd.to_numeric(base[minutes_col], errors="coerce").fillna(0.0).astype(float)

    # Apply overrides from either canonical data_root or a provided file, without mutating the user's data_root.
    effective_root = data_root
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    if overrides_payload is not None:
        tmp_dir = tempfile.TemporaryDirectory(prefix="minutes-override-attrib-")
        effective_root = Path(tmp_dir.name)
        canonical_path = overrides_path(game_date, data_root=effective_root)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text(json.dumps(overrides_payload, indent=2, sort_keys=True), encoding="utf-8")

    try:
        pre = apply_ops_overrides(
            base,
            game_date=game_date,
            data_root=effective_root,
            reconcile_team_minutes=False,
            log_diagnostics=False,
            force_reconcile=False,
        )
        post = apply_ops_overrides(
            base,
            game_date=game_date,
            data_root=effective_root,
            reconcile_team_minutes=True,
            log_diagnostics=False,
            force_reconcile=True,
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    pre = pre.copy()
    post = post.copy()
    for df in (pre, post):
        for c in ("game_id", "team_id", "player_id"):
            df[c] = _coerce_id_str(df[c])
        if "player_name" not in df.columns:
            df["player_name"] = ""

    mcol_pre = _minutes_col_for_attribution(pre)
    mcol_post = _minutes_col_for_attribution(post)
    base["m1"] = pd.to_numeric(pre.set_index(["game_id", "team_id", "player_id"])[mcol_pre], errors="coerce").reindex(
        base.set_index(["game_id", "team_id", "player_id"]).index
    ).fillna(0.0).to_numpy(dtype=float)
    base["m2"] = pd.to_numeric(post.set_index(["game_id", "team_id", "player_id"])[mcol_post], errors="coerce").reindex(
        base.set_index(["game_id", "team_id", "player_id"]).index
    ).fillna(0.0).to_numpy(dtype=float)

    # Carry a few useful flags from the effective layer (when present).
    for flag_col in (
        "ops_override_applied",
        "minutes_delta",
        "minutes_delta_applied",
        "minutes_target",
        "minutes_lock",
        "minutes_target_eff",
        "minutes_lock_eff",
        "status",
        "play_prob",
    ):
        if flag_col in pre.columns:
            s = pre.set_index(["game_id", "team_id", "player_id"])[flag_col]
            base[flag_col] = s.reindex(base.set_index(["game_id", "team_id", "player_id"]).index).to_numpy()
        else:
            base[flag_col] = pd.NA

    base["ops_override_applied"] = base["ops_override_applied"].fillna(False).astype(bool)
    base["minutes_delta_applied"] = base["minutes_delta_applied"].fillna(False).astype(bool)
    base["minutes_lock_eff"] = base["minutes_lock_eff"].fillna(False).astype(bool)
    base["play_prob"] = pd.to_numeric(base["play_prob"], errors="coerce").fillna(1.0).astype(float)
    base["status"] = base["status"].fillna("").astype(str)

    # Locked-mask approximation matching current ops hard-target semantics:
    # - OUT and play_prob<=0 always locked (enforced to 0)
    # - minutes_lock_eff is the authoritative fixed-mask used by ops reconcile + sim allocator
    status_lower = base["status"].astype(str).str.strip().str.lower()
    locked_out = status_lower.eq("out") | (base["play_prob"] <= 0.0)
    locked = locked_out.to_numpy(dtype=bool) | base["minutes_lock_eff"].to_numpy(dtype=bool)
    locked_mask = pd.Series(locked, index=base.index, dtype=bool)

    diags = _compute_reconcile_diagnostics(base, minutes_col=minutes_col, locked_mask=locked_mask)

    meta = {
        "minutes_col": minutes_col,
        "n_overrides": int(len(list(_iter_override_items(overrides_payload)))) if overrides_payload else 0,
    }
    diag_rows = []
    for (gid, tid), d in diags.items():
        rec = {"game_id": gid, "team_id": tid, **asdict(d)}
        diag_rows.append(rec)
    diag_df = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame(columns=["game_id", "team_id"])
    meta["reconcile_diagnostics"] = diag_df

    base["d10"] = base["m1"] - base["m0"]
    base["d21"] = base["m2"] - base["m1"]
    return base, meta


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Slate date (YYYY-MM-DD)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Minutes run_id (defaults to latest)."),
    sim_run_id: str | None = typer.Option(None, "--sim-run-id", help="Sim run_id (defaults to minutes run_id/latest)."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional override for PROJECTIONS_DATA_ROOT."),
    overrides_file: Path | None = typer.Option(None, "--overrides-file", help="Explicit ops overrides JSON to apply."),
    game_id: list[str] = typer.Option([], "--game-id", help="Filter to one or more game_id values."),
    team_id: list[str] = typer.Option([], "--team-id", help="Filter to one or more team_id values."),
    top: int = typer.Option(8, "--top", help="Top gainers/donors to show per stage per team."),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Output directory for JSON + table."),
) -> None:
    root = Path(data_root) if data_root is not None else paths.data_path()
    slate_day = date_cls.fromisoformat(date)

    minutes_base = root / "artifacts" / "minutes_v1" / "daily" / date
    minutes_run_dir, resolved_minutes_run_id = _resolve_run_dir(
        minutes_base,
        run_id=run_id,
        required_file="minutes.parquet",
    )
    minutes_path = minutes_run_dir / "minutes.parquet"
    baseline = pd.read_parquet(minutes_path)
    if baseline.empty:
        raise typer.BadParameter(f"Baseline minutes empty: {minutes_path}")

    if "game_date" in baseline.columns:
        baseline = baseline.copy()
        baseline["game_date"] = pd.to_datetime(baseline["game_date"], errors="coerce").dt.date
        baseline = baseline.loc[baseline["game_date"] == slate_day].copy()

    required = {"game_id", "team_id", "player_id"}
    if not required <= set(baseline.columns):
        raise typer.BadParameter(f"minutes.parquet missing required columns: {sorted(required - set(baseline.columns))}")

    for c in ("game_id", "team_id", "player_id"):
        baseline[c] = _coerce_id_str(baseline[c])
    if "player_name" not in baseline.columns:
        baseline["player_name"] = ""

    if game_id:
        baseline = baseline.loc[baseline["game_id"].isin([str(x) for x in game_id])].copy()
    if team_id:
        baseline = baseline.loc[baseline["team_id"].isin([str(x) for x in team_id])].copy()
    if baseline.empty:
        typer.echo("[audit] No minutes rows after filters; exiting.")
        raise typer.Exit(0)

    overrides_payload, overrides_path_used = _load_ops_overrides_payload(
        game_date=slate_day,
        data_root=root,
        overrides_file=overrides_file,
    )

    attrib_df, meta = compute_attribution(
        baseline_minutes=baseline,
        game_date=slate_day,
        data_root=root,
        overrides_payload=overrides_payload,
    )

    # Load sim mean minutes (m3).
    sim_df, sim_meta = _load_sim_minutes_mean(
        data_root=root,
        game_date=date,
        sim_run_id=sim_run_id or resolved_minutes_run_id,
    )
    sim_df = sim_df.copy()
    for c in ("game_id", "team_id", "player_id"):
        sim_df[c] = _coerce_id_str(sim_df[c])

    attrib_df = attrib_df.merge(sim_df, on=["game_id", "team_id", "player_id"], how="left")
    attrib_df["m3"] = pd.to_numeric(attrib_df.get("minutes_m3"), errors="coerce").astype(float)
    attrib_df["m3"] = attrib_df["m3"].where(attrib_df["m3"].notna(), np.nan)
    attrib_df = attrib_df.drop(columns=["minutes_m3"], errors="ignore")
    attrib_df["d32"] = attrib_df["m3"] - attrib_df["m2"]

    diag_df: pd.DataFrame = meta["reconcile_diagnostics"]
    if not diag_df.empty:
        diag_df = diag_df.copy()
        diag_df["game_id"] = _coerce_id_str(diag_df["game_id"])
        diag_df["team_id"] = _coerce_id_str(diag_df["team_id"])

    # Summaries per team-game.
    team_rows: list[dict[str, Any]] = []
    group_cols = ["game_id", "team_id"]
    for (gid, tid), g in attrib_df.groupby(group_cols, sort=False):
        sum_m0 = float(pd.to_numeric(g["m0"], errors="coerce").fillna(0.0).sum())
        sum_m1 = float(pd.to_numeric(g["m1"], errors="coerce").fillna(0.0).sum())
        sum_m2 = float(pd.to_numeric(g["m2"], errors="coerce").fillna(0.0).sum())
        sum_m3 = float(pd.to_numeric(g["m3"], errors="coerce").fillna(0.0).sum()) if g["m3"].notna().any() else None

        diag_row = None
        if not diag_df.empty:
            match = diag_df.loc[(diag_df["game_id"] == gid) & (diag_df["team_id"] == tid)]
            if len(match) == 1:
                diag_row = match.iloc[0].to_dict()

        stage_moves = {
            "m1_minus_m0": _top_movers(g, delta_col="d10", n=top),
            "m2_minus_m1": _top_movers(g, delta_col="d21", n=top),
            "m3_minus_m2": _top_movers(g.loc[g["m3"].notna()], delta_col="d32", n=top) if sum_m3 is not None else {"gainers": [], "donors": []},
        }

        # “Undo” heuristic: players that gained minutes via overrides/reconcile but lose them in sim mean.
        undo = []
        if sum_m3 is not None:
            for _, row in g.iterrows():
                d10 = float(row.get("d10") or 0.0)
                d21 = float(row.get("d21") or 0.0)
                d32 = float(row.get("d32") or 0.0) if pd.notna(row.get("d32")) else 0.0
                if (d10 > 1.0 or d21 > 1.0) and d32 < -1.0:
                    undo.append(
                        {
                            "player_id": str(row["player_id"]),
                            "player_name": str(row.get("player_name") or ""),
                            "d10": d10,
                            "d21": d21,
                            "d32": d32,
                        }
                    )
            undo = sorted(undo, key=lambda r: abs(float(r.get("d32") or 0.0)), reverse=True)[: int(max(10, top))]

        team_summary = TeamStageSummary(
            game_id=str(gid),
            team_id=str(tid),
            sum_m0=sum_m0,
            sum_m1=sum_m1,
            sum_m2=sum_m2,
            sum_m3=sum_m3,
            dev_m0=sum_m0 - 240.0,
            dev_m1=sum_m1 - 240.0,
            dev_m2=sum_m2 - 240.0,
            dev_m3=(sum_m3 - 240.0) if sum_m3 is not None else None,
            reconcile_diag=diag_row,
        )

        team_rows.append(
            {
                "team": asdict(team_summary),
                "top_movers": stage_moves,
                "undo_candidates": undo,
            }
        )

    payload = {
        "version": 1,
        "date": date,
        "minutes_run_id": resolved_minutes_run_id,
        "minutes_path": str(minutes_path),
        "ops_overrides_path": str(overrides_path_used),
        "sim": sim_meta,
        "meta": {"minutes_col": meta["minutes_col"], "n_overrides": meta["n_overrides"]},
        "teams": team_rows,
    }

    out_base = out_dir
    if out_base is None:
        rid = resolved_minutes_run_id or "unknown"
        out_base = root / "artifacts" / "diagnostics" / "minutes_override_attribution" / f"game_date={date}" / f"run={rid}"
    out_base.mkdir(parents=True, exist_ok=True)

    json_path = out_base / "minutes_override_attribution.json"
    _atomic_write_json(json_path, payload)

    # Human-readable table.
    lines: list[str] = []
    lines.append(f"[audit] date={date} minutes_run_id={resolved_minutes_run_id} sim_run_id={sim_meta.get('sim_run_id')}")
    lines.append(f"[audit] minutes_path={minutes_path}")
    lines.append(f"[audit] overrides_path={overrides_path_used}")
    lines.append(f"[audit] sim_dir={sim_meta.get('sim_run_dir')} minutes_source={sim_meta.get('minutes_source')}")
    if sim_meta.get("minutes_matrix_path"):
        lines.append(f"[audit] minutes_matrix_path={sim_meta.get('minutes_matrix_path')}")
    lines.append("")

    for entry in payload["teams"]:
        team = entry["team"]
        diag = team.get("reconcile_diag") or {}
        diag_bits = []
        if diag:
            diag_bits.append(f"passes={int(diag.get('passes', 0))}")
            diag_bits.append(f"cap_infeasible={bool(diag.get('cap_infeasible', False))}")
            diag_bits.append(f"locked_infeasible={bool(diag.get('locked_infeasible', False))}")
            diag_bits.append(f"residual={float(diag.get('residual', 0.0)):+.4f}")
        diag_str = " ".join(diag_bits) if diag_bits else "n/a"
        lines.append(
            f"[team] game_id={team['game_id']} team_id={team['team_id']} "
            f"sum(m0)={team['sum_m0']:.1f} sum(m1)={team['sum_m1']:.1f} sum(m2)={team['sum_m2']:.1f} "
            f"sum(m3)={(team['sum_m3'] if team['sum_m3'] is not None else float('nan')):.1f} | {diag_str}"
        )

        for stage_key, label in (
            ("m1_minus_m0", "d10 (override/delta pre-reconcile)"),
            ("m2_minus_m1", "d21 (team reconcile)"),
            ("m3_minus_m2", "d32 (sim mean vs effective)"),
        ):
            movers = entry["top_movers"].get(stage_key) or {}
            gainers = movers.get("gainers") or []
            donors = movers.get("donors") or []
            if not gainers and not donors:
                continue
            lines.append(f"  {label}:")
            for r in gainers:
                lock_eff = bool(r.get("minutes_lock_eff", False))
                target_eff = r.get("minutes_target_eff")
                target = target_eff if target_eff is not None else r.get("minutes_target")
                suffix = ""
                if lock_eff:
                    suffix = " lock"
                    if target is not None:
                        try:
                            suffix += f" tgt={float(target):.1f}"
                        except (TypeError, ValueError):
                            pass
                lines.append(
                    f"    + {r['player_id']} {str(r.get('player_name',''))[:24]:24s} {float(r['delta']):+.2f}{suffix}"
                )
            for r in donors:
                lock_eff = bool(r.get("minutes_lock_eff", False))
                target_eff = r.get("minutes_target_eff")
                target = target_eff if target_eff is not None else r.get("minutes_target")
                suffix = ""
                if lock_eff:
                    suffix = " lock"
                    if target is not None:
                        try:
                            suffix += f" tgt={float(target):.1f}"
                        except (TypeError, ValueError):
                            pass
                lines.append(
                    f"    - {r['player_id']} {str(r.get('player_name',''))[:24]:24s} {float(r['delta']):+.2f}{suffix}"
                )

        undo = entry.get("undo_candidates") or []
        if undo:
            lines.append("  undo_candidates (gained earlier, lost in sim):")
            for r in undo[: min(len(undo), top)]:
                lines.append(
                    f"    * {r['player_id']} {r.get('player_name','')[:24]:24s} d10={r['d10']:+.1f} d21={r['d21']:+.1f} d32={r['d32']:+.1f}"
                )
        lines.append("")

    table_path = out_base / "minutes_override_attribution.txt"
    table_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    typer.echo("\n".join(lines).rstrip())
    typer.echo(f"\n[wrote] {json_path}")
    typer.echo(f"[wrote] {table_path}")


if __name__ == "__main__":
    app()
