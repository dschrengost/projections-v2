"""Audit play_prob provenance end-to-end and its impact on worlds DNP masking.

This script is intentionally read-only. It traces play_prob for a single slate date across:
  1) live minutes features (prior_play_prob)
  2) minutes model output (minutes.parquet + effective_minutes.parquet)
  3) LGBM play_prob head raw output (when available)
  4) unified projections (artifacts/projections/.../projections.parquet)
  5) optimizer pool (build_player_pool) mapping
  6) sim_v2 inputs (artifacts/sim_v2/.../projections.parquet) + worlds_matrix.parquet DNP rate

Usage:
  uv run python scripts/diagnostics/audit_play_prob_provenance.py --date 2026-01-16
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import typer

from projections.api.optimizer_service import build_player_pool
from projections.models.minutes_lgbm import predict_play_probability
from projections.paths import data_path
from projections.pipeline import control_plane
from projections.sim_v2.config import load_sim_v2_profile

app = typer.Typer(add_completion=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ResolvedRun:
    run_id: str | None
    run_dir: Path | None


def _resolve_run_dir(base_dir: Path, *, run_id: str | None) -> ResolvedRun:
    if run_id:
        candidate = base_dir / f"run={run_id}"
        return ResolvedRun(run_id=run_id, run_dir=candidate if candidate.exists() else None)

    promoted = control_plane.read_promoted_run_id(base_dir)
    if promoted:
        candidate = base_dir / f"run={promoted}"
        if candidate.exists():
            return ResolvedRun(run_id=promoted, run_dir=candidate)

    if control_plane.allow_unpromoted_run_reads():
        run_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run=")],
            reverse=True,
        )
        if run_dirs:
            rid = run_dirs[0].name.split("=", 1)[1]
            return ResolvedRun(run_id=rid, run_dir=run_dirs[0])

    return ResolvedRun(run_id=None, run_dir=None)


def _pick_main_draft_group_id(*, data_root: Path, game_date: str) -> int | None:
    base = data_root / "gold" / "dk_salaries" / "site=dk" / f"game_date={game_date}"
    if not base.exists():
        return None
    slates: list[tuple[int, int, int]] = []
    for dg_dir in base.glob("draft_group_id=*"):
        pq = dg_dir / "salaries.parquet"
        if not pq.exists():
            continue
        dg_id = int(dg_dir.name.split("=", 1)[1])
        try:
            df = pd.read_parquet(pq, columns=["salary"])
        except Exception:  # noqa: BLE001
            df = pd.read_parquet(pq)
        n_rows = int(len(df))
        n_salary = int(df["salary"].notna().sum()) if "salary" in df.columns else 0
        slates.append((dg_id, n_rows, n_salary))
    if not slates:
        return None
    has_salary = any(n_salary > 0 for _, _, n_salary in slates)
    candidates = [s for s in slates if s[2] > 0] if has_salary else slates
    candidates.sort(key=lambda x: (-x[2], -x[1], x[0]))
    return int(candidates[0][0])


def _as_eastern_game_date() -> str:
    et = ZoneInfo("America/New_York")
    return datetime.now(tz=et).date().isoformat()


def _normalize_status(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    return text


def _select_players(minutes_df: pd.DataFrame, *, n_total: int) -> list[str]:
    if minutes_df.empty:
        return []
    df = minutes_df.copy()
    if "player_id" not in df.columns:
        return []
    df["player_id"] = df["player_id"].astype(str)
    if "status" in df.columns:
        df["_status"] = df["status"].map(_normalize_status)
    else:
        df["_status"] = ""
    if "play_prob" in df.columns:
        df["_play_prob"] = pd.to_numeric(df["play_prob"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        df["_play_prob"] = 1.0
    if "minutes_p50" in df.columns:
        df["_minutes_p50"] = pd.to_numeric(df["minutes_p50"], errors="coerce").fillna(0.0)
    else:
        df["_minutes_p50"] = 0.0
    if "is_starter" in df.columns:
        df["_is_starter"] = pd.to_numeric(df["is_starter"], errors="coerce").fillna(0).astype(int)
    else:
        df["_is_starter"] = pd.to_numeric(df.get("starter_flag"), errors="coerce").fillna(0).astype(int)

    picked: list[str] = []

    def _take(mask: pd.Series, k: int) -> None:
        nonlocal picked
        if k <= 0:
            return
        take = df.loc[mask & ~df["player_id"].isin(picked)].sort_values("_minutes_p50", ascending=False).head(k)
        picked.extend(take["player_id"].tolist())

    # At least 2 OUT players.
    _take(df["_status"].eq("OUT") | (df["_play_prob"] <= 0.0), 2)
    # At least 3 Q/PROB players (or any <1.0 players if tags absent).
    _take(df["_status"].isin(["Q", "PROB"]), 3)
    if len(picked) < 5:
        _take((df["_play_prob"] > 0.0) & (df["_play_prob"] < 1.0), 5 - len(picked))
    # At least 2 healthy starters.
    _take((df["_is_starter"] > 0) & ~df["_status"].eq("OUT"), 2)

    # Fill remaining with top minutes.
    if len(picked) < n_total:
        _take(pd.Series(True, index=df.index), n_total - len(picked))

    return picked[:n_total]


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, *, on: str, suffix: str) -> pd.DataFrame:
    if right.empty or on not in right.columns:
        return left
    merged = left.merge(right, on=on, how="left", suffixes=("", suffix))
    return merged


def _load_table(path: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if columns is None:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path, columns=list(columns))
    except Exception:  # noqa: BLE001
        df = pd.read_parquet(path)
        cols = [c for c in columns if c in df.columns]
        return df.loc[:, cols].copy()


def _bundle_kind(bundle_dir: Path) -> str:
    if (bundle_dir / "lgbm_quantiles.joblib").exists():
        return "lgbm_quantiles"
    if (bundle_dir / "rotation_share_model.joblib").exists():
        return "rotation_share"
    if (bundle_dir / "minute_share_model.joblib").exists():
        return "minute_share"
    return "unknown"


def _load_worlds_selected(
    *,
    matrix_path: Path,
    player_ids: list[str],
) -> pd.DataFrame:
    if not matrix_path.exists() or not player_ids:
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq

        cols = set(pq.ParquetFile(matrix_path).schema.names)
    except Exception:  # noqa: BLE001
        cols = set(pd.read_parquet(matrix_path, engine="pyarrow").columns)

    keep = [pid for pid in player_ids if pid in cols]
    if not keep:
        return pd.DataFrame()
    return pd.read_parquet(matrix_path, columns=keep)


def _compute_dnp_rates(worlds_df: pd.DataFrame) -> dict[str, float]:
    if worlds_df.empty:
        return {}
    mat = worlds_df.to_numpy(dtype=np.float64, copy=False)
    rates = np.mean(mat == 0.0, axis=0)
    return {str(pid): float(r) for pid, r in zip(worlds_df.columns, rates, strict=True)}


def _provenance_label(
    *,
    bundle_kind: str,
    minutes_play_prob: float | None,
    head_play_prob: float | None,
    status: str,
) -> str:
    st = _normalize_status(status)
    p = minutes_play_prob
    if p is None or not np.isfinite(p):
        return "missing"
    if st == "OUT" and p <= 0.0:
        if head_play_prob is not None and np.isfinite(head_play_prob) and head_play_prob > 0.0:
            return "forced_out_override"
        return "out"
    if bundle_kind == "lgbm_quantiles":
        if head_play_prob is None or not np.isfinite(head_play_prob):
            # LGBM bundle but no head artifacts or couldn't compute.
            if p >= 0.999:
                return "lgbm_head_missing_fallback_ones"
            return "lgbm_head_missing"
        # Heuristic: if p is exactly 1 for everyone, call out fallback.
        if p >= 0.999 and head_play_prob >= 0.999:
            return "lgbm_head_all_ones"
        return "lgbm_head"
    if bundle_kind == "rotation_share":
        return "rotation_share_model"
    if bundle_kind == "minute_share":
        if p >= 0.999:
            return "minute_share_default_ones"
        return "minute_share"
    if p >= 0.999:
        return "fallback_ones"
    return "unknown"


@app.command()
def main(
    date: str = typer.Option(None, "--date", help="Slate date (YYYY-MM-DD). Defaults to today's ET date."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Defaults to PROJECTIONS_DATA_ROOT."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional pipeline run_id override."),
    sim_output_root: Path | None = typer.Option(
        None,
        "--sim-output-root",
        help="Optional override for sim_v2 worlds root (directory containing game_date=.../run=...).",
    ),
    sim_run_id: str | None = typer.Option(None, "--sim-run-id", help="Optional sim_v2 run_id override."),
    n_players: int = typer.Option(10, "--n-players", min=5, max=25, help="How many players to print."),
    build_select: str = typer.Option(
        "largest",
        "--build-select",
        help="Which saved contest_sim build to use for top150 any-zero distribution: largest|latest",
    ),
    output_csv: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Where to write the CSV (default: /tmp/audit_play_prob_provenance_<date>.csv).",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = data_root or data_path()
    game_date = date or _as_eastern_game_date()
    typer.echo(f"[audit] data_root={root}")
    typer.echo(f"[audit] date={game_date}")
    if run_id:
        typer.echo(f"[audit] run_id override={run_id}")
    if sim_output_root is not None:
        typer.echo(f"[audit] sim_output_root override={sim_output_root}")
    if sim_run_id is not None:
        typer.echo(f"[audit] sim_run_id override={sim_run_id}")

    # -------------------- Stage 1: minutes live features --------------------
    feat_base = root / "live" / "features_minutes_v1" / game_date
    feat_run = _resolve_run_dir(feat_base, run_id=run_id) if feat_base.exists() else ResolvedRun(None, None)
    feat_path = feat_run.run_dir / "features.parquet" if feat_run.run_dir else None
    typer.echo(f"[audit] features_minutes_v1: run_id={feat_run.run_id} path={feat_path}")
    feat_df = _load_table(
        feat_path,
        columns=[
            "player_id",
            "player_name",
            "team_tricode",
            "team_abbr",
            "status",
            "prior_play_prob",
            "injury_as_of_ts",
            "injury_row_present",
            "injury_snapshot_missing",
            "is_out",
            "is_q",
            "is_prob",
            "starter_flag",
            "is_starter",
            "lineup_role",
        ],
    ) if feat_path else pd.DataFrame()
    if not feat_df.empty:
        feat_df["player_id"] = feat_df["player_id"].astype(str)
        feat_df = feat_df.rename(columns={"prior_play_prob": "play_prob_stage1_prior"})
    else:
        typer.echo("[audit] warning: minutes features not found; stage1 will be empty", err=True)

    # -------------------- Stage 2: minutes model output (and effective layer) --------------------
    minutes_base = root / "artifacts" / "minutes_v1" / "daily" / game_date
    minutes_run = _resolve_run_dir(minutes_base, run_id=run_id) if minutes_base.exists() else ResolvedRun(None, None)
    minutes_path = minutes_run.run_dir / "minutes.parquet" if minutes_run.run_dir else None
    minutes_summary_path = minutes_run.run_dir / "summary.json" if minutes_run.run_dir else None
    eff_path = minutes_run.run_dir / "effective_minutes.parquet" if minutes_run.run_dir else None
    typer.echo(f"[audit] minutes_v1: run_id={minutes_run.run_id} minutes={minutes_path} effective={eff_path}")

    minutes_df = _load_table(
        minutes_path,
        columns=[
            "player_id",
            "player_name",
            "team_tricode",
            "team_abbr",
            "status",
            "play_prob",
            "minutes_p50",
            "minutes_p50_cond",
            "starter_flag",
            "is_starter",
            "rotation_prob",
        ],
    ) if minutes_path else pd.DataFrame()
    if not minutes_df.empty:
        minutes_df["player_id"] = minutes_df["player_id"].astype(str)
        minutes_df = minutes_df.rename(columns={"play_prob": "play_prob_stage2_minutes"})

    eff_df = _load_table(
        eff_path,
        columns=[
            "player_id",
            "play_prob",
            "status",
            "minutes_p50",
            "minutes_p50_cond",
            "starter_flag",
            "is_starter",
        ],
    ) if eff_path else pd.DataFrame()
    if not eff_df.empty:
        eff_df["player_id"] = eff_df["player_id"].astype(str)
        eff_df = eff_df.rename(columns={"play_prob": "play_prob_stage2_effective"})

    bundle_dir: Path | None = None
    bundle_kind = "unknown"
    if minutes_summary_path and minutes_summary_path.exists():
        summary = _read_json(minutes_summary_path)
        raw_dir = summary.get("bundle_dir")
        if raw_dir:
            bundle_dir = Path(str(raw_dir))
            bundle_kind = _bundle_kind(bundle_dir)
    typer.echo(f"[audit] minutes bundle: kind={bundle_kind} dir={bundle_dir}")
    play_prob_producer = {
        "lgbm_quantiles": "projections.cli.score_minutes_v1:_score_rows → projections.models.minutes_lgbm.predict_play_probability",
        "rotation_share": "projections.cli.score_minutes_v1:_score_rows_rotshare → projections.minutes_v1.rotation_share.predict_play_prob",
        "minute_share": "projections.cli.score_minutes_v1:_score_rows_share (play_prob=1 except OUT)",
        "unknown": "unknown",
    }.get(bundle_kind, "unknown")
    typer.echo(f"[audit] play_prob producer (minutes): {play_prob_producer}")
    typer.echo("[audit] play_prob prior (features): projections.features.availability.attach_availability_features")
    typer.echo("[audit] play_prob carried to unified: projections.cli.finalize_projections (from effective_minutes.parquet)")
    typer.echo("[audit] play_prob used for worlds: scripts.sim_v2.generate_worlds_fpts_v2 (active_mask sampling)")

    # -------------------- Stage 3: LGBM play_prob head raw (if available) --------------------
    head_df = pd.DataFrame()
    head_reason = "not_applicable"
    if bundle_dir and bundle_kind == "lgbm_quantiles":
        try:
            import joblib

            bundle = joblib.load(bundle_dir / "lgbm_quantiles.joblib")
            feature_cols = bundle.get("feature_columns") or []
            artifacts = bundle.get("play_probability")
            enabled = bool(bundle.get("play_prob_enabled", True))
            if not enabled:
                head_reason = "bundle.play_prob_enabled=false"
            elif artifacts is None:
                head_reason = "bundle.play_probability missing"
            elif feat_path is None or not Path(feat_path).exists():
                head_reason = "stage1 features.parquet missing"
            else:
                # Load the full feature slice required by the play_prob head (cheap: parquet column projection).
                feat_matrix = _load_table(Path(feat_path), columns=["player_id", *feature_cols])
                if feat_matrix.empty:
                    head_reason = "stage1 features empty"
                else:
                    feat_matrix["player_id"] = feat_matrix["player_id"].astype(str)
                    missing = [c for c in feature_cols if c not in feat_matrix.columns]
                    if missing:
                        head_reason = f"features missing {len(missing)} required cols"
                    else:
                        X = feat_matrix.loc[:, feature_cols].copy()
                        play_prob_raw = predict_play_probability(artifacts, X)
                        head_df = pd.DataFrame(
                            {
                                "player_id": feat_matrix["player_id"].to_numpy(),
                                "play_prob_stage3_lgbm_head_raw": play_prob_raw.astype(float),
                            }
                        )
                        head_reason = f"ok (n={len(head_df)})"
        except Exception as exc:  # noqa: BLE001
            head_reason = f"failed to compute lgbm head: {exc}"
    else:
        head_reason = f"bundle_kind={bundle_kind}"
    typer.echo(f"[audit] play_prob head: {head_reason}")

    # -------------------- Stage 4: unified projections --------------------
    proj_base = root / "artifacts" / "projections" / game_date
    proj_run = _resolve_run_dir(proj_base, run_id=run_id) if proj_base.exists() else ResolvedRun(None, None)
    proj_path = proj_run.run_dir / "projections.parquet" if proj_run.run_dir else None
    typer.echo(f"[audit] unified projections: run_id={proj_run.run_id} path={proj_path}")
    proj_df = _load_table(
        proj_path,
        columns=["player_id", "player_name", "team_tricode", "team_abbr", "status", "play_prob", "pred_own_pct"],
    ) if proj_path else pd.DataFrame()
    if not proj_df.empty:
        proj_df["player_id"] = proj_df["player_id"].astype(str)
        proj_df = proj_df.rename(columns={"play_prob": "play_prob_stage4_unified"})

    # -------------------- Stage 5: optimizer pool --------------------
    dg_id = _pick_main_draft_group_id(data_root=root, game_date=game_date)
    pool_df = pd.DataFrame()
    pool_reason = None
    if dg_id is None:
        pool_reason = "could not resolve draft_group_id"
    else:
        try:
            pool = build_player_pool(
                game_date=game_date,
                draft_group_id=int(dg_id),
                site="dk",
                run_id=proj_run.run_id,
                data_root=root,
                use_user_overrides=False,
            )
            pool_df = pd.DataFrame(pool)
            if not pool_df.empty and "player_id" in pool_df.columns:
                pool_df["player_id"] = pool_df["player_id"].astype(str)
                # Stage5 play_prob is the unified projections play_prob keyed by player_id (optimizer carries it through merge).
                if not proj_df.empty:
                    own_map = dict(zip(proj_df["player_id"], proj_df["play_prob_stage4_unified"], strict=False))
                    pool_df["play_prob_stage5_pool"] = pool_df["player_id"].map(own_map)
        except Exception as exc:  # noqa: BLE001
            pool_reason = str(exc)
    typer.echo(f"[audit] optimizer pool: draft_group_id={dg_id} ok={not pool_df.empty} err={pool_reason}")
    pool_stage_df = pd.DataFrame()
    if not pool_df.empty and "player_id" in pool_df.columns:
        pool_stage_df = pool_df.loc[:, ["player_id", "play_prob_stage5_pool"]].copy()
        pool_stage_df["in_optimizer_pool"] = True

    # -------------------- Stage 6: sim_v2 inputs + worlds dnp rates --------------------
    sim_base_root = sim_output_root or (root / "artifacts" / "sim_v2" / "worlds_fpts_v2")
    sim_base = sim_base_root / f"game_date={game_date}"
    sim_run = _resolve_run_dir(sim_base, run_id=(sim_run_id or run_id)) if sim_base.exists() else ResolvedRun(None, None)
    sim_dir = sim_run.run_dir
    sim_proj_path = sim_dir / "projections.parquet" if sim_dir else None
    matrix_path = sim_dir / "worlds_matrix.parquet" if sim_dir else None
    typer.echo(f"[audit] sim_v2: run_id={sim_run.run_id} dir={sim_dir}")
    typer.echo(f"[audit] sim_v2: projections={sim_proj_path}")
    typer.echo(f"[audit] sim_v2: worlds_matrix={matrix_path}")

    sim_proj_df = _load_table(
        sim_proj_path,
        columns=["player_id", "play_prob", "status_bucket", "eligible_flag"],
    ) if sim_proj_path else pd.DataFrame()
    if not sim_proj_df.empty:
        sim_proj_df["player_id"] = sim_proj_df["player_id"].astype(str)
        sim_proj_df = sim_proj_df.rename(columns={"play_prob": "play_prob_stage6_sim_input"})

    if sim_dir is not None:
        manifest_path = sim_dir / "sim_manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            profile = str(manifest.get("profile") or manifest.get("sim_profile") or "unknown")
            masking = manifest.get("play_prob_masking")
            typer.echo(f"[audit] sim_manifest: profile={profile} play_prob_masking={masking}")
            try:
                cfg = load_sim_v2_profile(profile=profile)
                typer.echo(
                    f"[audit] sim_profile cfg: use_play_prob_masking={cfg.use_play_prob_masking} min_play_prob={cfg.min_play_prob}"
                )
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"[audit] sim_profile cfg: failed to load ({exc})", err=True)

    # -------------------- Join per-player stages --------------------
    base = None
    for frame in (minutes_df, proj_df, feat_df):
        if frame is not None and not frame.empty and "player_id" in frame.columns:
            base = frame.loc[:, ["player_id"]].drop_duplicates().copy()
            break
    if base is None:
        typer.echo("[audit] ERROR: could not find any per-player base frame (minutes/projections/features).", err=True)
        raise typer.Exit(code=2)

    merged = base.copy()
    stage_frames = [
        ("feat", feat_df),
        ("minutes", minutes_df),
        ("effective", eff_df),
        ("head", head_df),
        ("unified", proj_df),
        ("pool", pool_stage_df),
        ("sim", sim_proj_df),
    ]
    for stage, df in stage_frames:
        if df is None or df.empty:
            continue
        merged = _safe_merge(merged, df.drop_duplicates(subset=["player_id"]), on="player_id", suffix=f"_{stage}")

    # Fill identity columns from whichever stage has them.
    for col in ("player_name", "team_tricode", "team_abbr"):
        candidates = [c for c in merged.columns if c == col or c.startswith(f"{col}_")]
        for c in candidates:
            if col not in merged.columns:
                merged[col] = merged[c]
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[c])

    # Build sample selection from minutes output when possible.
    sample_pids = _select_players(minutes_df if not minutes_df.empty else merged, n_total=int(n_players))
    if not sample_pids:
        typer.echo("[audit] ERROR: could not select sample players (no minutes rows).", err=True)
        raise typer.Exit(code=2)

    sample = merged.loc[merged["player_id"].astype(str).isin(sample_pids)].copy()
    sample["status"] = sample.get("status", "").map(_normalize_status)

    # Add provenance labels.
    sample["play_prob_provenance"] = [
        _provenance_label(
            bundle_kind=bundle_kind,
            minutes_play_prob=float(p) if p is not None and pd.notna(p) else None,
            head_play_prob=float(h) if h is not None and pd.notna(h) else None,
            status=str(s) if s is not None else "",
        )
        for p, h, s in zip(
            sample.get("play_prob_stage2_minutes", pd.Series([np.nan] * len(sample))),
            sample.get("play_prob_stage3_lgbm_head_raw", pd.Series([np.nan] * len(sample))),
            sample.get("status", pd.Series([""] * len(sample))),
            strict=False,
        )
    ]

    # Worlds DNP rates for the sample players.
    if matrix_path and matrix_path.exists():
        worlds_sel = _load_worlds_selected(matrix_path=matrix_path, player_ids=sample["player_id"].astype(str).tolist())
        dnp = _compute_dnp_rates(worlds_sel)
        sample["worlds_dnp_rate"] = sample["player_id"].map(dnp)
    else:
        sample["worlds_dnp_rate"] = np.nan

    out_cols = [
        "player_id",
        "player_name",
        "team_tricode",
        "team_abbr",
        "status",
        "lineup_role",
        "is_out",
        "is_q",
        "is_prob",
        "starter_flag",
        "is_starter",
        "injury_row_present",
        "injury_snapshot_missing",
        "in_optimizer_pool",
        "play_prob_stage1_prior",
        "play_prob_stage2_minutes",
        "play_prob_stage2_effective",
        "play_prob_stage3_lgbm_head_raw",
        "play_prob_stage4_unified",
        "play_prob_stage5_pool",
        "play_prob_stage6_sim_input",
        "worlds_dnp_rate",
        "play_prob_provenance",
    ]
    out_cols = [c for c in out_cols if c in sample.columns]
    sample_out = sample.loc[:, out_cols].copy()

    typer.echo("\n## play_prob provenance (sample)")
    typer.echo(sample_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Sanity: Q/PROB players DNP rate vs (1 - play_prob)
    if "status" in sample_out.columns and "worlds_dnp_rate" in sample_out.columns:
        qmask = sample_out["status"].isin(["Q", "PROB"])
        if qmask.any() and "play_prob_stage6_sim_input" in sample_out.columns:
            sub = sample_out.loc[qmask, ["player_id", "status", "play_prob_stage6_sim_input", "worlds_dnp_rate"]].copy()
            sub["one_minus_play_prob"] = 1.0 - pd.to_numeric(sub["play_prob_stage6_sim_input"], errors="coerce")
            sub["dnp_minus_expected"] = pd.to_numeric(sub["worlds_dnp_rate"], errors="coerce") - sub["one_minus_play_prob"]
            typer.echo("\n## DNP masking check (Q/PROB sample)")
            typer.echo(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Optional: top150 P(any zero player) distribution for the slate using saved contest_sim build.
    builds_dir = root / "builds" / "contest_sim" / game_date
    if builds_dir.exists() and matrix_path and matrix_path.exists():
        build_files = [p for p in builds_dir.glob("*.json") if p.is_file()]
        build_path = None
        if build_files:
            if build_select == "latest":
                build_path = max(build_files, key=lambda p: p.stat().st_mtime)
            elif build_select == "largest":
                build_path = max(build_files, key=lambda p: p.stat().st_size)
            else:
                raise typer.BadParameter("build_select must be one of: latest, largest", param_name="build_select")
        if build_path is not None:
            payload = _read_json(build_path)
            results = payload.get("results") or []
            if isinstance(results, list) and results:
                select_scores = np.asarray(
                    [r.get("select_score") if r.get("select_score") is not None else float("-inf") for r in results],
                    dtype=np.float64,
                )
                order = np.argsort(-select_scores) if np.isfinite(select_scores).any() else np.arange(len(results))
                top_idx = order[: min(150, len(results))]

                lineups = [
                    [str(pid).strip() for pid in (results[int(i)].get("player_ids") or []) if str(pid).strip()]
                    for i in top_idx
                ]
                union_pids = sorted({pid for lu in lineups for pid in lu})
                worlds_cols = _load_worlds_selected(matrix_path=matrix_path, player_ids=union_pids)
                if not worlds_cols.empty:
                    pid_to_col = {pid: j for j, pid in enumerate(worlds_cols.columns)}
                    mat = worlds_cols.to_numpy(dtype=np.float64, copy=False)
                    rates: list[float] = []
                    for lu in lineups:
                        cols = [pid_to_col.get(pid) for pid in lu]
                        cols_i = np.asarray([int(c) for c in cols if c is not None], dtype=np.int64)
                        if cols_i.size == 0:
                            continue
                        sub = np.take(mat, cols_i, axis=1)
                        rates.append(float(np.mean(np.any(sub == 0.0, axis=1))))
                    if rates:
                        typer.echo("\n## Top150 P(any zero-score player) distribution (from saved contest_sim build)")
                        typer.echo(
                            f"mean={float(np.mean(rates)):.4f} "
                            f"p50={float(np.median(rates)):.4f} "
                            f"p90={float(np.percentile(rates, 90)):.4f} "
                            f"(build={build_path.name})"
                        )

    out_path = output_csv or Path(f"/tmp/audit_play_prob_provenance_{game_date}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_out.to_csv(out_path, index=False)
    typer.echo(f"\n[audit] wrote {len(sample_out)} rows to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
