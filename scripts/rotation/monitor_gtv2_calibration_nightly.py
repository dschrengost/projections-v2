#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.analyze_accuracy import parse_boxscores

ET_TZ = ZoneInfo("America/New_York")
RUN_TS_SUFFIX_RE = re.compile(r"(\d{8}T\d{6}Z)$")
_WORLD_CONTRACT_TOL = 1e-4


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    run_path: Path
    run_ts: pd.Timestamp | None
    game_ids: set[int]


def _resolve_default_date() -> str:
    return (datetime.now(tz=ET_TZ).date() - timedelta(days=1)).isoformat()


def _resolve_season_for_date(day: date) -> int:
    return day.year if day.month >= 10 else day.year - 1


def _parse_run_timestamp(run_id: str) -> pd.Timestamp | None:
    m = RUN_TS_SUFFIX_RE.search(str(run_id))
    if not m:
        return None
    try:
        return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))
    except ValueError:
        return None


def _load_actuals_with_tip(*, game_date: str, data_root: Path) -> pd.DataFrame:
    day = date.fromisoformat(game_date)
    season = _resolve_season_for_date(day)
    boxscore_path = (
        data_root
        / "bronze"
        / "boxscores_raw"
        / f"season={season}"
        / f"date={game_date}"
        / "boxscores_raw.parquet"
    )
    if not boxscore_path.exists():
        return pd.DataFrame()

    actuals = parse_boxscores(boxscore_path, game_date)
    if actuals.empty:
        return pd.DataFrame()

    tips = pd.read_parquet(boxscore_path, columns=["game_id", "tip_ts"]).copy()
    tips["game_id"] = pd.to_numeric(tips["game_id"], errors="coerce").astype("Int64")
    tips["tip_ts"] = pd.to_datetime(tips["tip_ts"], errors="coerce", utc=True)
    tips = tips.dropna(subset=["game_id", "tip_ts"]).drop_duplicates(subset=["game_id"], keep="last")

    actuals["game_id"] = pd.to_numeric(actuals["game_id"], errors="coerce").astype("Int64")
    actuals["player_id"] = pd.to_numeric(actuals["player_id"], errors="coerce").astype("Int64")
    actuals = actuals.dropna(subset=["game_id", "player_id"]).copy()
    actuals = actuals.merge(tips, on="game_id", how="left")
    return actuals


def _build_run_index(proj_paths: list[Path]) -> dict[str, RunMeta]:
    run_index: dict[str, RunMeta] = {}
    for proj_path in proj_paths:
        run_dir = proj_path.parent.name
        if not run_dir.startswith("run="):
            continue
        run_id = run_dir.split("=", 1)[1]
        try:
            run_df = pd.read_parquet(proj_path, columns=["game_id"])
        except Exception:
            continue
        gids = (
            pd.to_numeric(run_df["game_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        run_index[run_id] = RunMeta(
            run_id=run_id,
            run_path=proj_path,
            run_ts=_parse_run_timestamp(run_id),
            game_ids=set(gids),
        )
    return run_index


def _select_game_to_run(
    *,
    game_tips: dict[int, pd.Timestamp],
    run_index: dict[str, RunMeta],
) -> dict[int, str]:
    game_to_run: dict[int, str] = {}
    for game_id, tip_ts in game_tips.items():
        candidates: list[RunMeta] = [meta for meta in run_index.values() if game_id in meta.game_ids]
        if not candidates:
            continue

        pretip = [meta for meta in candidates if meta.run_ts is not None and meta.run_ts <= tip_ts]
        if pretip:
            best = max(pretip, key=lambda m: m.run_ts)
            game_to_run[game_id] = best.run_id
            continue

        with_ts = [meta for meta in candidates if meta.run_ts is not None]
        if with_ts:
            best = max(with_ts, key=lambda m: m.run_ts)
            game_to_run[game_id] = best.run_id
            continue

        # No parseable timestamps at all; fall back to lexical run_id ordering.
        game_to_run[game_id] = sorted([meta.run_id for meta in candidates])[-1]
    return game_to_run


def _load_selected_predictions(
    *,
    game_date: str,
    data_root: Path,
    actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, str], dict[str, Any]]:
    root = data_root / "artifacts" / "gtv2_worlds" / f"game_date={game_date}"
    proj_paths = sorted(root.glob("run=*/projections.parquet"))
    if not proj_paths:
        return pd.DataFrame(), {}, {"reason": "missing_projections"}

    run_index = _build_run_index(proj_paths)
    if not run_index:
        return pd.DataFrame(), {}, {"reason": "empty_run_index"}

    tips = (
        actuals.loc[:, ["game_id", "tip_ts"]]
        .dropna(subset=["game_id"])
        .drop_duplicates(subset=["game_id"], keep="last")
    )
    game_tips = {
        int(row.game_id): row.tip_ts
        for row in tips.itertuples(index=False)
        if pd.notna(row.tip_ts)
    }
    game_to_run = _select_game_to_run(game_tips=game_tips, run_index=run_index)

    if not game_to_run:
        return pd.DataFrame(), {}, {"reason": "no_game_run_matches", "runs_found": len(run_index)}

    runs_to_games: dict[str, list[int]] = {}
    for gid, rid in game_to_run.items():
        runs_to_games.setdefault(rid, []).append(int(gid))

    frames: list[pd.DataFrame] = []
    for run_id, game_ids in runs_to_games.items():
        run_meta = run_index.get(run_id)
        if run_meta is None:
            continue
        try:
            proj = pd.read_parquet(run_meta.run_path)
        except Exception:
            continue
        if "game_id" not in proj.columns:
            continue
        proj["game_id"] = pd.to_numeric(proj["game_id"], errors="coerce").astype("Int64")
        proj["player_id"] = pd.to_numeric(proj["player_id"], errors="coerce").astype("Int64")
        proj = proj[proj["game_id"].isin(game_ids)].copy()
        proj["snapshot_run_id"] = run_id
        frames.append(proj)

    if not frames:
        return pd.DataFrame(), game_to_run, {"reason": "selected_runs_not_readable", "selected_games": len(game_to_run)}

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["game_id", "player_id"]).copy()
    out = out.drop_duplicates(subset=["game_id", "player_id"], keep="last")

    meta = {
        "selected_games": int(len(game_to_run)),
        "selected_runs": int(len(runs_to_games)),
        "pretip_game_coverage": float(len(game_to_run) / max(1, len(game_tips))),
    }
    return out, game_to_run, meta


def _compute_has_props_by_player(features_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["game_id", "team_id", "player_id"]
    for c in key_cols:
        features_df[c] = pd.to_numeric(features_df[c], errors="coerce").astype("Int64")

    has_cols = sorted([c for c in features_df.columns if str(c).startswith("an_has_")])
    line_cols = sorted([c for c in features_df.columns if str(c).startswith("an_") and str(c).endswith("_line")])

    indicator_cols: list[str] = []
    if "an_props_market_count" in features_df.columns:
        indicator_cols.append("an_props_market_count")
    indicator_cols.extend(has_cols)
    indicator_cols.extend(line_cols)

    if not indicator_cols:
        out = features_df.loc[:, key_cols].dropna(subset=["game_id", "team_id", "player_id"]).drop_duplicates().copy()
        out["has_any_props"] = False
        return out

    feat = features_df.loc[:, key_cols + [c for c in indicator_cols if c in features_df.columns]].copy()
    feat = feat.dropna(subset=["game_id", "team_id", "player_id"]).copy()

    agg_dict: dict[str, str] = {}
    for col in feat.columns:
        if col in key_cols:
            continue
        if str(col).startswith("an_has_") or str(col) == "an_props_market_count":
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "first"

    feat = feat.groupby(key_cols, dropna=False, as_index=False).agg(agg_dict)

    has_any_props = np.zeros(len(feat), dtype=bool)
    has_explicit = False

    if "an_has_any_props" in feat.columns:
        has_explicit = True
        has_any_props |= (
            pd.to_numeric(feat["an_has_any_props"], errors="coerce").fillna(0.0).ge(0.5).to_numpy(dtype=bool)
        )
    if "an_props_market_count" in feat.columns:
        has_explicit = True
        has_any_props |= (
            pd.to_numeric(feat["an_props_market_count"], errors="coerce").fillna(0.0).ge(1.0).to_numpy(dtype=bool)
        )
    for col in has_cols:
        if col in feat.columns:
            has_explicit = True
            has_any_props |= (
                pd.to_numeric(feat[col], errors="coerce").fillna(0.0).ge(0.5).to_numpy(dtype=bool)
            )

    if not has_explicit:
        for col in line_cols:
            if col in feat.columns:
                has_any_props |= (
                    pd.to_numeric(feat[col], errors="coerce")
                    .fillna(0.0)
                    .abs()
                    .gt(float(_WORLD_CONTRACT_TOL))
                    .to_numpy(dtype=bool)
                )

    out = feat.loc[:, key_cols].copy()
    out["has_any_props"] = has_any_props
    return out


def _load_props_presence_overlay(
    *,
    game_date: str,
    data_root: Path,
    game_to_run: dict[int, str],
) -> pd.DataFrame:
    if not game_to_run:
        return pd.DataFrame(columns=["game_id", "player_id", "has_any_props"])

    runs_to_games: dict[str, list[int]] = {}
    for gid, rid in game_to_run.items():
        runs_to_games.setdefault(rid, []).append(gid)

    frames: list[pd.DataFrame] = []
    for run_id, game_ids in runs_to_games.items():
        feat_path = data_root / "live" / "features_gtv2_v1" / game_date / f"run={run_id}" / "features.parquet"
        if not feat_path.exists():
            continue
        try:
            feat = pd.read_parquet(feat_path)
        except Exception:
            continue
        if not {"game_id", "team_id", "player_id"}.issubset(feat.columns):
            continue

        feat["game_id"] = pd.to_numeric(feat["game_id"], errors="coerce").astype("Int64")
        feat = feat[feat["game_id"].isin(game_ids)].copy()
        if feat.empty:
            continue

        props = _compute_has_props_by_player(feat)
        frames.append(props)

    if not frames:
        return pd.DataFrame(columns=["game_id", "player_id", "has_any_props"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["game_id", "player_id"]).copy()
    out = out.groupby(["game_id", "player_id"], as_index=False)["has_any_props"].max()
    return out


def _safe_rate(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    return float(mask.mean())


def _compute_day_metrics(*, actuals: pd.DataFrame, projections: pd.DataFrame, props_overlay: pd.DataFrame) -> dict[str, Any]:
    if actuals.empty or projections.empty:
        return {
            "status": "missing_inputs",
            "n_actual_rows": int(len(actuals)),
            "n_projection_rows": int(len(projections)),
        }

    cols = [
        c
        for c in ["game_date", "game_id", "player_id", "minutes_mean", "dk_fpts_mean", "dk_fpts_p95", "dk_fpts_p05", "snapshot_run_id"]
        if c in projections.columns
    ]
    proj = projections.loc[:, cols].copy()
    proj["game_id"] = pd.to_numeric(proj["game_id"], errors="coerce").astype("Int64")
    proj["player_id"] = pd.to_numeric(proj["player_id"], errors="coerce").astype("Int64")
    proj = proj.dropna(subset=["game_id", "player_id"]).copy()

    merged = actuals.merge(
        proj,
        on=["game_id", "player_id"],
        how="inner",
        suffixes=("_actual", "_pred"),
    )
    if merged.empty:
        return {
            "status": "no_matched_rows",
            "n_actual_rows": int(len(actuals)),
            "n_projection_rows": int(len(proj)),
            "n_matched_rows": 0,
        }

    if not props_overlay.empty:
        merged = merged.merge(props_overlay, on=["game_id", "player_id"], how="left")
    merged["has_any_props"] = merged.get("has_any_props", False).fillna(False).astype(bool)

    actual = pd.to_numeric(merged["actual_dk_fpts"], errors="coerce")
    pred_mean = pd.to_numeric(merged.get("dk_fpts_mean"), errors="coerce")
    p95 = pd.to_numeric(merged.get("dk_fpts_p95"), errors="coerce")
    p05 = pd.to_numeric(merged.get("dk_fpts_p05"), errors="coerce")
    minutes_mean = pd.to_numeric(merged.get("minutes_mean"), errors="coerce")

    ok = actual.notna() & pred_mean.notna() & p95.notna() & p05.notna()
    eval_df = merged.loc[ok].copy()
    actual = actual.loc[ok]
    pred_mean = pred_mean.loc[ok]
    p95 = p95.loc[ok]
    p05 = p05.loc[ok]
    minutes_mean = minutes_mean.loc[ok]

    if eval_df.empty:
        return {
            "status": "no_valid_eval_rows",
            "n_matched_rows": int(len(merged)),
            "n_valid_rows": 0,
        }

    huge_mask = (
        pred_mean.abs().gt(1e6)
        | p95.abs().gt(1e6)
        | p05.abs().gt(1e6)
        | (~np.isfinite(pred_mean.to_numpy(dtype=float)))
        | (~np.isfinite(p95.to_numpy(dtype=float)))
        | (~np.isfinite(p05.to_numpy(dtype=float)))
    )

    over_p95 = actual > p95

    propless_mask = ~eval_df["has_any_props"].to_numpy(dtype=bool)
    m1220_mask = minutes_mean.ge(12.0).fillna(False) & minutes_mean.lt(20.0).fillna(False)

    metrics: dict[str, Any] = {
        "status": "ok",
        "n_matched_rows": int(len(merged)),
        "n_valid_rows": int(len(eval_df)),
        "fpts_mae": float((pred_mean - actual).abs().mean()),
        "fpts_bias": float((pred_mean - actual).mean()),
        "over_p95": float(over_p95.mean()),
        "under_p05": float((actual < p05).mean()),
        "huge_pred_rows": int(huge_mask.sum()),
        "huge_pred_rate": float(huge_mask.mean()),
        "propless_n": int(np.count_nonzero(propless_mask)),
        "propless_over_p95": _safe_rate(pd.Series(over_p95.to_numpy(dtype=bool)[propless_mask])),
        "minutes_12_20_n": int(np.count_nonzero(m1220_mask.to_numpy(dtype=bool))),
        "minutes_12_20_over_p95": _safe_rate(pd.Series(over_p95.to_numpy(dtype=bool)[m1220_mask.to_numpy(dtype=bool)])),
        "selected_runs": sorted(set(str(x) for x in eval_df.get("snapshot_run_id", pd.Series(dtype=str)).dropna().tolist())),
    }
    return metrics


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and "date" in row:
            out.append(row)
    return out


def _to_float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None


def _compute_alerts(
    *,
    today_row: dict[str, Any],
    history_rows: list[dict[str, Any]],
    baseline_window_days: int,
    drift_abs_band: float,
    drift_std_mult: float,
    min_bucket_n: int,
) -> tuple[list[str], dict[str, Any]]:
    alerts: list[str] = []
    diagnostics: dict[str, Any] = {}

    if int(today_row.get("huge_pred_rows", 0) or 0) > 0:
        alerts.append("huge_pred_rows")

    metrics = [
        ("propless_over_p95", "propless_n"),
        ("minutes_12_20_over_p95", "minutes_12_20_n"),
    ]

    day = date.fromisoformat(str(today_row["date"]))
    start = day - timedelta(days=int(max(1, baseline_window_days)))

    baseline_rows = [
        r for r in history_rows if "date" in r and start.isoformat() <= str(r["date"]) < day.isoformat()
    ]

    for metric_name, n_name in metrics:
        cur = _to_float_or_none(today_row.get(metric_name))
        cur_n = int(today_row.get(n_name, 0) or 0)
        if cur is None or cur_n < int(min_bucket_n):
            diagnostics[metric_name] = {
                "status": "insufficient_current_sample",
                "current": cur,
                "current_n": cur_n,
            }
            continue

        vals: list[float] = []
        for row in baseline_rows:
            v = _to_float_or_none(row.get(metric_name))
            n = int(row.get(n_name, 0) or 0)
            if v is None or n < int(min_bucket_n):
                continue
            vals.append(v)

        if len(vals) < 3:
            diagnostics[metric_name] = {
                "status": "insufficient_baseline",
                "current": cur,
                "current_n": cur_n,
                "baseline_n": len(vals),
            }
            continue

        arr = np.array(vals, dtype=float)
        base_mean = float(np.mean(arr))
        base_std = float(np.std(arr, ddof=0))
        limit = base_mean + max(float(drift_abs_band), float(drift_std_mult) * base_std)

        diagnostics[metric_name] = {
            "status": "ok",
            "current": cur,
            "current_n": cur_n,
            "baseline_mean": base_mean,
            "baseline_std": base_std,
            "baseline_n": int(len(vals)),
            "upper_limit": limit,
        }

        if cur > limit:
            alerts.append(f"{metric_name}_drift")

    return alerts, diagnostics


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly GTv2 calibration monitor")
    parser.add_argument("--date", type=str, default=_resolve_default_date(), help="Game date YYYY-MM-DD (default: yesterday ET)")
    parser.add_argument("--data-root", type=Path, default=Path("/home/daniel/projections-data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data_root>/reports/gtv2_calibration)",
    )
    parser.add_argument("--history-file", type=Path, default=None, help="History JSON path (default: <output_dir>/nightly_history.json)")
    parser.add_argument("--baseline-window-days", type=int, default=14)
    parser.add_argument("--drift-abs-band", type=float, default=0.01)
    parser.add_argument("--drift-std-mult", type=float, default=2.0)
    parser.add_argument("--min-bucket-n", type=int, default=100)
    parser.add_argument("--history-keep-days", type=int, default=120)
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    game_date = str(args.date)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else data_root / "reports" / "gtv2_calibration")
    history_path = Path(args.history_file).expanduser().resolve() if args.history_file else output_dir / "nightly_history.json"

    actuals = _load_actuals_with_tip(game_date=game_date, data_root=data_root)
    projections, game_to_run, selection_meta = _load_selected_predictions(
        game_date=game_date,
        data_root=data_root,
        actuals=actuals,
    )
    props_overlay = _load_props_presence_overlay(game_date=game_date, data_root=data_root, game_to_run=game_to_run)

    metrics = _compute_day_metrics(actuals=actuals, projections=projections, props_overlay=props_overlay)
    row: dict[str, Any] = {
        "date": game_date,
        "created_ts_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "n_actual_rows": int(len(actuals)),
        "n_projection_rows": int(len(projections)),
        **selection_meta,
        **metrics,
    }

    history = _load_history(history_path)
    by_date = {str(r["date"]): r for r in history}
    by_date[game_date] = row
    keep_start = (date.fromisoformat(game_date) - timedelta(days=int(max(1, args.history_keep_days)))).isoformat()
    history_new = [v for k, v in sorted(by_date.items()) if k >= keep_start]

    alerts, alert_diag = _compute_alerts(
        today_row=row,
        history_rows=history_new,
        baseline_window_days=int(args.baseline_window_days),
        drift_abs_band=float(args.drift_abs_band),
        drift_std_mult=float(args.drift_std_mult),
        min_bucket_n=int(args.min_bucket_n),
    )

    report = {
        "date": game_date,
        "alerts": alerts,
        "has_alerts": bool(alerts),
        "alert_diagnostics": alert_diag,
        "row": row,
    }

    _write_json(output_dir / f"gtv2_calibration_{game_date}.json", report)
    _write_json(output_dir / "gtv2_calibration_latest.json", report)
    _write_json(history_path, history_new)

    print(json.dumps(report, indent=2, sort_keys=True))

    if alerts and bool(args.fail_on_alert):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
