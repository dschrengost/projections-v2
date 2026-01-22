#!/usr/bin/env python
"""Audit harness for sim_v3 (sim_v2 engine).

Runs the sim for a date range and produces a single JSON + CSV bundle under:
  reports/sim_audit/<tag>/

Usage:
  uv run python scripts/sim_v2/audit_sim_v3.py --date-from 2026-01-01 --date-to 2026-01-03 --profile-name sim_v3 --num-worlds 500 --out-dir reports/sim_audit/local_20260101_20260103
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main

app = typer.Typer(add_completion=False)

TEAM_MINUTES_TARGET = 240.0
MINUTES_CAP_SIM_V3 = 41.0


def _parse_date(d: str) -> date_cls:
    return pd.Timestamp(d).date()


def _date_range(d0: date_cls, d1: date_cls) -> Iterable[date_cls]:
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def _git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root))
            return out.decode("utf-8").strip()
        except Exception:
            return ""

    return {
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "branch", "--show-current"]),
        "git_dirty": _run(["git", "status", "--porcelain=v1"]) != "",
    }


def _season_from_date(d: date_cls) -> int:
    # NBA season label convention in this repo: Jan..Jun belong to season=year; Oct..Dec belong to season=year+1.
    return d.year + 1 if d.month >= 10 else d.year


def _boxscores_raw_path(data_root: Path, season: int) -> Path | None:
    # season=2026 -> nba_boxscores_2025_26.json
    prev = season - 1
    suffix = str(season)[-2:]
    candidates = [
        data_root / "raw" / f"nba_boxscores_{prev}_{suffix}.json",
        data_root / "raw_from_minutes_v0" / f"nba_boxscores_{prev}_{suffix}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


_DUR_RE = re.compile(r"^PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$")


def _parse_duration_minutes(value: Any) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    m = _DUR_RE.match(s)
    if not m:
        return 0.0
    h = float(m.group("h") or 0.0)
    mm = float(m.group("m") or 0.0)
    sec = float(m.group("s") or 0.0)
    return h * 60.0 + mm + sec / 60.0


def _load_actuals_from_raw_boxscores(
    *, data_root: Path, date_from: date_cls, date_to: date_cls
) -> pd.DataFrame:
    seasons = sorted({_season_from_date(d) for d in _date_range(date_from, date_to)})
    out_frames: list[pd.DataFrame] = []

    for season in seasons:
        path = _boxscores_raw_path(data_root, season)
        if path is None:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for game in raw:
            try:
                game_time_local = game.get("game_time_local") or game.get("game_time_utc")
                game_date = pd.Timestamp(game_time_local).date()
            except Exception:
                continue
            if game_date < date_from or game_date > date_to:
                continue

            game_id = int(game["game_id"])
            for side in ("home", "away"):
                team = game.get(side, {}) or {}
                team_id = int(team.get("team_id")) if team.get("team_id") is not None else None
                players = team.get("players", []) or []
                for p in players:
                    stats = (p.get("statistics", {}) or {}) if isinstance(p, dict) else {}
                    rows.append(
                        {
                            "game_date": game_date,
                            "game_id": game_id,
                            "team_id": team_id,
                            "player_id": int(p.get("person_id")),
                            "minutes_actual": _parse_duration_minutes(
                                stats.get("minutesCalculated") or stats.get("minutes") or "PT0M0S"
                            ),
                            "pts": float(stats.get("points") or 0.0),
                            "reb": float(stats.get("reboundsTotal") or 0.0),
                            "oreb": float(stats.get("reboundsOffensive") or 0.0),
                            "dreb": float(stats.get("reboundsDefensive") or 0.0),
                            "ast": float(stats.get("assists") or 0.0),
                            "stl": float(stats.get("steals") or 0.0),
                            "blk": float(stats.get("blocks") or 0.0),
                            "tov": float(stats.get("turnovers") or 0.0),
                            "fgm": float(stats.get("fieldGoalsMade") or 0.0),
                            "fga": float(stats.get("fieldGoalsAttempted") or 0.0),
                            "fg3m": float(stats.get("threePointersMade") or 0.0),
                            "fg3a": float(stats.get("threePointersAttempted") or 0.0),
                            "ftm": float(stats.get("freeThrowsMade") or 0.0),
                            "fta": float(stats.get("freeThrowsAttempted") or 0.0),
                            "pf": float(stats.get("foulsPersonal") or 0.0),
                            "plus_minus": float(stats.get("plusMinusPoints") or 0.0),
                        }
                    )

        if rows:
            df = pd.DataFrame(rows)
            from projections.fpts_v2.scoring import compute_dk_fpts

            df["dk_fpts_actual"] = compute_dk_fpts(df)
            out_frames.append(df[["game_date", "game_id", "team_id", "player_id", "minutes_actual", "dk_fpts_actual"]])

    if not out_frames:
        return pd.DataFrame(columns=["game_date", "game_id", "team_id", "player_id", "minutes_actual", "dk_fpts_actual"])
    return pd.concat(out_frames, ignore_index=True)


def _load_minutes_actuals_from_labels(*, data_root: Path, date_from: date_cls, date_to: date_cls) -> pd.DataFrame:
    seasons = sorted({_season_from_date(d) for d in _date_range(date_from, date_to)})
    out_frames: list[pd.DataFrame] = []
    for season in seasons:
        path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["game_date", "game_id", "player_id", "minutes"])
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        mask = (df["game_date"] >= date_from) & (df["game_date"] <= date_to)
        df = df.loc[mask, ["game_date", "game_id", "player_id", "minutes"]].copy()
        df = df.rename(columns={"minutes": "minutes_actual"})
        df["player_id"] = df["player_id"].astype(int)
        df["game_id"] = df["game_id"].astype(int)
        out_frames.append(df)
    if not out_frames:
        return pd.DataFrame(columns=["game_date", "game_id", "player_id", "minutes_actual"])
    return pd.concat(out_frames, ignore_index=True)


@dataclass(frozen=True)
class MinutesInvariantSummary:
    max_abs_team_world_sum_err: float
    pct_team_world_within_eps: float
    count_negative_minutes: int
    count_player_worlds_hit_cap_41: int
    count_unique_players_hit_cap_41: int


def _minutes_invariant(
    *, proj_df: pd.DataFrame, minutes_matrix: np.ndarray, eps: float = 1e-3, cap: float = MINUTES_CAP_SIM_V3
) -> MinutesInvariantSummary:
    if minutes_matrix.size == 0:
        return MinutesInvariantSummary(
            max_abs_team_world_sum_err=0.0,
            pct_team_world_within_eps=1.0,
            count_negative_minutes=0,
            count_player_worlds_hit_cap_41=0,
            count_unique_players_hit_cap_41=0,
        )

    group_map: dict[tuple[int, int], list[int]] = {}
    # minutes_matrix columns are emitted in the same player order as projections.parquet rows.
    for i, (gid, tid) in enumerate(
        zip(
            proj_df["game_id"].astype(int).to_numpy(),
            proj_df["team_id"].astype(int).to_numpy(),
        )
    ):
        group_map.setdefault((int(gid), int(tid)), []).append(int(i))

    errs: list[np.ndarray] = []
    for _, idxs in group_map.items():
        team_sum = minutes_matrix[:, idxs].sum(axis=1, dtype=float)
        errs.append(np.abs(team_sum - TEAM_MINUTES_TARGET))
    err_all = np.concatenate(errs) if errs else np.zeros(0, dtype=float)

    max_err = float(err_all.max()) if err_all.size else 0.0
    within = float((err_all <= float(eps)).mean()) if err_all.size else 1.0

    neg_count = int((minutes_matrix < -1e-9).sum())
    hit_cap_mask = np.isclose(minutes_matrix, float(cap), rtol=0.0, atol=1e-4)
    hit_cap_cells = int(hit_cap_mask.sum())
    hit_cap_unique_players = int(np.any(hit_cap_mask, axis=0).sum())

    return MinutesInvariantSummary(
        max_abs_team_world_sum_err=max_err,
        pct_team_world_within_eps=within,
        count_negative_minutes=neg_count,
        count_player_worlds_hit_cap_41=hit_cap_cells,
        count_unique_players_hit_cap_41=hit_cap_unique_players,
    )


def _safe_float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def _empirical_means(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0, dtype=float) if x.size else np.zeros((x.shape[1],), dtype=float)


def _empirical_quantiles(x: np.ndarray, qs: list[float]) -> np.ndarray:
    if x.size == 0:
        return np.zeros((len(qs), x.shape[1]), dtype=float)
    return np.quantile(x, qs, axis=0, method="linear").astype(float)


def _bucket_minutes_target(minutes_target: np.ndarray) -> pd.Categorical:
    bins = [0.0, 8.0, 16.0, 24.0, 32.0, 48.0, float("inf")]
    labels = ["[0,8)", "[8,16)", "[16,24)", "[24,32)", "[32,48)", "[48,inf)"]
    return pd.cut(minutes_target, bins=bins, labels=labels, right=False, include_lowest=True)


def _coverage_table(
    *,
    actual: np.ndarray,
    q_preds: np.ndarray,
    qs: list[float],
    bucket: pd.Categorical,
    value_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qi, q in enumerate(qs):
        pred = q_preds[qi]
        ok = np.isfinite(pred) & np.isfinite(actual)
        if not ok.any():
            continue
        covered = (actual[ok] <= pred[ok]).astype(float)
        rows.append(
            {
                "value": value_name,
                "quantile": q,
                "bucket": "ALL",
                "n": int(ok.sum()),
                "coverage": float(covered.mean()),
            }
        )
        for b in bucket.categories:
            mask_b = ok & (bucket == b)
            if not mask_b.any():
                continue
            covered_b = (actual[mask_b] <= pred[mask_b]).astype(float)
            rows.append(
                {
                    "value": value_name,
                    "quantile": q,
                    "bucket": str(b),
                    "n": int(mask_b.sum()),
                    "coverage": float(covered_b.mean()),
                }
            )
    return pd.DataFrame(rows)


def _sample_pair_corr(
    *,
    x: np.ndarray,  # (W, P) centered residuals
    group_key: np.ndarray,  # (P,) group id used to decide "same"
    alt_key: np.ndarray | None = None,  # (P,) optional second key (e.g., game_id) to restrict pairs
    mode: str,  # "same" | "diff"
    n_pairs: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    n_worlds, n_players = x.shape
    if n_players < 2 or n_worlds < 2:
        return {"n_pairs": 0, "mean": float("nan"), "std": float("nan")}

    std = x.std(axis=0, ddof=0)
    valid = std > 1e-9
    valid_idx = np.where(valid)[0]
    if valid_idx.size < 2:
        return {"n_pairs": 0, "mean": float("nan"), "std": float("nan")}

    rng = np.random.default_rng(seed)
    corrs: list[float] = []
    attempts = 0
    max_attempts = max(10_000, n_pairs * 20)

    while len(corrs) < n_pairs and attempts < max_attempts:
        i, j = rng.choice(valid_idx, size=2, replace=False)
        if alt_key is not None and alt_key[i] != alt_key[j]:
            attempts += 1
            continue
        same = group_key[i] == group_key[j]
        if (mode == "same" and not same) or (mode == "diff" and same):
            attempts += 1
            continue
        xi = x[:, i]
        xj = x[:, j]
        cov = float(np.dot(xi, xj) / float(n_worlds))
        corr = cov / float(std[i] * std[j])
        if np.isfinite(corr):
            corrs.append(float(corr))
        attempts += 1

    if not corrs:
        return {"n_pairs": 0, "mean": float("nan"), "std": float("nan")}

    arr = np.asarray(corrs, dtype=float)
    return {"n_pairs": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _team_total_corr(
    *,
    worlds: np.ndarray,  # (W, P)
    game_id: np.ndarray,  # (P,)
    team_id: np.ndarray,  # (P,)
) -> dict[str, float | int]:
    # Correlation between opposing team totals within each game, then averaged.
    rows: list[float] = []
    for gid in np.unique(game_id):
        mask_g = game_id == gid
        teams = np.unique(team_id[mask_g])
        if teams.size != 2:
            continue
        t0, t1 = teams[0], teams[1]
        tot0 = worlds[:, mask_g & (team_id == t0)].sum(axis=1, dtype=float)
        tot1 = worlds[:, mask_g & (team_id == t1)].sum(axis=1, dtype=float)
        if tot0.std(ddof=0) < 1e-9 or tot1.std(ddof=0) < 1e-9:
            continue
        corr = float(np.corrcoef(tot0, tot1)[0, 1])
        if np.isfinite(corr):
            rows.append(corr)
    arr = np.asarray(rows, dtype=float)
    return {
        "n_games": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
    }


@app.command()
def main(
    date_from: str = typer.Option(..., "--date-from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--date-to", help="End date (YYYY-MM-DD)"),
    profile_name: str = typer.Option("sim_v3", "--profile-name", help="Sim profile name (e.g., sim_v3)"),
    num_worlds: int = typer.Option(500, "--num-worlds", help="Worlds per date"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for JSON/CSV bundle"),
    data_root: Path | None = typer.Option(None, "--data-root", help="Data root (default: PROJECTIONS_DATA_ROOT or ./data)"),
    profiles_path: Path | None = typer.Option(None, "--profiles-path", help="Override sim profiles JSON path"),
    seed: int = typer.Option(0, "--seed", help="RNG seed for sim reproducibility"),
    sim_run_id: str | None = typer.Option(None, "--sim-run-id", help="Explicit sim run_id to write under artifacts/"),
) -> None:
    d0 = _parse_date(date_from)
    d1 = _parse_date(date_to)
    if d1 < d0:
        raise typer.BadParameter("--date-to must be >= --date-from")

    root = (data_root or data_path()).resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_from": d0.isoformat(),
        "date_to": d1.isoformat(),
        "profile_name": profile_name,
        "num_worlds": int(num_worlds),
        "seed": int(seed),
        "data_root": str(root),
        **_git_info(repo_root),
    }

    run_id = sim_run_id or f"audit_w{num_worlds}"
    meta["sim_run_id"] = run_id
    meta["sim_output_root"] = str(out_dir)

    # Ensure minutes matrix is persisted for invariant checks.
    os.environ["PROJECTIONS_SIM_WRITE_MINUTES_MATRIX"] = "1"

    typer.echo(f"[audit_sim_v3] Running sim: {d0}..{d1} profile={profile_name} worlds={num_worlds} run_id={run_id}")
    generate_worlds_main(
        start_date=d0.isoformat(),
        end_date=d1.isoformat(),
        n_worlds=num_worlds,
        profile=profile_name,
        data_root=root,
        profiles_path=profiles_path,
        output_root=out_dir,
        sim_run_id=run_id,
        use_rates_noise=None,
        rates_noise_split=None,
        team_sigma_scale=None,
        player_sigma_scale=None,
        rates_run_id=None,
        minutes_run_id=None,
        use_minutes_noise=None,
        minutes_noise_run_id=None,
        minutes_sigma_min=None,
        seed=seed,
        min_play_prob=None,
        team_factor_sigma=None,
        team_factor_gamma=None,
        use_efficiency_scoring=None,
        export_attempt_means=False,
    )

    typer.echo("[audit_sim_v3] Loading actuals (raw boxscores) ...")
    fpts_actuals_df = _load_actuals_from_raw_boxscores(data_root=root, date_from=d0, date_to=d1)
    minutes_actuals_df = _load_minutes_actuals_from_labels(data_root=root, date_from=d0, date_to=d1)
    meta["minutes_actuals_rows"] = int(len(minutes_actuals_df))
    meta["fpts_actuals_rows"] = int(len(fpts_actuals_df))

    # Per-date audit accumulation.
    per_date_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []
    corr_rows: list[dict[str, Any]] = []
    minutes_inv_summaries: list[MinutesInvariantSummary] = []

    for d in _date_range(d0, d1):
        date_str = d.isoformat()
        run_dir = out_dir / f"game_date={date_str}" / f"run={run_id}"
        proj_path = run_dir / "projections.parquet"
        minutes_path = run_dir / "minutes_matrix.parquet"
        worlds_path = run_dir / "worlds_matrix.parquet"

        if not proj_path.exists():
            typer.echo(f"[audit_sim_v3] WARNING: missing projections.parquet for {date_str} ({proj_path})", err=True)
            continue
        if not minutes_path.exists():
            raise FileNotFoundError(f"minutes_matrix.parquet not found for {date_str}: {minutes_path}")
        if not worlds_path.exists():
            raise FileNotFoundError(f"worlds_matrix.parquet not found for {date_str}: {worlds_path}")

        proj_df = pd.read_parquet(proj_path)
        minutes_df = pd.read_parquet(minutes_path)
        worlds_df = pd.read_parquet(worlds_path)

        minutes = minutes_df.to_numpy(dtype=float, copy=False)
        worlds = worlds_df.to_numpy(dtype=float, copy=False)

        inv = _minutes_invariant(proj_df=proj_df, minutes_matrix=minutes, eps=1e-3, cap=MINUTES_CAP_SIM_V3)
        minutes_inv_summaries.append(inv)

        if "play_prob" in proj_df.columns:
            play_prob = _safe_float_series(proj_df, "play_prob", default=1.0)
        else:
            play_prob = _safe_float_series(proj_df, "sim_p_active", default=1.0)
        minutes_target_cond = _safe_float_series(proj_df, "minutes_mean", default=0.0)
        fpts_target_cond = _safe_float_series(proj_df, "dk_fpts_mean", default=0.0)

        minutes_mean_uncond = _empirical_means(minutes)
        fpts_mean_uncond = _empirical_means(worlds)

        minutes_target_uncond = minutes_target_cond * play_prob
        fpts_target_uncond = fpts_target_cond * play_prob

        drift_minutes = minutes_mean_uncond - minutes_target_uncond
        drift_fpts = fpts_mean_uncond - fpts_target_uncond

        per_date_rows.append(
            {
                "game_date": date_str,
                **asdict(inv),
                "median_abs_drift_minutes": float(np.median(np.abs(drift_minutes))) if drift_minutes.size else 0.0,
                "median_abs_drift_fpts": float(np.median(np.abs(drift_fpts))) if drift_fpts.size else 0.0,
            }
        )

        # Keep per-player drift rows for top-N extraction across the window.
        for i, row in proj_df.reset_index(drop=True).iterrows():
            drift_rows.append(
                {
                    "game_date": date_str,
                    "game_id": int(row.get("game_id")),
                    "team_id": int(row.get("team_id")),
                    "player_id": str(row.get("player_id")),
                    "minutes_target_uncond": float(minutes_target_uncond[i]),
                    "minutes_sim_mean_uncond": float(minutes_mean_uncond[i]),
                    "minutes_drift_uncond": float(drift_minutes[i]),
                    "fpts_target_uncond": float(fpts_target_uncond[i]),
                    "fpts_sim_mean_uncond": float(fpts_mean_uncond[i]),
                    "fpts_drift_uncond": float(drift_fpts[i]),
                }
            )

        # Quantile coverage vs actuals (minutes + fpts)
        join_keys = ["game_id", "player_id"]
        minutes_slice = minutes_actuals_df[minutes_actuals_df["game_date"] == d][
            ["game_id", "player_id", "minutes_actual"]
        ].copy()
        minutes_slice["player_id"] = minutes_slice["player_id"].astype(str)

        fpts_slice = fpts_actuals_df[fpts_actuals_df["game_date"] == d][
            ["game_id", "player_id", "dk_fpts_actual"]
        ].copy()
        fpts_slice["player_id"] = fpts_slice["player_id"].astype(str)

        proj_join = proj_df[["game_id", "player_id"]].copy()
        proj_join["player_id"] = proj_join["player_id"].astype(str)
        joined = proj_join.merge(minutes_slice, on=join_keys, how="left").merge(fpts_slice, on=join_keys, how="left")

        minutes_actual = pd.to_numeric(joined["minutes_actual"], errors="coerce").to_numpy(dtype=float)
        fpts_actual = pd.to_numeric(joined["dk_fpts_actual"], errors="coerce").to_numpy(dtype=float)

        qs = [0.50, 0.75, 0.90, 0.95, 0.99]
        minutes_q = _empirical_quantiles(minutes, qs)
        fpts_q = _empirical_quantiles(worlds, qs)

        bucket = _bucket_minutes_target(minutes_target_cond)
        coverage_frames.append(
            _coverage_table(actual=minutes_actual, q_preds=minutes_q, qs=qs, bucket=bucket, value_name="minutes")
        )
        coverage_frames.append(
            _coverage_table(actual=fpts_actual, q_preds=fpts_q, qs=qs, bucket=bucket, value_name="dk_fpts")
        )

        # Correlations: residual pairs and game total proxy.
        centered = worlds - worlds.mean(axis=0, dtype=float, keepdims=True)
        team_id = proj_df["team_id"].astype(int).to_numpy()
        game_id = proj_df["game_id"].astype(int).to_numpy()

        same_team = _sample_pair_corr(
            x=centered, group_key=team_id, alt_key=game_id, mode="same", n_pairs=2000, seed=seed
        )
        cross_team = _sample_pair_corr(
            x=centered, group_key=team_id, alt_key=game_id, mode="diff", n_pairs=2000, seed=seed
        )
        team_tot = _team_total_corr(worlds=worlds, game_id=game_id, team_id=team_id)
        corr_rows.append(
            {
                "game_date": date_str,
                "same_team_offdiag_mean": same_team["mean"],
                "same_team_offdiag_std": same_team["std"],
                "same_team_pairs": same_team["n_pairs"],
                "cross_team_offdiag_mean": cross_team["mean"],
                "cross_team_offdiag_std": cross_team["std"],
                "cross_team_pairs": cross_team["n_pairs"],
                "game_team_total_corr_mean": team_tot["mean"],
                "game_team_total_corr_std": team_tot["std"],
                "game_team_total_corr_games": team_tot["n_games"],
            }
        )

    # Aggregate summaries
    per_date_df = pd.DataFrame(per_date_rows)
    drift_df = pd.DataFrame(drift_rows)
    corr_df = pd.DataFrame(corr_rows)
    coverage_df = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame(
        columns=["value", "quantile", "bucket", "n", "coverage"]
    )

    inv_max = max((s.max_abs_team_world_sum_err for s in minutes_inv_summaries), default=0.0)
    inv_neg = sum((s.count_negative_minutes for s in minutes_inv_summaries), 0)
    inv_total_tw = len(minutes_inv_summaries)

    worst20: list[dict[str, Any]] = []
    median_abs_drift_minutes = 0.0
    median_abs_drift_fpts = 0.0
    if not drift_df.empty:
        drift_df["abs_minutes_drift_uncond"] = drift_df["minutes_drift_uncond"].abs()
        drift_df["abs_fpts_drift_uncond"] = drift_df["fpts_drift_uncond"].abs()
        median_abs_drift_minutes = float(drift_df["abs_minutes_drift_uncond"].median())
        median_abs_drift_fpts = float(drift_df["abs_fpts_drift_uncond"].median())
        worst = drift_df.sort_values(
            ["abs_minutes_drift_uncond", "abs_fpts_drift_uncond"], ascending=False
        ).head(20)
        worst20 = worst.to_dict(orient="records")

    report = {
        "meta": meta,
        "minutes_invariant": {
            "max_abs_team_world_sum_err": float(inv_max),
            "count_negative_minutes": int(inv_neg),
            "n_dates": int(inv_total_tw),
        },
        "mean_preservation": {
            "target_minutes_uncond": "minutes_mean * play_prob",
            "target_fpts_uncond": "dk_fpts_mean * play_prob",
            "median_abs_drift_minutes_uncond": median_abs_drift_minutes,
            "median_abs_drift_fpts_uncond": median_abs_drift_fpts,
            "top_20_worst_player_games": worst20,
        },
        "per_date": per_date_rows,
        "notes": {
            "minutes_target": "minutes_mean * play_prob (unconditional target)",
            "fpts_target": "dk_fpts_mean * play_prob (unconditional target)",
            "quantile_coverage": "P(actual <= sim_quantile) by minutes_mean bucket (unconditional)",
            "correlation": "pairwise corr on centered FPTS worlds; pairs restricted within same game",
        },
    }

    (audit_dir / "sim_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    per_date_df.to_csv(audit_dir / "per_date_summary.csv", index=False)
    corr_df.to_csv(audit_dir / "correlation_summary.csv", index=False)
    coverage_df.to_csv(audit_dir / "quantile_coverage.csv", index=False)
    if not drift_df.empty:
        worst = drift_df.sort_values(["abs_minutes_drift_uncond", "abs_fpts_drift_uncond"], ascending=False).head(20)
        worst.to_csv(audit_dir / "mean_drift_top20.csv", index=False)

    typer.echo(f"[audit_sim_v3] Wrote audit bundle: {audit_dir}")


if __name__ == "__main__":
    app()
