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


def _fetch_fpts_actuals_from_nba_com(*, game_ids: list[int], timeout: float = 10.0) -> pd.DataFrame:
    if not game_ids:
        return pd.DataFrame(columns=["game_date", "game_id", "player_id", "dk_fpts_actual"])

    from scrapers.nba_boxscore import NbaComBoxScoreScraper
    from projections.fpts_v2.scoring import compute_dk_fpts

    scraper = NbaComBoxScoreScraper(timeout=timeout, request_delay=0.0)
    rows: list[dict[str, Any]] = []
    for gid in sorted(set(int(x) for x in game_ids)):
        game = scraper.fetch_box_score(str(gid))
        if game is None:
            continue
        game_date = game.game_time_local.date() if game.game_time_local else None
        for team in (game.home, game.away):
            if team is None:
                continue
            for p in team.players:
                stats = p.statistics or {}
                rows.append(
                    {
                        "game_date": game_date,
                        "game_id": int(game.game_id),
                        "team_id": int(team.team_id),
                        "player_id": int(p.person_id),
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
    if not rows:
        return pd.DataFrame(columns=["game_date", "game_id", "player_id", "dk_fpts_actual"])

    df = pd.DataFrame(rows)
    df["dk_fpts_actual"] = compute_dk_fpts(df)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    return df[["game_date", "game_id", "player_id", "dk_fpts_actual"]].copy()


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


def _coverage_from_player_games(
    *,
    df: pd.DataFrame,
    value_name: str,
    actual_col: str,
    pred_prefix: str,
    qs: list[float],
    bucket_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in qs:
        q_label = int(round(q * 100))
        pred_col = f"{pred_prefix}_q{q_label:02d}"
        if pred_col not in df.columns or actual_col not in df.columns:
            continue
        pred = pd.to_numeric(df[pred_col], errors="coerce")
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        ok = pred.notna() & actual.notna()
        if ok.any():
            covered = (actual[ok] <= pred[ok]).astype(float)
            rows.append(
                {"value": value_name, "quantile": q, "bucket": "ALL", "n": int(ok.sum()), "coverage": float(covered.mean())}
            )
        for bucket, grp in df.loc[ok, [bucket_col, actual_col, pred_col]].groupby(bucket_col, dropna=False):
            if grp.empty:
                continue
            covered_b = (grp[actual_col] <= grp[pred_col]).astype(float)
            rows.append(
                {
                    "value": value_name,
                    "quantile": q,
                    "bucket": str(bucket) if bucket is not None and bucket == bucket else "NA",
                    "n": int(len(grp)),
                    "coverage": float(covered_b.mean()),
                }
            )
    return pd.DataFrame(rows)


def _minutes_bucket_stats(
    *,
    minutes_matrix: np.ndarray,  # (W, P)
    minutes_p50: np.ndarray,  # (P,)
) -> pd.DataFrame:
    """
    Aggregate minutes diagnostics by minutes_p50 bucket:
      - E[minutes | plays] where plays := (minutes > 0)
      - zero-mass rate := P(minutes == 0)
    """
    mins = np.asarray(minutes_matrix, dtype=float)
    if mins.ndim != 2 or mins.size == 0:
        return pd.DataFrame(
            columns=[
                "bucket",
                "n_player_games",
                "n_plays",
                "minutes_sum",
                "n_player_world_cells",
                "play_rate",
                "zero_mass_rate",
                "minutes_mean_conditional",
            ]
        )

    plays = mins > 0.0
    play_counts = plays.sum(axis=0).astype(float)  # (P,)
    minutes_sums = mins.sum(axis=0, dtype=float)  # (P,)

    bucket = _bucket_minutes_target(np.asarray(minutes_p50, dtype=float))
    rows: list[dict[str, Any]] = []
    w = float(mins.shape[0])
    for b in bucket.categories:
        idx = np.where(bucket == b)[0]
        if idx.size == 0:
            continue
        total_cells = w * float(idx.size)
        total_plays = float(play_counts[idx].sum())
        total_minutes = float(minutes_sums[idx].sum())
        minutes_mean_cond = (total_minutes / total_plays) if total_plays > 0 else float("nan")
        play_rate = (total_plays / total_cells) if total_cells > 0 else float("nan")
        zero_mass_rate = 1.0 - play_rate if np.isfinite(play_rate) else float("nan")
        rows.append(
            {
                "bucket": str(b),
                "n_player_games": int(idx.size),
                "n_plays": int(total_plays),
                "minutes_sum": float(total_minutes),
                "n_player_world_cells": int(total_cells),
                "play_rate": float(play_rate),
                "zero_mass_rate": float(zero_mass_rate),
                "minutes_mean_conditional": float(minutes_mean_cond),
            }
        )
    return pd.DataFrame(rows)


def _tail_coverage_from_player_games_split(
    *,
    df: pd.DataFrame,
    actual_col: str,
    pred_prefix: str,
    qs: list[float],
    bucket_col: str,
    split_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for q in qs:
        q_label = int(round(q * 100))
        pred_col = f"{pred_prefix}_q{q_label:02d}"
        if pred_col not in df.columns or actual_col not in df.columns or bucket_col not in df.columns or split_col not in df.columns:
            continue

        pred = pd.to_numeric(df[pred_col], errors="coerce")
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        split = df[split_col]
        bucket = df[bucket_col]
        ok = pred.notna() & actual.notna() & split.notna() & bucket.notna()
        if not ok.any():
            continue

        slim = pd.DataFrame(
            {split_col: split[ok], bucket_col: bucket[ok], actual_col: actual[ok], pred_col: pred[ok]}
        )
        for (split_v, bucket_v), grp in slim.groupby([split_col, bucket_col], dropna=False):
            if grp.empty:
                continue
            covered = (grp[actual_col] <= grp[pred_col]).astype(float)
            rows.append(
                {
                    "value": pred_prefix,
                    "quantile": q,
                    "bucket": str(bucket_v) if bucket_v is not None and bucket_v == bucket_v else "NA",
                    "split": str(split_v),
                    "n": int(len(grp)),
                    "coverage": float(covered.mean()),
                }
            )
    return pd.DataFrame(rows)


def _residual_corr_summary(
    *,
    df: pd.DataFrame,  # columns: game_id, team_id, residual
) -> dict[str, float | int]:
    """
    Estimate average residual correlation using a global-variance proxy:
      same-team:  E[r_i r_j] / Var(r) over pairs within the same (game_id, team_id)
      cross-team: E[r_i r_j] / Var(r) over pairs across opposing teams within the same game_id
    where residual := actual_fpts - sim_mean_fpts.
    """
    if df.empty:
        return {
            "same_team_resid_corr": float("nan"),
            "same_team_pairs": 0,
            "cross_team_resid_corr": float("nan"),
            "cross_team_pairs": 0,
            "resid_var": float("nan"),
            "resid_rows": 0,
        }

    resid = pd.to_numeric(df["residual"], errors="coerce")
    ok = resid.notna()
    slim = df.loc[ok, ["game_id", "team_id"]].copy()
    slim["residual"] = resid.loc[ok].astype(float)
    if slim.empty:
        return {
            "same_team_resid_corr": float("nan"),
            "same_team_pairs": 0,
            "cross_team_resid_corr": float("nan"),
            "cross_team_pairs": 0,
            "resid_var": float("nan"),
            "resid_rows": 0,
        }

    resid_var = float(slim["residual"].var(ddof=0))
    if not np.isfinite(resid_var) or resid_var < 1e-12:
        return {
            "same_team_resid_corr": float("nan"),
            "same_team_pairs": 0,
            "cross_team_resid_corr": float("nan"),
            "cross_team_pairs": 0,
            "resid_var": float(resid_var),
            "resid_rows": int(len(slim)),
        }

    slim = slim.copy()
    slim["residual2"] = slim["residual"] * slim["residual"]
    g = (
        slim.groupby(["game_id", "team_id"], as_index=False)
        .agg(sum_r=("residual", "sum"), sum_r2=("residual2", "sum"), n=("residual", "size"))
        .reset_index(drop=True)
    )

    # Same-team: sum_{i<j} r_i r_j = (S^2 - sum r_i^2)/2
    n = g["n"].to_numpy(dtype=float)
    sum_r = g["sum_r"].to_numpy(dtype=float)
    sum_r2 = g["sum_r2"].to_numpy(dtype=float)
    valid_team = n >= 2
    same_pairs = int((n[valid_team] * (n[valid_team] - 1.0) / 2.0).sum())
    same_prod = float(((sum_r[valid_team] * sum_r[valid_team] - sum_r2[valid_team]) / 2.0).sum())
    same_corr = (same_prod / float(same_pairs) / resid_var) if same_pairs > 0 else float("nan")

    # Cross-team within game: sum_{i in A, j in B} r_i r_j = S_A * S_B
    g2 = g.sort_values(["game_id", "team_id"]).copy()
    game_counts = g2.groupby("game_id")["team_id"].transform("size")
    g2 = g2.loc[game_counts == 2].copy()
    if not g2.empty:
        g2["row_in_game"] = g2.groupby("game_id").cumcount()
        a = g2[g2["row_in_game"] == 0].set_index("game_id")
        b = g2[g2["row_in_game"] == 1].set_index("game_id")
        common = a.index.intersection(b.index)
        a = a.loc[common]
        b = b.loc[common]
        cross_pairs = int((a["n"] * b["n"]).sum())
        cross_prod = float((a["sum_r"] * b["sum_r"]).sum())
    else:
        cross_pairs = 0
        cross_prod = 0.0
    cross_corr = (cross_prod / float(cross_pairs) / resid_var) if cross_pairs > 0 else float("nan")

    return {
        "same_team_resid_corr": float(same_corr),
        "same_team_pairs": int(same_pairs),
        "cross_team_resid_corr": float(cross_corr),
        "cross_team_pairs": int(cross_pairs),
        "resid_var": float(resid_var),
        "resid_rows": int(len(slim)),
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
    fetch_fpts_actuals: bool = typer.Option(
        True,
        "--fetch-fpts-actuals/--no-fetch-fpts-actuals",
        help="If local boxscore FPTS actuals are unavailable, fetch from NBA.com liveData endpoints.",
    ),
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
    meta["fpts_actuals_source"] = "raw_boxscores_json" if len(fpts_actuals_df) else "none"

    fetched_fpts_cache: dict[int, pd.DataFrame] = {}

    # Per-date audit accumulation.
    per_date_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    minutes_inv_summaries: list[MinutesInvariantSummary] = []
    player_game_cal_rows: list[dict[str, Any]] = []
    minutes_bucket_rows: list[dict[str, Any]] = []

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

        minutes_target_cond = _safe_float_series(proj_df, "minutes_mean", default=0.0)
        minutes_p50 = _safe_float_series(proj_df, "minutes_p50", default=minutes_target_cond)
        minutes_sim_mean_cond = _safe_float_series(proj_df, "minutes_sim_mean", default=0.0)
        drift_minutes = minutes_sim_mean_cond - minutes_target_cond

        fpts_target_cond = _safe_float_series(proj_df, "dk_fpts_mean_target", default=0.0)
        fpts_sim_mean_cond = _safe_float_series(proj_df, "dk_fpts_mean", default=0.0)
        drift_fpts = fpts_sim_mean_cond - fpts_target_cond

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
                    "minutes_target_cond": float(minutes_target_cond[i]),
                    "minutes_sim_mean_cond": float(minutes_sim_mean_cond[i]),
                    "minutes_drift_cond": float(drift_minutes[i]),
                    "dk_fpts_target_cond": float(fpts_target_cond[i]),
                    "dk_fpts_sim_mean_cond": float(fpts_sim_mean_cond[i]),
                    "dk_fpts_drift_cond": float(drift_fpts[i]),
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
        if fpts_slice.empty and fetch_fpts_actuals:
            # Fetch once per date, keyed by game_id.
            game_ids_for_date = sorted({int(x) for x in proj_df["game_id"].astype(int).tolist()})
            missing_gids = [gid for gid in game_ids_for_date if gid not in fetched_fpts_cache]
            if missing_gids:
                typer.echo(f"[audit_sim_v3] fetching NBA.com boxscores for {date_str} (n_games={len(missing_gids)})")
                fetched = _fetch_fpts_actuals_from_nba_com(game_ids=missing_gids)
                for gid in missing_gids:
                    fetched_fpts_cache[gid] = fetched[fetched["game_id"] == gid].copy()
                meta["fpts_actuals_source"] = "nba_com_fetch"
            fetched_for_date = pd.concat(
                [fetched_fpts_cache[gid] for gid in game_ids_for_date if gid in fetched_fpts_cache],
                ignore_index=True,
            )
            fpts_slice = fetched_for_date[fetched_for_date["game_date"] == d][
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

        bucket = _bucket_minutes_target(minutes_p50)
        for i, row in proj_df.reset_index(drop=True).iterrows():
            payload: dict[str, Any] = {
                "game_date": date_str,
                "game_id": int(row.get("game_id")),
                "team_id": int(row.get("team_id")),
                "player_id": str(row.get("player_id")),
                "minutes_bucket": str(bucket[i]) if bucket[i] == bucket[i] else "NA",
                "is_starter": int(float(row.get("is_starter", 0.0) or 0.0) > 0.5),
                "minutes_actual": float(minutes_actual[i]) if np.isfinite(minutes_actual[i]) else np.nan,
                "dk_fpts_actual": float(fpts_actual[i]) if np.isfinite(fpts_actual[i]) else np.nan,
            }
            for qi, q in enumerate(qs):
                q_label = int(round(q * 100))
                payload[f"minutes_q{q_label:02d}"] = float(minutes_q[qi, i])
                payload[f"dk_fpts_q{q_label:02d}"] = float(fpts_q[qi, i])
            player_game_cal_rows.append(payload)

        mb = _minutes_bucket_stats(minutes_matrix=minutes, minutes_p50=minutes_p50)
        if not mb.empty:
            mb = mb.copy()
            mb.insert(0, "game_date", date_str)
            minutes_bucket_rows.extend(mb.to_dict(orient="records"))

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

        resid_df = pd.DataFrame(
            {
                "game_id": game_id.astype(int, copy=False),
                "team_id": team_id.astype(int, copy=False),
                "residual": fpts_actual - fpts_sim_mean_cond,
            }
        )
        resid_corr = _residual_corr_summary(df=resid_df)
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
                **resid_corr,
            }
        )

    # Aggregate summaries
    per_date_df = pd.DataFrame(per_date_rows)
    drift_df = pd.DataFrame(drift_rows)
    corr_df = pd.DataFrame(corr_rows)
    cal_df = pd.DataFrame(player_game_cal_rows)
    minutes_bucket_df = pd.DataFrame(minutes_bucket_rows)
    coverage_df = pd.concat(
        [
            _coverage_from_player_games(
                df=cal_df,
                value_name="minutes",
                actual_col="minutes_actual",
                pred_prefix="minutes",
                qs=[0.50, 0.75, 0.90, 0.95, 0.99],
                bucket_col="minutes_bucket",
            ),
            _coverage_from_player_games(
                df=cal_df,
                value_name="dk_fpts",
                actual_col="dk_fpts_actual",
                pred_prefix="dk_fpts",
                qs=[0.50, 0.75, 0.90, 0.95, 0.99],
                bucket_col="minutes_bucket",
            ),
        ],
        ignore_index=True,
    ) if not cal_df.empty else pd.DataFrame(columns=["value", "quantile", "bucket", "n", "coverage"])

    fpts_tail_coverage_df = (
        _tail_coverage_from_player_games_split(
            df=cal_df,
            actual_col="dk_fpts_actual",
            pred_prefix="dk_fpts",
            qs=[0.90, 0.95, 0.99],
            bucket_col="minutes_bucket",
            split_col="is_starter",
        )
        if not cal_df.empty
        else pd.DataFrame(columns=["value", "quantile", "bucket", "split", "n", "coverage"])
    )
    if not fpts_tail_coverage_df.empty:
        fpts_tail_coverage_df = fpts_tail_coverage_df.copy()
        fpts_tail_coverage_df["split"] = fpts_tail_coverage_df["split"].map({"0": "bench", "1": "starter"}).fillna(
            fpts_tail_coverage_df["split"]
        )

    inv_max = max((s.max_abs_team_world_sum_err for s in minutes_inv_summaries), default=0.0)
    inv_neg = sum((s.count_negative_minutes for s in minutes_inv_summaries), 0)
    inv_total_tw = len(minutes_inv_summaries)

    worst20: list[dict[str, Any]] = []
    median_abs_drift_minutes = 0.0
    median_abs_drift_fpts = 0.0
    if not drift_df.empty:
        drift_df["abs_minutes_drift_cond"] = drift_df["minutes_drift_cond"].abs()
        drift_df["abs_fpts_drift_cond"] = drift_df["dk_fpts_drift_cond"].abs()
        median_abs_drift_minutes = float(drift_df["abs_minutes_drift_cond"].median())
        median_abs_drift_fpts = float(drift_df["abs_fpts_drift_cond"].median())
        worst = drift_df.sort_values(
            ["abs_minutes_drift_cond", "abs_fpts_drift_cond"], ascending=False
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
            "target_minutes_cond": "minutes_mean (input) vs minutes_sim_mean (sim)",
            "target_fpts_cond": "dk_fpts_mean_target (input) vs dk_fpts_mean (sim)",
            "median_abs_drift_minutes_cond": median_abs_drift_minutes,
            "median_abs_drift_fpts_cond": median_abs_drift_fpts,
            "top_20_worst_player_games": worst20,
        },
        "per_date": per_date_rows,
        "minutes_bucket": (
            (
                lambda agg: agg.assign(
                    play_rate=agg["n_plays"] / agg["n_player_world_cells"],
                    zero_mass_rate=1.0 - (agg["n_plays"] / agg["n_player_world_cells"]),
                    minutes_mean_conditional=agg["minutes_sum"] / agg["n_plays"],
                )
                .replace([np.inf, -np.inf], np.nan)
                .to_dict(orient="records")
            )(
                minutes_bucket_df.drop(columns=["game_date"], errors="ignore")
                .groupby("bucket", as_index=False)
                .agg(
                    n_player_games=("n_player_games", "sum"),
                    n_player_world_cells=("n_player_world_cells", "sum"),
                    n_plays=("n_plays", "sum"),
                    minutes_sum=("minutes_sum", "sum"),
                )
            )
            if not minutes_bucket_df.empty
            else []
        ),
        "fpts_tail_coverage": fpts_tail_coverage_df.to_dict(orient="records") if not fpts_tail_coverage_df.empty else [],
        "notes": {
            "minutes_target": "minutes_mean vs minutes_sim_mean (conditional-on-active means)",
            "fpts_target": "dk_fpts_mean_target vs dk_fpts_mean (conditional-on-active means)",
            "quantile_coverage": "P(actual <= sim_quantile) by minutes_p50 bucket",
            "minutes_bucket": "E[minutes | plays] and P(minutes == 0) by minutes_p50 bucket",
            "correlation": "centered sim-world corr + residual corr (actual_fpts - sim_mean_fpts)",
        },
    }

    (audit_dir / "sim_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    per_date_df.to_csv(audit_dir / "per_date_summary.csv", index=False)
    corr_df.to_csv(audit_dir / "correlation_summary.csv", index=False)
    coverage_df.to_csv(audit_dir / "quantile_coverage.csv", index=False)
    if not minutes_bucket_df.empty:
        minutes_bucket_df.to_csv(audit_dir / "minutes_bucket_stats_per_date.csv", index=False)
    if not fpts_tail_coverage_df.empty:
        fpts_tail_coverage_df.to_csv(audit_dir / "fpts_tail_coverage_by_bucket_starter.csv", index=False)
    if not drift_df.empty:
        worst = drift_df.sort_values(["abs_minutes_drift_cond", "abs_fpts_drift_cond"], ascending=False).head(20)
        worst.to_csv(audit_dir / "mean_drift_top20.csv", index=False)

    typer.echo(f"[audit_sim_v3] Wrote audit bundle: {audit_dir}")


if __name__ == "__main__":
    app()
