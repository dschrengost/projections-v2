"""Evaluate sim_v2 scoring calibration vs actual box scores.

Example:
uv run python -m scripts.sim_v2.eval_scoring_calibration \
  --start-date 2025-12-01 \
  --end-date 2025-12-05 \
  --profile rates_v0 \
  --data-root /home/daniel/projections-data
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import typer

from projections.sim_v2.config import DEFAULT_PROFILES_PATH, load_sim_v2_profile
from projections.fpts_v2.scoring import compute_dk_fpts

app = typer.Typer(add_completion=False)


def _daterange(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    return pd.date_range(start, end, freq="D")


def _season_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def load_worlds(
    root: Path, start: pd.Timestamp, end: pd.Timestamp, *, profile: str | None = None
) -> pd.DataFrame:
    base = root / "artifacts" / "sim_v2" / "worlds_fpts_v2"
    frames: list[pd.DataFrame] = []
    wanted_cols = {
        "game_date",
        "player_id",
        "dk_fpts_world",
        "pts_world",
        "minutes_sim",
        "sim_profile",
        "minutes_p50",
        "minutes_mean",
        "fga2_sim",
        "fga3_sim",
        "fta_sim",
    }
    for day in _daterange(start, end):
        day_dir = base / f"game_date={day.date()}"
        if not day_dir.exists():
            continue
        files = sorted(day_dir.glob("world=*.parquet")) or [day_dir]
        for fp in files:
            df_full = pd.read_parquet(fp, engine="pyarrow")
            cols = [c for c in df_full.columns if c in wanted_cols]
            df = df_full[cols].copy()
            if "minutes_sim" not in df.columns:
                for fallback in ("minutes_p50", "minutes_mean"):
                    if fallback in df.columns:
                        df["minutes_sim"] = df[fallback]
                        break
            if "pts_world" not in df.columns:
                if {"fga2_sim", "fga3_sim", "fta_sim"}.issubset(df.columns):
                    df["pts_world"] = 2.0 * df["fga2_sim"] + 3.0 * df["fga3_sim"] + 0.75 * df["fta_sim"]
                else:
                    df["pts_world"] = np.nan
            if "sim_profile" not in df.columns:
                df["sim_profile"] = None
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    if profile is not None and "sim_profile" in df.columns:
        df = df[df["sim_profile"].fillna("") == profile]
    return df


def aggregate_worlds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "dk_fpts_mean": g["dk_fpts_world"].mean(),
                "dk_fpts_std": g["dk_fpts_world"].std(ddof=0),
                "dk_fpts_p90": g["dk_fpts_world"].quantile(0.90),
                "dk_fpts_p95": g["dk_fpts_world"].quantile(0.95),
                "pts_mean": g["pts_world"].mean(),
                "pts_std": g["pts_world"].std(ddof=0),
                "pts_p90": g["pts_world"].quantile(0.90),
                "pts_p95": g["pts_world"].quantile(0.95),
                "minutes_sim_mean": g["minutes_sim"].mean(),
            }
        )
    out = df.groupby(["game_date", "player_id"], dropna=False).apply(_agg).reset_index()
    return out


def _extract_player_rows(payload: dict, game_id: int, game_date: pd.Timestamp, side: str) -> list[dict]:
    rows: list[dict] = []
    team_payload = payload.get(side) or {}
    opp_payload = payload.get("home" if side == "away" else "away") or {}
    team_id = int(team_payload.get("team_id") or team_payload.get("teamId") or 0)
    opponent_id = int(opp_payload.get("team_id") or opp_payload.get("teamId") or 0)
    home_flag = 1 if side == "home" else 0
    for player in team_payload.get("players", []):
        stats = player.get("statistics") or {}
        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "team_id": team_id,
                "opponent_id": opponent_id,
                "player_id": int(player.get("person_id") or player.get("personId") or 0),
                "home_flag": home_flag,
                "minutes": stats.get("minutes"),
                "points": float(stats.get("points") or 0.0),
                "fga": float(stats.get("fieldGoalsAttempted") or 0.0),
                "fgm": float(stats.get("fieldGoalsMade") or 0.0),
                "three_pa": float(stats.get("threePointersAttempted") or 0.0),
                "three_pm": float(stats.get("threePointersMade") or 0.0),
                "fta": float(stats.get("freeThrowsAttempted") or 0.0),
                "ftm": float(stats.get("freeThrowsMade") or 0.0),
                "oreb": float(stats.get("reboundsOffensive") or 0.0),
                "dreb": float(stats.get("reboundsDefensive") or 0.0),
                "assists": float(stats.get("assists") or 0.0),
                "steals": float(stats.get("steals") or 0.0),
                "blocks": float(stats.get("blocks") or 0.0),
                "turnovers": float(stats.get("turnovers") or 0.0),
            }
        )
    return rows


def load_actuals(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    records: list[dict] = []
    for day in _daterange(start, end):
        season = _season_from_day(day)
        path = root / "bronze" / "boxscores_raw" / f"season={season}" / f"date={day.date()}" / "boxscores_raw.parquet"
        if not path.exists():
            continue
        bronze = pd.read_parquet(path)
        for row in bronze.itertuples():
            payload = json.loads(row.payload)
            tip_ts = payload.get("game_time_utc") or payload.get("game_time_local")
            try:
                game_dt = pd.to_datetime(tip_ts).tz_convert("America/New_York").tz_localize(None).normalize()
            except Exception:
                game_dt = pd.Timestamp(day).normalize()
            game_id = int(str(payload.get("game_id") or row.game_id).zfill(10))
            for side in ("home", "away"):
                records.extend(_extract_player_rows(payload, game_id, game_dt, side))
        # end rows
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    # compute DK fpts
    box_df = pd.DataFrame(
        {
            "pts": df["points"],
            "fgm": df["fgm"],
            "fga": df["fga"],
            "fg3m": df["three_pm"],
            "fg3a": df["three_pa"],
            "ftm": df["ftm"],
            "fta": df["fta"],
            "reb": df["oreb"] + df["dreb"],
            "oreb": df["oreb"],
            "dreb": df["dreb"],
            "ast": df["assists"],
            "stl": df["steals"],
            "blk": df["blocks"],
            "tov": df["turnovers"],
            "pf": np.zeros(len(df)),
            "plus_minus": np.zeros(len(df)),
        }
    )
    df["dk_fpts_actual"] = compute_dk_fpts(box_df)
    df["pts_actual"] = df["points"]
    # minutes parsing
    def _parse_minutes(val):
        if val is None:
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and ":" in val:
            try:
                mins, secs = val.split(":")
                return float(mins) + float(secs) / 60.0
            except Exception:
                return np.nan
        try:
            return float(val)
        except Exception:
            return np.nan

    df["minutes_actual"] = df["minutes"].apply(_parse_minutes)
    df["game_date"] = pd.to_datetime(df.get("game_date", start)).dt.normalize()
    return df[
        [
            "game_date",
            "game_id",
            "team_id",
            "player_id",
            "pts_actual",
            "dk_fpts_actual",
            "minutes_actual",
            "fga",
            "fta",
        ]
    ]


def bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 10, 20, 30, 40, 90]
    df["minutes_bucket"] = pd.cut(df["minutes_sim_mean"], bins=bins, right=False)
    df["usage_proxy"] = df["fga"].fillna(0) + 0.44 * df["fta"].fillna(0)
    try:
        df["usage_bucket"] = pd.qcut(df["usage_proxy"].fillna(0), 4, duplicates="drop")
    except ValueError:
        df["usage_bucket"] = np.nan

    out_rows = []
    for by in ("minutes_bucket", "usage_bucket"):
        if by not in df.columns:
            continue
        grouped = df.dropna(subset=[by]).groupby(by)
        for bucket, frame in grouped:
            if frame.empty:
                continue
            pred_p95 = frame["dk_fpts_p95"]
            pred_p90 = frame["dk_fpts_p90"]
            actual_fpts = frame["dk_fpts_actual"]
            valid_ratio = actual_fpts > 0
            row = {
                "bucket_type": by,
                "bucket": str(bucket),
                "n": len(frame),
                "mae_fpts": float(np.abs(frame["dk_fpts_mean"] - frame["dk_fpts_actual"]).mean()),
                "mae_pts": float(np.abs(frame["pts_mean"] - frame["pts_actual"]).mean()),
                "p95_ratio_fpts": float((pred_p95[valid_ratio] / actual_fpts[valid_ratio]).replace([np.inf, -np.inf], np.nan).dropna().mean()) if valid_ratio.any() else np.nan,
                "p90_ratio_fpts": float((pred_p90[valid_ratio] / actual_fpts[valid_ratio]).replace([np.inf, -np.inf], np.nan).dropna().mean()) if valid_ratio.any() else np.nan,
            }
            out_rows.append(row)
    return pd.DataFrame(out_rows)


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD"),
    profile: str = typer.Option("rates_v0", help="Sim profile name for filtering worlds."),
    data_root: Path = typer.Option(..., help="Base data root (contains artifacts/sim_v2/worlds_fpts_v2 and bronze boxscores)."),
    profiles_path: Optional[Path] = typer.Option(None, help="Optional profile config path."),
) -> None:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    root = data_root.expanduser().resolve()

    profile_cfg = load_sim_v2_profile(profile=profile, profiles_path=profiles_path or DEFAULT_PROFILES_PATH)
    typer.echo(f"[eval] profile={profile_cfg.name} start={start_ts.date()} end={end_ts.date()}")

    worlds = load_worlds(root, start_ts, end_ts, profile=profile_cfg.name)
    if worlds.empty:
        typer.echo("[eval] no worlds found; aborting")
        raise SystemExit(1)
    worlds_agg = aggregate_worlds(worlds)

    actuals = load_actuals(root, start_ts, end_ts)
    if actuals.empty:
        typer.echo("[eval] no boxscores found; aborting")
        raise SystemExit(1)

    merged = worlds_agg.merge(actuals, on=["game_date", "player_id"], how="inner")
    if merged.empty:
        typer.echo("[eval] no overlap between worlds and actuals")
        raise SystemExit(1)

    merged["fpts_error"] = merged["dk_fpts_mean"] - merged["dk_fpts_actual"]
    merged["pts_error"] = merged["pts_mean"] - merged["pts_actual"]

    # Overall summary
    overall = {
        "n": int(len(merged)),
        "mae_fpts": float(np.abs(merged["fpts_error"]).mean()),
        "rmse_fpts": float(np.sqrt((merged["fpts_error"] ** 2).mean())),
        "mae_pts": float(np.abs(merged["pts_error"]).mean()),
        "rmse_pts": float(np.sqrt((merged["pts_error"] ** 2).mean())),
    }

    buckets = bucket_stats(merged)

    typer.echo("\nOverall:")
    typer.echo(json.dumps(overall, indent=2))
    if not buckets.empty:
        typer.echo("\nBuckets:")
        typer.echo(buckets.to_string(index=False))

    out_dir = root / "artifacts" / "sim_v2" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile_cfg.name,
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "overall": overall,
        "buckets": buckets.to_dict(orient="records"),
    }
    out_path = out_dir / f"calibration_{profile_cfg.name}_{start_ts.date()}_{end_ts.date()}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    app()
