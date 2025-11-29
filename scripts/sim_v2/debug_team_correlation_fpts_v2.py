"""Debug same-team residual correlations (empirical vs simulated worlds)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.fpts_v2.features import CATEGORICAL_FEATURES_DEFAULT, build_fpts_design_matrix
from projections.fpts_v2.loader import load_fpts_bundle

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid date: {value}") from exc


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "fpts_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "fpts_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    return sorted(partitions)


def _pairwise_mean_corr(series: pd.Series) -> Optional[float]:
    if series.shape[0] < 2:
        return None
    corr_mat = series.to_frame().corr()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    valid = upper.stack()
    if valid.empty:
        return None
    return float(valid.mean())


def _team_pairwise_corr(values: pd.Series, players: pd.Series) -> Optional[float]:
    df = pd.DataFrame({"player_id": players, "resid": values}).dropna()
    if df["player_id"].nunique() < 2:
        return None
    pivot = df.pivot_table(index="player_id", values="resid", aggfunc="mean")
    if pivot.shape[0] < 2:
        return None
    corr_mat = pivot.corr()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    valid = upper.stack()
    if valid.empty:
        return None
    return float(valid.mean())


def _pair_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _mean_pairwise_corr(residuals: pd.Series | np.ndarray) -> float:
    arr = np.asarray(residuals, dtype=float)
    if arr.size < 2:
        return np.nan
    corr = np.atleast_2d(np.corrcoef(arr))
    if corr.shape[0] < 2:
        return np.nan
    n = corr.shape[0]
    upper = corr[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return np.nan
    return float(np.nanmean(upper))


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root"),
    worlds_root: Path = typer.Option(..., "--worlds-root"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    n_worlds: int = typer.Option(2000, "--n-worlds"),
    fpts_run_id: Optional[str] = typer.Option(None, "--fpts-run-id"),
) -> None:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    start_ts = pd.Timestamp(start_dt).normalize()
    end_ts = pd.Timestamp(end_dt).normalize()

    partitions = _iter_partitions(data_root, start_ts, end_ts)
    if not partitions:
        typer.echo("[team_corr] no base partitions in range; exiting.")
        raise typer.Exit(code=0)
    base_frames = [pd.read_parquet(p) for p in partitions]
    base_df = pd.concat(base_frames, ignore_index=True)
    base_df["game_date"] = pd.to_datetime(base_df["game_date"]).dt.normalize()
    typer.echo(
        f"[team_corr_debug][base] rows={len(base_df)} "
        f"dates={base_df['game_date'].nunique()} "
        f"non_null_actual={base_df['dk_fpts_actual'].notna().sum() if 'dk_fpts_actual' in base_df.columns else 0} "
        f"non_null_keys={base_df[['game_id','team_id','player_id']].notna().all(axis=1).sum() if ('game_id' in base_df and 'team_id' in base_df and 'player_id' in base_df) else 0}"
    )
    required = {"dk_fpts_actual", "game_id", "team_id", "player_id"}
    missing = [c for c in required if c not in base_df.columns]
    if missing:
        raise RuntimeError(f"[team_corr] missing columns in base data: {missing}")

    # Ensure dk_fpts_mean exists; if missing/NaN, score using the provided run.
    needs_pred = ("dk_fpts_mean" not in base_df.columns) or base_df["dk_fpts_mean"].isna().all()
    if needs_pred:
        if not fpts_run_id:
            raise RuntimeError("[team_corr] dk_fpts_mean missing and fpts_run_id not provided; cannot score predictions.")
        bundle = load_fpts_bundle(fpts_run_id, data_root=data_root)
        features = build_fpts_design_matrix(
            base_df,
            bundle.feature_cols,
            categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
            fill_missing_with_zero=True,
        )
        num_iter = getattr(bundle.model, "best_iteration", None) or getattr(bundle.model, "best_iteration_", None)
        preds = bundle.model.predict(
            features.values, num_iteration=int(num_iter) if num_iter and num_iter > 0 else None
        )
        base_df["dk_fpts_mean"] = preds
    elif "dk_fpts_mean" not in base_df.columns and "dk_fpts_pred" in base_df.columns:
        typer.echo("[team_corr_debug] using dk_fpts_pred as dk_fpts_mean fallback")
        base_df["dk_fpts_mean"] = base_df["dk_fpts_pred"]

    if "dk_fpts_mean" not in base_df.columns:
        raise RuntimeError("[team_corr] dk_fpts_mean not available after scoring; cannot proceed.")

    typer.echo(f"[team_corr_debug][empirical] rows_before_key_filter={len(base_df)}")
    base_df = base_df[
        base_df["dk_fpts_actual"].notna()
        & base_df["game_id"].notna()
        & base_df["team_id"].notna()
        & base_df["player_id"].notna()
    ]
    typer.echo(f"[team_corr_debug][empirical] rows_after_key_filter={len(base_df)}")

    base_df = base_df.dropna(subset=["dk_fpts_actual", "dk_fpts_mean"])
    base_df["resid_emp"] = base_df["dk_fpts_actual"] - base_df["dk_fpts_mean"]

    # Build sequences per team across games, then compute pairwise correlations across games.
    emp_corrs: list[float] = []
    base_df = base_df.sort_values(["team_id", "player_id", "game_date", "game_id"])
    if "dk_fpts_mean" not in base_df.columns and "dk_fpts_pred" in base_df.columns:
        base_df["dk_fpts_mean"] = base_df["dk_fpts_pred"]
    if "dk_fpts_mean" not in base_df.columns:
        raise RuntimeError("[team_corr] dk_fpts_mean missing after scoring/fallback.")

    emp_seq = base_df[["team_id", "player_id", "game_date", "game_id", "resid_emp", "dk_fpts_mean"]].copy()
    for team_id, g_team in emp_seq.groupby("team_id"):
        g_team = g_team.copy()
        g_team["game_key"] = list(zip(g_team["game_date"], g_team["game_id"]))
        pivot = g_team.pivot_table(index="game_key", columns="player_id", values="resid_emp", aggfunc="first")
        player_ids = pivot.columns.to_numpy()
        if len(player_ids) < 2:
            continue
        for i in range(len(player_ids)):
            for j in range(i + 1, len(player_ids)):
                x = pivot[player_ids[i]]
                y = pivot[player_ids[j]]
                c = _pair_corr(x, y)
                if not np.isnan(c):
                    emp_corrs.append(c)
    if emp_corrs:
        emp_mean = float(np.mean(emp_corrs))
        emp_median = float(np.median(emp_corrs))
        typer.echo(
            f"[team_corr][empirical] mean={emp_mean:.3f} median={emp_median:.3f} n_pairs={len(emp_corrs)}"
        )
    else:
        emp_mean = emp_median = None
        typer.echo("[team_corr] no empirical correlations; check date window / coverage")

    world_corrs: list[float] = []
    # base keys for dk_fpts_mean to join into worlds
    base_keys = emp_seq[["game_date", "game_id", "team_id", "player_id", "dk_fpts_mean"]].drop_duplicates()

    for game_date in pd.date_range(start=start_dt, end=end_dt, freq="D"):
        date_dir = worlds_root / f"game_date={game_date.date().isoformat()}"
        if not date_dir.exists():
            typer.echo(f"[team_corr_debug][sim] missing date_dir={date_dir}")
            continue
        world_files = sorted(date_dir.glob("world=*.parquet"))[:n_worlds]
        if not world_files:
            typer.echo(f"[team_corr_debug][sim] no world files for {game_date.date()}")
            continue

        df_w_list: list[pd.DataFrame] = []
        for wf in world_files:
            df_w = pd.read_parquet(wf)
            # ensure unique world_id per file
            name = wf.stem
            try:
                world_id_val = int(name.split("=", 1)[1])
            except Exception:
                world_id_val = len(df_w_list)
            df_w["world_id"] = world_id_val
            df_w_list.append(df_w)

        if not df_w_list:
            continue

        df_w_all = pd.concat(df_w_list, ignore_index=True)
        typer.echo(
            f"[team_corr_debug][sim] date={game_date.date()} rows_worlds_before_join={len(df_w_all)} "
            f"unique_world_ids={df_w_all['world_id'].nunique()}"
        )

        if "dk_fpts_mean" not in base_keys.columns and "dk_fpts_pred" in base_keys.columns:
            base_keys = base_keys.rename(columns={"dk_fpts_pred": "dk_fpts_mean"})
        df_sim = df_w_all.merge(
            base_keys,
            on=["game_date", "game_id", "team_id", "player_id"],
            how="inner",
            validate="many_to_one",
        )
        typer.echo(
            f"[team_corr_debug][sim] date={game_date.date()} rows_after_join={len(df_sim)} "
            f"unique_world_ids={df_sim['world_id'].nunique()}"
        )

        if df_sim.empty or df_sim["world_id"].nunique() < 2:
            typer.echo(f"[team_corr_debug][sim] date={game_date.date()} not enough worlds after join")
            continue

        df_sim = df_sim.copy()
        sim_col = "dk_fpts_world" if "dk_fpts_world" in df_sim.columns else "dk_fpts_sim"
        if sim_col not in df_sim.columns:
            typer.echo(f"[team_corr_debug][sim] missing simulated fpts column in joined data for {game_date.date()}")
            continue
        if "dk_fpts_mean" not in df_sim.columns:
            if "dk_fpts_mean_x" in df_sim.columns or "dk_fpts_mean_y" in df_sim.columns:
                df_sim["dk_fpts_mean"] = df_sim.get("dk_fpts_mean_x")
                if "dk_fpts_mean_y" in df_sim.columns:
                    df_sim["dk_fpts_mean"] = df_sim["dk_fpts_mean"].combine_first(df_sim["dk_fpts_mean_y"])
            else:
                typer.echo(f"[team_corr_debug][sim] missing dk_fpts_mean after join for {game_date.date()}")
                continue
        # clean duplicate mean cols
        for col in ["dk_fpts_mean_x", "dk_fpts_mean_y"]:
            if col in df_sim.columns:
                df_sim = df_sim.drop(columns=[col])
        df_sim["resid_sim"] = df_sim[sim_col] - df_sim["dk_fpts_mean"]

        for (_, g_id, t_id), g_team in df_sim.groupby(["game_date", "game_id", "team_id"]):
            pivot = g_team.pivot_table(
                index="world_id",
                columns="player_id",
                values="resid_sim",
                aggfunc="first",
            )
            player_ids = pivot.columns.to_numpy()
            if len(player_ids) < 2:
                continue
            for i in range(len(player_ids)):
                for j in range(i + 1, len(player_ids)):
                    x = pivot[player_ids[i]]
                    y = pivot[player_ids[j]]
                    c = _pair_corr(x, y)
                    if not np.isnan(c):
                        world_corrs.append(c)

    def _summary(corrs: list[float]) -> tuple[float, float, int]:
        if not corrs:
            return float("nan"), float("nan"), 0
        arr = np.array(corrs, dtype=float)
        return float(np.nanmean(arr)), float(np.nanmedian(arr)), int(len(arr))

    emp_mean, emp_median, emp_n = _summary(emp_corrs)
    sim_mean, sim_median, sim_n = _summary(world_corrs)

    if emp_n == 0:
        typer.echo("[team_corr] no empirical correlations; see [team_corr_debug] logs above")
    else:
        typer.echo(
            f"[team_corr][empirical] mean={emp_mean:.3f} median={emp_median:.3f} n_pairs={emp_n}"
        )
    if sim_n == 0:
        typer.echo("[team_corr] no simulated correlations; see [team_corr_debug] logs above")
    else:
        typer.echo(
            f"[team_corr][simulated] mean={sim_mean:.3f} median={sim_median:.3f} n_pairs={sim_n}"
        )

    out_path = (
        data_root
        / "artifacts"
        / "fpts_v2"
        / "runs"
        / (fpts_run_id or "unknown_run")
        / "team_corr_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "window": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
        "n_worlds": n_worlds,
        "empirical": {
            "mean_corr": emp_mean if emp_n > 0 else None,
            "median_corr": emp_median if emp_n > 0 else None,
            "n_pairs": emp_n,
        },
        "simulated": {
            "mean_corr": sim_mean if sim_n > 0 else None,
            "median_corr": sim_median if sim_n > 0 else None,
            "n_pairs": sim_n,
        },
        "args": {
            "data_root": str(data_root),
            "worlds_root": str(worlds_root),
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "n_worlds": n_worlds,
            "fpts_run_id": fpts_run_id,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[team_corr] wrote {out_path}")


if __name__ == "__main__":
    app()
