from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_injury_semantics_for_rmh(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize injury columns to match RMH training semantics.

    Training semantics (rotation_train_v1_boxscore_20260112):
      - status == "Ava" means no injury-feed row for the player
      - For Ava rows:
          injury_as_of_ts is null
          injury_snapshot_missing == 1
          prior_play_prob is null

    Live minutes features include an explicit boolean `injury_row_present`. We use it to
    reconstruct training semantics without changing the upstream minutes feature builder.
    """

    if "injury_row_present" not in df.columns:
        return df

    out = df.copy()
    no_row = pd.to_numeric(out["injury_row_present"], errors="coerce").fillna(0).astype(int) == 0
    if not no_row.any():
        return out

    if "status" not in out.columns:
        out["status"] = "Ava"
    else:
        out.loc[no_row, "status"] = "Ava"

    if "injury_as_of_ts" in out.columns:
        out["injury_as_of_ts"] = pd.to_datetime(out["injury_as_of_ts"], utc=True, errors="coerce")
        out.loc[no_row, "injury_as_of_ts"] = pd.NaT
    else:
        out["injury_as_of_ts"] = pd.NaT

    if "injury_snapshot_missing" not in out.columns:
        out["injury_snapshot_missing"] = 0
    out["injury_snapshot_missing"] = (
        pd.to_numeric(out["injury_snapshot_missing"], errors="coerce").fillna(0).astype("int64")
    )
    out.loc[no_row, "injury_snapshot_missing"] = 1

    if "prior_play_prob" not in out.columns:
        out["prior_play_prob"] = np.nan
    out["prior_play_prob"] = pd.to_numeric(out["prior_play_prob"], errors="coerce")
    out.loc[no_row, "prior_play_prob"] = np.nan

    return out


def add_rmh_live_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RMH_v1 derived features expected by the production bundle.

    The RMH bundle expects several derived features that are not currently produced by
    `build_minutes_live`. These are computed from (a) injury/roster flags and
    (b) rotation priors joined onto the frame (minutes_from_stints_prior_20, etc).
    """

    out = df.copy()

    required_cols = {"game_id", "team_id", "player_id", "pos_bucket", "is_out", "minutes_from_stints_prior_20"}
    missing = sorted(required_cols.difference(out.columns))
    if missing:
        raise ValueError(f"RMH live derived features missing required columns: {missing}")

    # Core typing / normalization
    out["pos_bucket"] = out["pos_bucket"].astype("string").fillna("UNK").astype(str)
    out["is_out"] = pd.to_numeric(out["is_out"], errors="coerce").fillna(0).astype("int64")
    out["minutes_from_stints_prior_20"] = (
        pd.to_numeric(out["minutes_from_stints_prior_20"], errors="coerce").fillna(0.0).astype("float64")
    )

    # started_proxy: best available pre-tip proxy (5-per-team in live features).
    started = pd.Series(False, index=out.index)
    for col in ("is_confirmed_starter", "is_projected_starter"):
        if col in out.columns:
            started = started | out[col].fillna(False).astype(bool)
    if "starter_flag" in out.columns:
        started = started | (pd.to_numeric(out["starter_flag"], errors="coerce").fillna(0).astype(int) > 0)
    started = started & (out["is_out"] == 0)
    out["started_proxy"] = started.astype("int8")

    # has_injury_row per RMH spec.
    status = out.get("status", pd.Series(["Ava"] * len(out), index=out.index)).astype(str)
    injury_as_of_ts = (
        pd.to_datetime(out.get("injury_as_of_ts", pd.Series([pd.NaT] * len(out), index=out.index)), utc=True, errors="coerce")
    )
    out["has_injury_row"] = ((injury_as_of_ts.notna()) | (status != "Ava")).astype("int8")

    # Team-level counts.
    team_keys = [out["game_id"], out["team_id"]]
    out["team_n_players"] = out["player_id"].groupby(team_keys, sort=False).transform("size").astype("int16")
    out["team_n_not_out"] = (out["is_out"] == 0).groupby(team_keys, sort=False).transform("sum").astype("int16")

    # available_*_not_out (B == BIG)
    is_not_out = out["is_out"] == 0
    out["available_B_not_out"] = ((out["pos_bucket"] == "BIG") & is_not_out).groupby(team_keys, sort=False).transform(
        "sum"
    ).astype("int16")
    out["available_G_not_out"] = ((out["pos_bucket"] == "G") & is_not_out).groupby(team_keys, sort=False).transform(
        "sum"
    ).astype("int16")
    out["available_W_not_out"] = ((out["pos_bucket"] == "W") & is_not_out).groupby(team_keys, sort=False).transform(
        "sum"
    ).astype("int16")

    # depth_same_pos_not_out matches training:
    #   available_{bucket}_not_out - 1 if player is not out, else available_{bucket}_not_out.
    depth_same = pd.Series(0, index=out.index, dtype="int64")
    depth_same[out["pos_bucket"] == "BIG"] = out.loc[out["pos_bucket"] == "BIG", "available_B_not_out"].astype("int64")
    depth_same[out["pos_bucket"] == "G"] = out.loc[out["pos_bucket"] == "G", "available_G_not_out"].astype("int64")
    depth_same[out["pos_bucket"] == "W"] = out.loc[out["pos_bucket"] == "W", "available_W_not_out"].astype("int64")
    depth_same = depth_same - is_not_out.astype("int64")
    depth_same[out["pos_bucket"] == "UNK"] = 0
    out["depth_same_pos_not_out"] = depth_same.clip(lower=0).astype("int16")

    # Vacancy features (based on priors).
    vac_out = out["minutes_from_stints_prior_20"] * (out["is_out"] == 1).astype("float64")
    out["vacated_minutes_prior_20_total"] = vac_out.groupby(team_keys, sort=False).transform("sum").astype("float64")
    out["vacated_minutes_prior_20_same_pos"] = vac_out.groupby(
        [out["game_id"], out["team_id"], out["pos_bucket"]], sort=False
    ).transform("sum").astype("float64")
    unk_mask = out["pos_bucket"] == "UNK"
    if unk_mask.any():
        out.loc[unk_mask, "vacated_minutes_prior_20_same_pos"] = out.loc[unk_mask, "vacated_minutes_prior_20_total"]

    # team_prior_minutes_20_not_out + share (exact training semantics).
    prior_not_out = out["minutes_from_stints_prior_20"] * is_not_out.astype("float64")
    out["team_prior_minutes_20_not_out"] = prior_not_out.groupby(team_keys, sort=False).transform("sum").astype("float64")
    denom = out["team_prior_minutes_20_not_out"].replace({0.0: np.nan})
    share = (out["minutes_from_stints_prior_20"] / denom).fillna(0.0) * is_not_out.astype("float64")
    out["prior_minutes_share_20"] = share.astype("float64")

    # vac_missing: mirror training builder behavior (all vacancy columns null).
    vac_cols = [c for c in ("vac_min_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn") if c in out.columns]
    if vac_cols:
        out["vac_missing"] = out[vac_cols].isna().all(axis=1).astype("int64")
    else:
        out["vac_missing"] = 1
    out["vac_missing"] = pd.to_numeric(out["vac_missing"], errors="coerce").fillna(1).astype("int8")

    # Rotation missing flags are not observable pre-tip; approximate with injury OUT signal.
    out["rotation_team_missing"] = 0
    out["rotation_missing"] = 0
    out["rotation_player_row_missing_raw"] = (out["is_out"] == 1).astype("int8")
    out["rotation_player_filled_zero"] = out["rotation_player_row_missing_raw"].astype("int8")

    # Team-shape features (depth / dispersion / bench concentration).
    minutes_proxy = out["minutes_from_stints_prior_20"].where(is_not_out, 0.0).astype("float64")
    team_sum = minutes_proxy.groupby(team_keys, sort=False).transform("sum").astype("float64")
    scale = (240.0 / team_sum.replace({0.0: np.nan})).fillna(0.0)
    scaled = (minutes_proxy * scale).astype("float64")
    out["_rmh_scaled_minutes_proxy"] = scaled

    team_metrics: list[dict[str, float | int]] = []
    for (game_id, team_id), g in out.groupby(["game_id", "team_id"], sort=False):
        mins = pd.to_numeric(g["_rmh_scaled_minutes_proxy"], errors="coerce").fillna(0.0).astype("float64")
        team_total = float(mins.sum())
        starters_mask = g["started_proxy"].astype(bool)
        starter_pool = float(mins.loc[starters_mask].sum())
        bench_pool = float(team_total - starter_pool)
        bench_sorted = mins.loc[~starters_mask].sort_values(ascending=False).to_list()
        top1 = float(bench_sorted[0]) if len(bench_sorted) >= 1 else 0.0
        top2 = float(bench_sorted[1]) if len(bench_sorted) >= 2 else 0.0

        if team_total > 0:
            shares = (mins / team_total).to_numpy()
            denom_sq = float((shares**2).sum())
            effective_n = float(1.0 / denom_sq) if denom_sq > 0 else 0.0
        else:
            effective_n = 0.0

        team_metrics.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "depth_6": float((mins >= 6.0).sum()),
                "depth_10": float((mins >= 10.0).sum()),
                "depth_14": float((mins >= 14.0).sum()),
                "effective_n": float(effective_n),
                "starter_pool_minutes": float(starter_pool),
                "bench_pool_minutes": float(bench_pool),
                "bench_conc_top1": float(top1 / bench_pool) if bench_pool > 0 else 0.0,
                "bench_conc_top2": float((top1 + top2) / bench_pool) if bench_pool > 0 else 0.0,
            }
        )

    metrics_df = pd.DataFrame(team_metrics)
    out = out.merge(metrics_df, on=["game_id", "team_id"], how="left", validate="m:1")
    out = out.drop(columns=["_rmh_scaled_minutes_proxy"], errors="ignore")

    return out


def prepare_live_features_for_rmh(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_injury_semantics_for_rmh(df)
    out = add_rmh_live_derived_features(out)
    return out


__all__ = [
    "add_rmh_live_derived_features",
    "normalize_injury_semantics_for_rmh",
    "prepare_live_features_for_rmh",
]

