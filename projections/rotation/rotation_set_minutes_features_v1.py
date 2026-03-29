"""Shared feature engineering for rotation_set_minutes_v1.

These helpers are used by both:
- training dataset builder(s) under `scripts/rotation/`, and
- live feature builders used for production inference.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from projections.rotation.utils import zfill_game_id_series

ODDS_COLS: tuple[str, str] = ("spread_home", "total")
OUT_LIKE_STATUSES: frozenset[str] = frozenset(
    {"OUT", "O", "DOUBTFUL", "D", "INACTIVE"}
)
SAME_POS_CONTEXT_BUCKETS: tuple[str, ...] = ("thin", "normal", "deep")
CONTEXT_PRIOR_WINDOWS: tuple[int, ...] = (5, 10, 20)
CONTEXT_PRIOR_MIN_MATCH_GAMES = 2

# Shared derived features used by rotation-set/gameflow minutes datasets.
ROTATION_SET_DERIVED_FEATURES: tuple[str, ...] = (
    "vac_missing",
    "team_n_players",
    "team_n_not_out",
    "available_G_not_out",
    "available_W_not_out",
    "available_B_not_out",
    "depth_same_pos_not_out",
    "vacated_minutes_prior_20_total",
    "vacated_minutes_prior_20_same_pos",
    "team_prior_minutes_20_not_out",
    "prior_minutes_share_20",
    # Role-change features: short-vs-long window prior divergence.
    # Positive values indicate a recent upgrade (e.g., bench → starter).
    "role_change_starter_5v20",
    "role_change_minutes_5v20",
    "role_change_starter_5v10",
    "role_change_minutes_5v10",
    # Context-conditioned priors: select the matched same-position availability
    # bucket when there is enough historical support, else back off to global priors.
    "ctx_minutes_from_stints_prior_5",
    "ctx_minutes_from_stints_prior_10",
    "ctx_minutes_from_stints_prior_20",
    "ctx_started_proxy_rate_prior_5",
    "ctx_started_proxy_rate_prior_10",
    "ctx_started_proxy_rate_prior_20",
    "ctx_prior_n_games_5",
    "ctx_prior_n_games_10",
    "ctx_prior_n_games_20",
    "ctx_prior_backoff_used_5",
    "ctx_prior_backoff_used_10",
    "ctx_prior_backoff_used_20",
)


def apply_odds_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute odds missing flags + fill missing odds with 0.0.

    This matches the logic used in `scripts/rotation/build_rotation_train_dataset_v1.py`.
    """

    for col in ODDS_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required odds column: {col}")
    out = df.copy()
    out["spread_home"] = pd.to_numeric(out["spread_home"], errors="coerce").astype("float64")
    out["total"] = pd.to_numeric(out["total"], errors="coerce").astype("float64")
    out["spread_home_missing"] = out["spread_home"].isna()
    out["total_missing"] = out["total"].isna()
    out["spread_home"] = out["spread_home"].fillna(0.0).astype("float64")
    out["total"] = out["total"].fillna(0.0).astype("float64")
    return out


def fill_numeric_missing_with_zero(
    df: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("game_id", "team_id", "player_id"),
) -> pd.DataFrame:
    """Fill missing numeric values with 0 and coerce booleans to int8.

    This is intentionally broad (all numeric columns), matching the training builder.
    """

    out = df.copy()
    key_set = set(key_cols)
    for col in out.columns:
        if col in key_set:
            continue
        series = out[col]
        if pd.api.types.is_bool_dtype(series):
            out[col] = series.fillna(False).astype("int8")
        elif pd.api.types.is_numeric_dtype(series):
            out[col] = series.fillna(0)
    return out


def propagate_team_level_columns(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    """Propagate team-level columns within each (game_id, team_id)."""

    if not cols:
        return df
    out = df.copy()
    group_cols = ["game_id", "team_id"]
    missing_group_cols = [c for c in group_cols if c not in out.columns]
    if missing_group_cols:
        raise ValueError(f"Cannot propagate team-level columns; missing keys: {missing_group_cols}")
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(group_cols, sort=False)[col].bfill().ffill()
    return out


def join_rotation_priors(
    df: pd.DataFrame,
    *,
    team_priors: pd.DataFrame,
    player_priors: pd.DataFrame,
) -> pd.DataFrame:
    """Join rotation_priors_v1 tables onto a minutes feature frame.

    Priors are keyed by:
      - team_priors: (game_id, team_id)
      - player_priors: (game_id, team_id, person_id) where person_id == player_id
    """

    required_keys = {"game_id", "team_id", "player_id"}
    if not required_keys.issubset(df.columns):
        raise ValueError(f"Minutes dataset missing join keys: {sorted(required_keys - set(df.columns))}")

    out = df.copy()
    out["game_id_norm"] = zfill_game_id_series(out["game_id"])
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["person_id"] = out["player_id"]
    if "opponent_team_id" in out.columns:
        out["opponent_team_id"] = pd.to_numeric(out["opponent_team_id"], errors="coerce").astype("Int64")

    if not team_priors.empty:
        tp = team_priors.copy()
        if "game_id_norm" not in tp.columns and "game_id" in tp.columns:
            tp["game_id_norm"] = zfill_game_id_series(tp["game_id"])
        tp["team_id"] = pd.to_numeric(tp["team_id"], errors="coerce").astype("Int64")
        tp = tp.drop_duplicates(subset=["game_id_norm", "team_id"], keep="last")
        tp_prior_cols = [c for c in tp.columns if "prior_" in str(c).lower()]
        tp_cols = ["game_id_norm", "team_id", *tp_prior_cols]
        tp_cols = [c for i, c in enumerate(tp_cols) if c in tp.columns and c not in tp_cols[:i]]
        out = out.merge(tp.loc[:, tp_cols], on=["game_id_norm", "team_id"], how="left", suffixes=("", "_prior_team"))

        if "opponent_team_id" in out.columns:
            opp_tp = tp.loc[:, tp_cols].rename(columns={"team_id": "opponent_team_id"}).copy()
            rename_map = {c: f"opp_{c}" for c in tp_prior_cols}
            opp_tp = opp_tp.rename(columns=rename_map)
            out = out.merge(
                opp_tp,
                on=["game_id_norm", "opponent_team_id"],
                how="left",
            )

    if not player_priors.empty:
        pp = player_priors.copy()
        if "game_id_norm" not in pp.columns and "game_id" in pp.columns:
            pp["game_id_norm"] = zfill_game_id_series(pp["game_id"])
        pp["team_id"] = pd.to_numeric(pp["team_id"], errors="coerce").astype("Int64")
        pp["person_id"] = pd.to_numeric(pp["person_id"], errors="coerce").astype("Int64")
        pp = pp.drop_duplicates(subset=["game_id_norm", "team_id", "person_id"], keep="last")
        pp_cols = ["game_id_norm", "team_id", "person_id"] + [c for c in pp.columns if "prior_" in str(c).lower()]
        pp_cols = [c for i, c in enumerate(pp_cols) if c in pp.columns and c not in pp_cols[:i]]
        out = out.merge(
            pp.loc[:, pp_cols],
            on=["game_id_norm", "team_id", "person_id"],
            how="left",
            suffixes=("", "_prior_player"),
        )

    prior_cols = [c for c in out.columns if "_prior_" in str(c).lower()]
    for col in prior_cols:
        if str(col).endswith("_missing") and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1).astype("int8")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    count_cols = [
        c
        for c in out.columns
        if str(c).startswith("team_prior_n_games_") or str(c).startswith("player_prior_n_games_")
    ]
    for col in count_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int16")

    out = out.drop(columns=["game_id_norm", "person_id"])
    return out


def _normalize_pos_bucket(df: pd.DataFrame) -> pd.Series:
    """Normalize position to pos_bucket (G/W/B) for derived feature computation.

    Handles multiple possible column names and position string formats:
    - pos_bucket: already normalized (G/W/B/UNK)
    - dk_pos/pos/position: raw position strings (PG/SG/SF/PF/C/G/F/W)

    Position mapping:
    - G (Guard): PG, SG, G
    - W (Wing): SF, PF, F, W
    - B (Big): C
    - UNK: unknown or missing
    """
    # Try pos_bucket first (already normalized)
    if "pos_bucket" in df.columns:
        pos = df["pos_bucket"].astype("string").fillna("UNK").str.upper().str.strip()
        # Validate it's in expected format
        valid_buckets = {"G", "W", "B", "BIG", "UNK"}
        if pos.isin(valid_buckets).all():
            # Normalize BIG -> B for consistency
            return pos.replace({"BIG": "B"})

    # Try other position columns
    pos_col = None
    for col in ("dk_pos", "pos", "position"):
        if col in df.columns:
            pos_col = col
            break

    if pos_col is None:
        # No position column found - return UNK
        return pd.Series("UNK", index=df.index, dtype="string")

    raw_pos = df[pos_col].astype("string").fillna("UNK").str.upper().str.strip()

    # Map to G/W/B
    mapping = {
        "PG": "G",
        "SG": "G",
        "G": "G",
        "SF": "W",
        "PF": "W",
        "F": "W",
        "W": "W",
        "C": "B",
        "BIG": "B",
        "B": "B",
    }
    return raw_pos.map(mapping).fillna("W").astype("string")


def bucket_same_pos_depth(depth: pd.Series) -> pd.Series:
    """Bucket same-position depth into coarse availability regimes."""

    values = pd.to_numeric(depth, errors="coerce").fillna(0).astype("float64")
    out = pd.Series("deep", index=values.index, dtype="string")
    out = out.mask(values <= 3.0, "normal")
    out = out.mask(values <= 1.0, "thin")
    return out


def compute_same_pos_depth(df: pd.DataFrame) -> pd.Series:
    """Compute same-position teammate depth, excluding self when possible."""

    if "depth_same_pos_not_out" in df.columns:
        return pd.to_numeric(df["depth_same_pos_not_out"], errors="coerce").fillna(0.0).astype("float64")
    if "depth_same_pos_active" in df.columns:
        return pd.to_numeric(df["depth_same_pos_active"], errors="coerce").fillna(0.0).astype("float64")

    pos_bucket = _normalize_pos_bucket(df)
    available_g = (
        pd.to_numeric(df["available_G_not_out"], errors="coerce").fillna(0.0)
        if "available_G_not_out" in df.columns
        else pd.to_numeric(df.get("available_G", 0.0), errors="coerce").fillna(0.0)
    )
    available_w = (
        pd.to_numeric(df["available_W_not_out"], errors="coerce").fillna(0.0)
        if "available_W_not_out" in df.columns
        else pd.to_numeric(df.get("available_W", 0.0), errors="coerce").fillna(0.0)
    )
    available_b = (
        pd.to_numeric(df["available_B_not_out"], errors="coerce").fillna(0.0)
        if "available_B_not_out" in df.columns
        else pd.to_numeric(df.get("available_B", 0.0), errors="coerce").fillna(0.0)
    )
    is_out = _normalize_is_out(df).astype(bool)

    depth_same = pd.Series(0.0, index=df.index, dtype="float64")
    depth_same = depth_same.where(pos_bucket != "G", available_g.astype("float64"))
    depth_same = depth_same.where(pos_bucket != "W", available_w.astype("float64"))
    depth_same = depth_same.where(~pos_bucket.isin(["B", "BIG"]), available_b.astype("float64"))
    depth_same = depth_same - (~is_out).astype("float64")
    depth_same = depth_same.where(pos_bucket != "UNK", 0.0)
    return depth_same.clip(lower=0.0)


def _normalize_is_out(df: pd.DataFrame) -> pd.Series:
    """Normalize out status from available columns.

    Checks multiple columns in priority order:
    1. is_out (explicit flag)
    2. status (string like "OUT")

    Returns Series of 0/1 integers.
    """
    out_mask = pd.Series(0, index=df.index, dtype="int8")

    if "is_out" in df.columns:
        is_out = pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int)
        out_mask = (is_out == 1).astype("int8")

    if "status" in df.columns:
        status = df["status"].astype("string").str.upper().str.strip()
        status_out = status.isin(OUT_LIKE_STATUSES).fillna(False)
        out_mask = out_mask | status_out.astype("int8")

    return out_mask.astype("int8")


def _normalize_prior_minutes_20(df: pd.DataFrame) -> pd.Series:
    """Normalize prior minutes column to a consistent name.

    Checks multiple possible column names:
    - minutes_from_stints_prior_20 (canonical)
    - prior_minutes_20
    - prior_20_minutes
    - prior20

    Returns Series of float64.
    """
    candidates = [
        "minutes_from_stints_prior_20",
        "prior_minutes_20",
        "prior_20_minutes",
        "prior20",
    ]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")

    # No prior minutes column found - return zeros
    return pd.Series(0.0, index=df.index, dtype="float64")


def _select_context_prior_with_backoff(
    df: pd.DataFrame,
    *,
    base_col: str,
    bucket_base_template: str,
    count_template: str,
    out_col: str,
    count_out_col: str,
    backoff_col: str,
    min_match_games: int = CONTEXT_PRIOR_MIN_MATCH_GAMES,
) -> pd.DataFrame:
    """Select context-specific prior by current bucket, with global backoff."""

    out = df.copy()
    current_bucket = bucket_same_pos_depth(compute_same_pos_depth(out))
    base_vals = (
        pd.to_numeric(out[base_col], errors="coerce").fillna(0.0)
        if base_col in out.columns
        else pd.Series(0.0, index=out.index, dtype="float64")
    )

    selected_vals = base_vals.copy()
    selected_counts = pd.Series(0.0, index=out.index, dtype="float64")
    used_match = pd.Series(False, index=out.index)

    for bucket in SAME_POS_CONTEXT_BUCKETS:
        value_col = bucket_base_template.format(bucket=bucket)
        count_col = count_template.format(bucket=bucket)
        mask = current_bucket.eq(bucket)
        if not mask.any():
            continue
        bucket_vals = (
            pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
            if value_col in out.columns
            else pd.Series(0.0, index=out.index, dtype="float64")
        )
        bucket_counts = (
            pd.to_numeric(out[count_col], errors="coerce").fillna(0.0)
            if count_col in out.columns
            else pd.Series(0.0, index=out.index, dtype="float64")
        )
        eligible = mask & bucket_counts.ge(float(min_match_games))
        selected_counts.loc[mask] = bucket_counts.loc[mask]
        selected_vals.loc[eligible] = bucket_vals.loc[eligible]
        used_match.loc[eligible] = True

    out[out_col] = selected_vals.astype("float64")
    out[count_out_col] = selected_counts.astype("int16")
    out[backoff_col] = (~used_match).astype("int8")
    return out


def add_rotation_set_derived_features(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add derived features required by rotation_set_minutes models.

    Computes shared derived features that depend on team-game context:
    - vac_missing: 1 if vacancy columns are missing/NaN, else 0
    - team_n_players: count of players per (game_id, team_id)
    - team_n_not_out: count of not-out players per team-game
    - available_G_not_out: count of not-out Guards
    - available_W_not_out: count of not-out Wings
    - available_B_not_out: count of not-out Bigs
    - depth_same_pos_not_out: count of not-out players at same position (minus self if not out)
    - vacated_minutes_prior_20_total: sum of prior_20 minutes for OUT players
    - vacated_minutes_prior_20_same_pos: sum of prior_20 minutes for OUT players at same pos
    - team_prior_minutes_20_not_out: sum of prior_20 minutes for not-out players
    - prior_minutes_share_20: player's share of team_prior_minutes_20_not_out
    - role_change_starter_5v20: started_proxy_rate_prior_5 - started_proxy_rate_prior_20
    - role_change_minutes_5v20: minutes_from_stints_prior_5 - minutes_from_stints_prior_20
    - role_change_starter_5v10: started_proxy_rate_prior_5 - started_proxy_rate_prior_10
    - role_change_minutes_5v10: minutes_from_stints_prior_5 - minutes_from_stints_prior_10
    - ctx_*: same-position-context priors with global backoff

    Args:
        df: DataFrame with game_id, team_id, player_id and position/injury columns
        feature_columns: If provided, only compute features that are in this list

    Returns:
        DataFrame with derived features added
    """
    # Check which features we need to compute
    needed = set(ROTATION_SET_DERIVED_FEATURES)
    if feature_columns is not None:
        needed = needed & set(feature_columns)

    if not needed:
        return df

    # Check required keys
    required_keys = {"game_id", "team_id", "player_id"}
    missing_keys = required_keys - set(df.columns)
    if missing_keys:
        raise ValueError(f"Missing required key columns: {sorted(missing_keys)}")

    out = df.copy()

    # Normalize inputs
    pos_bucket = _normalize_pos_bucket(out)
    is_out = _normalize_is_out(out)
    prior_20 = _normalize_prior_minutes_20(out)

    # Store normalized columns for computation
    out["_pos_bucket"] = pos_bucket
    out["_is_out"] = is_out
    out["_prior_20"] = prior_20
    out["_is_not_out"] = (is_out == 0).astype("int8")

    # Team grouping keys
    team_keys = [out["game_id"], out["team_id"]]

    # Team-level counts
    if "team_n_players" in needed:
        out["team_n_players"] = (
            out["player_id"].groupby(team_keys, sort=False).transform("size").astype("int16")
        )

    if "team_n_not_out" in needed:
        out["team_n_not_out"] = (
            out["_is_not_out"].groupby(team_keys, sort=False).transform("sum").astype("int16")
        )

    # Position-specific available counts (not-out players by position)
    is_not_out = out["_is_not_out"] == 1

    if "available_G_not_out" in needed:
        out["available_G_not_out"] = (
            ((out["_pos_bucket"] == "G") & is_not_out)
            .groupby(team_keys, sort=False)
            .transform("sum")
            .astype("int16")
        )

    if "available_W_not_out" in needed:
        out["available_W_not_out"] = (
            ((out["_pos_bucket"] == "W") & is_not_out)
            .groupby(team_keys, sort=False)
            .transform("sum")
            .astype("int16")
        )

    if "available_B_not_out" in needed:
        out["available_B_not_out"] = (
            ((out["_pos_bucket"] == "B") & is_not_out)
            .groupby(team_keys, sort=False)
            .transform("sum")
            .astype("int16")
        )

    # depth_same_pos_not_out: available at same position minus self (if not out)
    if "depth_same_pos_not_out" in needed:
        depth_same = pd.Series(0, index=out.index, dtype="int64")
        # Map position to the corresponding available count
        g_mask = out["_pos_bucket"] == "G"
        w_mask = out["_pos_bucket"] == "W"
        b_mask = out["_pos_bucket"] == "B"

        if "available_G_not_out" in out.columns:
            depth_same = depth_same.where(~g_mask, out["available_G_not_out"].astype("int64"))
        if "available_W_not_out" in out.columns:
            depth_same = depth_same.where(~w_mask, out["available_W_not_out"].astype("int64"))
        if "available_B_not_out" in out.columns:
            depth_same = depth_same.where(~b_mask, out["available_B_not_out"].astype("int64"))

        # Subtract self if not out (don't count yourself in depth)
        depth_same = depth_same - is_not_out.astype("int64")
        # UNK position always gets 0
        depth_same = depth_same.where(out["_pos_bucket"] != "UNK", 0)
        out["depth_same_pos_not_out"] = depth_same.clip(lower=0).astype("int16")

    # Vacancy features (minutes vacated by OUT players)
    if "vacated_minutes_prior_20_total" in needed:
        vac_out = out["_prior_20"] * (out["_is_out"] == 1).astype("float64")
        out["vacated_minutes_prior_20_total"] = (
            vac_out.groupby(team_keys, sort=False).transform("sum").astype("float64")
        )

    if "vacated_minutes_prior_20_same_pos" in needed:
        vac_out = out["_prior_20"] * (out["_is_out"] == 1).astype("float64")
        out["vacated_minutes_prior_20_same_pos"] = (
            vac_out.groupby([out["game_id"], out["team_id"], out["_pos_bucket"]], sort=False)
            .transform("sum")
            .astype("float64")
        )
        # UNK position uses total vacated
        unk_mask = out["_pos_bucket"] == "UNK"
        if unk_mask.any() and "vacated_minutes_prior_20_total" in out.columns:
            out.loc[unk_mask, "vacated_minutes_prior_20_same_pos"] = out.loc[
                unk_mask, "vacated_minutes_prior_20_total"
            ]

    # Team prior minutes for not-out players + share
    if "team_prior_minutes_20_not_out" in needed or "prior_minutes_share_20" in needed:
        prior_not_out = out["_prior_20"] * is_not_out.astype("float64")
        out["team_prior_minutes_20_not_out"] = (
            prior_not_out.groupby(team_keys, sort=False).transform("sum").astype("float64")
        )

    if "prior_minutes_share_20" in needed:
        denom = out["team_prior_minutes_20_not_out"].replace({0.0: np.nan})
        share = (out["_prior_20"] / denom).fillna(0.0) * is_not_out.astype("float64")
        out["prior_minutes_share_20"] = share.astype("float64")

    # vac_missing: mirror training builder behavior (all vacancy columns null)
    if "vac_missing" in needed:
        vac_cols = [
            c
            for c in ("vac_min_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn")
            if c in out.columns
        ]
        if vac_cols:
            out["vac_missing"] = out[vac_cols].isna().all(axis=1).astype("int8")
        else:
            # No vacancy columns present - mark as missing
            out["vac_missing"] = 1
        out["vac_missing"] = pd.to_numeric(out["vac_missing"], errors="coerce").fillna(1).astype("int8")

    # Role-change features: short-vs-long window prior divergence.
    # These signal "next man up" situations where a player's recent role
    # diverges from their historical role (e.g., bench player promoted to starter).
    _role_change_pairs = [
        ("role_change_starter_5v20", "started_proxy_rate_prior_5", "started_proxy_rate_prior_20"),
        ("role_change_minutes_5v20", "minutes_from_stints_prior_5", "minutes_from_stints_prior_20"),
        ("role_change_starter_5v10", "started_proxy_rate_prior_5", "started_proxy_rate_prior_10"),
        ("role_change_minutes_5v10", "minutes_from_stints_prior_5", "minutes_from_stints_prior_10"),
    ]
    for feat_name, short_col, long_col in _role_change_pairs:
        if feat_name not in needed:
            continue
        short_vals = (
            pd.to_numeric(out[short_col], errors="coerce").fillna(0.0)
            if short_col in out.columns
            else pd.Series(0.0, index=out.index)
        )
        long_vals = (
            pd.to_numeric(out[long_col], errors="coerce").fillna(0.0)
            if long_col in out.columns
            else pd.Series(0.0, index=out.index)
        )
        out[feat_name] = (short_vals - long_vals).astype("float64")

    for window in CONTEXT_PRIOR_WINDOWS:
        minutes_out = f"ctx_minutes_from_stints_prior_{window}"
        counts_out = f"ctx_prior_n_games_{window}"
        backoff_out = f"ctx_prior_backoff_used_{window}"
        if minutes_out in needed or counts_out in needed or backoff_out in needed:
            base_col = f"minutes_from_stints_prior_{window}"
            selected = _select_context_prior_with_backoff(
                out,
                base_col=base_col,
                bucket_base_template=f"minutes_from_stints_ctx_same_pos_{{bucket}}_prior_{window}",
                count_template=f"ctx_same_pos_{{bucket}}_prior_n_games_{window}",
                out_col=minutes_out,
                count_out_col=counts_out,
                backoff_col=backoff_out,
            )
            out[minutes_out] = pd.to_numeric(selected[minutes_out], errors="coerce").fillna(0.0).astype("float64")
            out[counts_out] = pd.to_numeric(selected[counts_out], errors="coerce").fillna(0).astype("int16")
            out[backoff_out] = pd.to_numeric(selected[backoff_out], errors="coerce").fillna(1).astype("int8")

        starter_out = f"ctx_started_proxy_rate_prior_{window}"
        starter_backoff_out = f"ctx_prior_backoff_used_{window}"
        if starter_out in needed:
            base_col = f"started_proxy_rate_prior_{window}"
            selected = _select_context_prior_with_backoff(
                out,
                base_col=base_col,
                bucket_base_template=f"started_proxy_rate_ctx_same_pos_{{bucket}}_prior_{window}",
                count_template=f"ctx_same_pos_{{bucket}}_prior_n_games_{window}",
                out_col=starter_out,
                count_out_col=counts_out,
                backoff_col=starter_backoff_out,
            )
            out[starter_out] = pd.to_numeric(selected[starter_out], errors="coerce").fillna(0.0).astype("float64")

    # Clean up temporary columns
    out = out.drop(columns=["_pos_bucket", "_is_out", "_prior_20", "_is_not_out"], errors="ignore")

    return out
