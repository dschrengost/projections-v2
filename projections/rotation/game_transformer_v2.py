"""Game-transformer v2 foundation: game collation + backbone + joint heads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from projections.rotation.joint_active_set import JointActiveSetHead, JointActiveSetOutputs
from projections.rotation.joint_game_flow import JointGameFlow, JointGameFlowOutputs
from projections.rotation.joint_minutes import JointMinutesHead, JointMinutesOutputs
from projections.rotation.set_model import zfill_game_id_series

MAX_PLAYERS_PER_TEAM = 15
TOTAL_PLAYERS_PER_GAME = 2 * MAX_PLAYERS_PER_TEAM
# With max_minutes_per_player=48, at least 5 players are required to make 240 feasible.
MIN_FEASIBLE_PLAYERS_PER_TEAM = 5
PROTECTED_PRIOR_PLAY_PROB_FLOOR = 0.938507
PROTECTED_PRIOR_MINUTES_FLOOR = 29.520922
OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP = 0.579943
OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10 = 6.053079
OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN = 0.117685
OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB = 2.202986
OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES = 0.051353
FLOW_TARGET_COLUMNS_V1 = [
    "fga2",
    "fg2m",
    "fga3",
    "fg3m",
    "fta",
    "ftm",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
]
FLOW_TARGET_COLUMNS_WITH_PF = [*FLOW_TARGET_COLUMNS_V1, "pf"]


def flow_target_columns(*, include_pf: bool) -> list[str]:
    return list(FLOW_TARGET_COLUMNS_WITH_PF if include_pf else FLOW_TARGET_COLUMNS_V1)


@dataclass(frozen=True)
class GameTransformerV2Config:
    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    game_feature_columns: list[str]
    team_feature_columns: list[str]
    d_model: int = 192
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.1
    ff_mult: float = 4.0
    min_active_count: int = 5
    max_active_count: int = 13
    active_threshold_minutes: float = 4.0
    total_minutes_per_team: float = 240.0
    max_minutes_per_player: float = 48.0
    flow_coupling_type: str = "affine"
    flow_num_blocks: int = 4
    flow_scale_clip: float = 3.0  # H1 fix: increased from 2.0 to reduce star under-projection
    flow_mean_ctx_weight: float = 1.0
    flow_context_mode: str = "attention"  # H2 fix: "attention" instead of "mean" for star concentration
    include_pf_in_flow_targets: bool = False
    overflow_protected_prior_play_prob_floor: float = PROTECTED_PRIOR_PLAY_PROB_FLOOR
    overflow_protected_prior_minutes_floor: float = PROTECTED_PRIOR_MINUTES_FLOOR
    overflow_risk_weight_consecutive_active_dnp: float = OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP
    overflow_risk_weight_active_but_dnp_rate_last10: float = OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10
    overflow_risk_weight_inactive_streak_len: float = OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN
    overflow_keep_weight_prior_play_prob: float = OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB
    overflow_keep_weight_prior_minutes: float = OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES
    version: str = "game_transformer_v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameTransformerV2Config":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in dict(payload).items() if k in known}
        return cls(**filtered)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GameTransformerV2Config":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GameLevelExample:
    player_features: np.ndarray  # (2,15,F)
    player_valid_mask: np.ndarray  # (2,15)
    player_ids: np.ndarray  # (2,15)
    team_ids: np.ndarray  # (2,)
    y_minutes: np.ndarray  # (2,15)
    flow_targets: np.ndarray  # (2,15,S)
    flow_observed_mask: np.ndarray  # (2,15,S)
    lineup_available: np.ndarray  # (2,15)
    game_features: np.ndarray  # (G,)
    team_features: np.ndarray  # (2,T)
    game_id_norm: str
    game_date: str


class GameLevelDataset(Dataset[GameLevelExample]):
    def __init__(self, examples: list[GameLevelExample]) -> None:
        if not examples:
            raise ValueError("GameLevelDataset requires >=1 example")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GameLevelExample:
        return self.examples[idx]


@dataclass(frozen=True)
class GameTransformerV2Outputs:
    game_state: torch.Tensor
    team_states: torch.Tensor
    player_states: torch.Tensor
    player_valid_mask: torch.Tensor
    player_team_index: torch.Tensor
    active: JointActiveSetOutputs
    minutes: JointMinutesOutputs
    flow: JointGameFlowOutputs | None


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for col in cols:
        if col not in df.columns:
            parts.append(pd.Series(0.0, index=df.index, name=col, dtype="float32"))
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            parts.append(df[col].astype("float32").rename(col))
        else:
            parts.append(pd.to_numeric(df[col], errors="coerce").rename(col))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _numeric_frame_with_nans_for_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts: list[pd.Series] = []
    for col in cols:
        if col not in df.columns:
            parts.append(pd.Series(np.nan, index=df.index, name=col, dtype="float32"))
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            parts.append(df[col].astype("float32").rename(col))
        else:
            parts.append(pd.to_numeric(df[col], errors="coerce").rename(col))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _resolve_home_away_team_ids(game_df: pd.DataFrame) -> tuple[int | None, int | None]:
    home_series = pd.to_numeric(game_df.get("home_team_id"), errors="coerce").dropna().astype("int64")
    away_series = pd.to_numeric(game_df.get("away_team_id"), errors="coerce").dropna().astype("int64")
    home_id = int(home_series.mode().iloc[0]) if not home_series.empty else None
    away_id = int(away_series.mode().iloc[0]) if not away_series.empty else None
    if home_id is not None and away_id is not None and home_id != away_id:
        return home_id, away_id

    # Fallback using home_flag if explicit team ids are unavailable.
    if "home_flag" in game_df.columns:
        hf = pd.to_numeric(game_df["home_flag"], errors="coerce")
        home_rows = game_df.loc[hf == 1]
        away_rows = game_df.loc[hf == 0]
        if not home_rows.empty and not away_rows.empty:
            home = pd.to_numeric(home_rows["team_id"], errors="coerce").dropna().astype("int64")
            away = pd.to_numeric(away_rows["team_id"], errors="coerce").dropna().astype("int64")
            if not home.empty and not away.empty:
                return int(home.mode().iloc[0]), int(away.mode().iloc[0])

    teams = sorted(
        {
            int(v)
            for v in pd.to_numeric(game_df["team_id"], errors="coerce").dropna().astype("int64").tolist()
        }
    )
    if len(teams) >= 2:
        return teams[0], teams[1]
    if len(teams) == 1:
        return teams[0], None
    return None, None


def _sort_team_rows(
    team_df: pd.DataFrame,
    *,
    protected_prior_play_prob_floor: float,
    protected_prior_minutes_floor: float,
    risk_weight_consecutive_active_dnp: float,
    risk_weight_active_but_dnp_rate_last10: float,
    risk_weight_inactive_streak_len: float,
    keep_weight_prior_play_prob: float,
    keep_weight_prior_minutes: float,
) -> pd.DataFrame:
    out = team_df.copy()
    if "is_out" not in out.columns:
        out["is_out"] = 0.0
    if "lineup_starter_announced" not in out.columns:
        out["lineup_starter_announced"] = 0
    if "prior_play_prob" not in out.columns:
        out["prior_play_prob"] = 0.0
    if "minutes_from_stints_prior_20" not in out.columns:
        out["minutes_from_stints_prior_20"] = 0.0
    if "an_has_any_props" not in out.columns:
        out["an_has_any_props"] = 0.0
    if "an_implied_minutes" not in out.columns:
        out["an_implied_minutes"] = 0.0
    if "consecutive_active_dnp" not in out.columns:
        out["consecutive_active_dnp"] = 0.0
    if "active_but_dnp_rate_last10" not in out.columns:
        out["active_but_dnp_rate_last10"] = 0.0
    if "inactive_streak_len" not in out.columns:
        out["inactive_streak_len"] = 0.0

    out["is_out"] = pd.to_numeric(out["is_out"], errors="coerce").fillna(0.0)
    out["lineup_starter_announced"] = pd.to_numeric(out["lineup_starter_announced"], errors="coerce").fillna(0.0)
    out["prior_play_prob"] = pd.to_numeric(out["prior_play_prob"], errors="coerce").fillna(0.0)
    out["minutes_from_stints_prior_20"] = pd.to_numeric(out["minutes_from_stints_prior_20"], errors="coerce").fillna(0.0)
    out["an_has_any_props"] = pd.to_numeric(out["an_has_any_props"], errors="coerce").fillna(0.0)
    out["an_implied_minutes"] = pd.to_numeric(out["an_implied_minutes"], errors="coerce").fillna(0.0)
    out["consecutive_active_dnp"] = pd.to_numeric(out["consecutive_active_dnp"], errors="coerce").fillna(0.0)
    out["active_but_dnp_rate_last10"] = pd.to_numeric(out["active_but_dnp_rate_last10"], errors="coerce").fillna(0.0)
    out["inactive_streak_len"] = pd.to_numeric(out["inactive_streak_len"], errors="coerce").fillna(0.0)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").fillna(0).astype("int64")

    starter = out["lineup_starter_announced"].ge(0.5)
    has_props = out["an_has_any_props"].ge(0.5)
    has_implied_minutes = out["an_implied_minutes"].gt(0.0)
    high_prior = out["prior_play_prob"].ge(float(protected_prior_play_prob_floor)) | out[
        "minutes_from_stints_prior_20"
    ].ge(float(protected_prior_minutes_floor))
    protected = starter | has_props | has_implied_minutes | high_prior
    out["overflow_protected"] = protected.astype(np.int8)

    # Lower values = higher risk of pre-tip DNP/zero-minute outcome.
    out["overflow_dnp_risk"] = (
        float(risk_weight_consecutive_active_dnp) * out["consecutive_active_dnp"].clip(lower=0.0, upper=20.0)
        + float(risk_weight_active_but_dnp_rate_last10) * out["active_but_dnp_rate_last10"].clip(lower=0.0, upper=1.0)
        + float(risk_weight_inactive_streak_len) * out["inactive_streak_len"].clip(lower=0.0, upper=20.0)
    )
    # Higher values = better keep candidates for overflow tie-breaks.
    out["overflow_keep_score"] = (
        float(keep_weight_prior_play_prob) * out["prior_play_prob"].clip(lower=0.0, upper=1.0)
        + float(keep_weight_prior_minutes) * out["minutes_from_stints_prior_20"].clip(lower=0.0, upper=48.0)
        - out["overflow_dnp_risk"]
    )

    return out.sort_values(
        by=[
            "is_out",
            "overflow_protected",
            "overflow_keep_score",
            "lineup_starter_announced",
            "an_has_any_props",
            "an_implied_minutes",
            "prior_play_prob",
            "minutes_from_stints_prior_20",
            "player_id",
        ],
        ascending=[True, False, False, False, False, False, False, False, True],
        kind="mergesort",
    )


def build_game_level_examples(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    game_feature_columns: list[str],
    team_feature_columns: list[str],
    flow_label_columns: list[str] | None = None,
    minutes_label_col: str = "minutes_label",
    max_players_per_team: int = MAX_PLAYERS_PER_TEAM,
    min_valid_players_per_team: int = MIN_FEASIBLE_PLAYERS_PER_TEAM,
    overflow_protected_prior_play_prob_floor: float = PROTECTED_PRIOR_PLAY_PROB_FLOOR,
    overflow_protected_prior_minutes_floor: float = PROTECTED_PRIOR_MINUTES_FLOOR,
    overflow_risk_weight_consecutive_active_dnp: float = OVERFLOW_RISK_WEIGHT_CONSECUTIVE_ACTIVE_DNP,
    overflow_risk_weight_active_but_dnp_rate_last10: float = OVERFLOW_RISK_WEIGHT_ACTIVE_BUT_DNP_RATE_LAST10,
    overflow_risk_weight_inactive_streak_len: float = OVERFLOW_RISK_WEIGHT_INACTIVE_STREAK_LEN,
    overflow_keep_weight_prior_play_prob: float = OVERFLOW_KEEP_WEIGHT_PRIOR_PLAY_PROB,
    overflow_keep_weight_prior_minutes: float = OVERFLOW_KEEP_WEIGHT_PRIOR_MINUTES,
) -> list[GameLevelExample]:
    """Convert flat player rows into per-game (home+away) training examples."""

    if max_players_per_team <= 0:
        raise ValueError("max_players_per_team must be > 0")
    if min_valid_players_per_team <= 0:
        raise ValueError("min_valid_players_per_team must be > 0")
    if min_valid_players_per_team > max_players_per_team:
        raise ValueError("min_valid_players_per_team must be <= max_players_per_team")
    if len(feature_columns) <= 0:
        raise ValueError("feature_columns must be non-empty")

    df = frame.copy().reset_index(drop=True)
    if "game_id_norm" not in df.columns:
        df["game_id_norm"] = zfill_game_id_series(df["game_id"])
    if "game_date" not in df.columns:
        raise ValueError("frame missing game_date")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    if df["game_date"].isna().any():
        raise ValueError("frame has invalid game_date values")

    x = _numeric_frame(df, feature_columns).to_numpy(dtype="float32", copy=False)
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    if mean.shape[0] != len(feature_columns) or std.shape[0] != len(feature_columns):
        raise ValueError("feature_mean/std must align with feature_columns")
    std = np.where(std <= 1e-6, 1.0, std)
    x = np.nan_to_num(x, nan=mean[None, :], posinf=mean[None, :], neginf=mean[None, :])
    x = (x - mean[None, :]) / std[None, :]

    if minutes_label_col not in df.columns:
        fallback = "minutes" if "minutes" in df.columns else None
        if fallback is None:
            raise ValueError(f"minutes label column not found: {minutes_label_col}")
        minutes_label_col = fallback

    y_minutes = pd.to_numeric(df[minutes_label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    flow_cols = list(flow_label_columns or [])
    if flow_cols:
        flow_raw = _numeric_frame_with_nans_for_missing(df, flow_cols).to_numpy(dtype=np.float32, copy=False)
        flow_observed = np.isfinite(flow_raw)
        flow_values = np.nan_to_num(flow_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    else:
        flow_values = np.zeros((len(df), 0), dtype=np.float32)
        flow_observed = np.zeros((len(df), 0), dtype=bool)
    lineup_available = (
        pd.to_numeric(df.get("lineup_available", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int64) > 0
    )

    x_by_idx = x
    y_by_idx = y_minutes
    flow_by_idx = flow_values
    flow_observed_by_idx = flow_observed
    lineup_by_idx = lineup_available

    game_feats_df = _numeric_frame(df, game_feature_columns) if game_feature_columns else pd.DataFrame(index=df.index)
    team_feats_df = _numeric_frame(df, team_feature_columns) if team_feature_columns else pd.DataFrame(index=df.index)

    examples: list[GameLevelExample] = []
    grouped = df.groupby(["game_id_norm", "game_date"], sort=False).indices
    for (game_id_norm, game_date), idx in grouped.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        game_df = df.iloc[idx_arr]
        home_id, away_id = _resolve_home_away_team_ids(game_df)

        player_features = np.zeros((2, max_players_per_team, len(feature_columns)), dtype=np.float32)
        player_valid = np.zeros((2, max_players_per_team), dtype=bool)
        player_ids = np.zeros((2, max_players_per_team), dtype=np.int64)
        team_ids = np.zeros((2,), dtype=np.int64)
        y_minutes_arr = np.zeros((2, max_players_per_team), dtype=np.float32)
        flow_arr = np.zeros((2, max_players_per_team, len(flow_cols)), dtype=np.float32)
        flow_obs_arr = np.zeros((2, max_players_per_team, len(flow_cols)), dtype=bool)
        lineup_arr = np.zeros((2, max_players_per_team), dtype=bool)

        for side_idx, team_id in enumerate([home_id, away_id]):
            if team_id is None:
                continue
            team_rows = game_df.loc[pd.to_numeric(game_df["team_id"], errors="coerce") == int(team_id)]
            if team_rows.empty:
                continue
            team_rows = _sort_team_rows(
                team_rows,
                protected_prior_play_prob_floor=float(overflow_protected_prior_play_prob_floor),
                protected_prior_minutes_floor=float(overflow_protected_prior_minutes_floor),
                risk_weight_consecutive_active_dnp=float(overflow_risk_weight_consecutive_active_dnp),
                risk_weight_active_but_dnp_rate_last10=float(overflow_risk_weight_active_but_dnp_rate_last10),
                risk_weight_inactive_streak_len=float(overflow_risk_weight_inactive_streak_len),
                keep_weight_prior_play_prob=float(overflow_keep_weight_prior_play_prob),
                keep_weight_prior_minutes=float(overflow_keep_weight_prior_minutes),
            ).head(max_players_per_team)
            team_ids[side_idx] = int(team_id)
            local_idx = team_rows.index.to_numpy(dtype=np.int64)
            n = local_idx.shape[0]

            player_features[side_idx, :n] = x_by_idx[local_idx]
            player_valid[side_idx, :n] = True
            player_ids[side_idx, :n] = (
                pd.to_numeric(team_rows["player_id"], errors="coerce").fillna(0).astype("int64").to_numpy(dtype=np.int64)
            )
            y_minutes_arr[side_idx, :n] = y_by_idx[local_idx]
            if flow_cols:
                flow_arr[side_idx, :n, :] = flow_by_idx[local_idx, :]
                flow_obs_arr[side_idx, :n, :] = flow_observed_by_idx[local_idx, :]
            lineup_arr[side_idx, :n] = lineup_by_idx[local_idx]

        if bool(player_valid[0].sum() == 0 and player_valid[1].sum() == 0):
            continue

        # Drop malformed games where either side has no valid/non-positive team id rows,
        # or where a side cannot satisfy the 240-minute team constraint under 48-minute caps.
        if int(team_ids[0]) <= 0 or int(team_ids[1]) <= 0:
            continue
        if int(player_valid[0].sum()) < int(min_valid_players_per_team):
            continue
        if int(player_valid[1].sum()) < int(min_valid_players_per_team):
            continue

        if game_feature_columns:
            gvals = game_feats_df.iloc[idx_arr].mean(axis=0, skipna=True).to_numpy(dtype=np.float32)
            gvals = np.nan_to_num(gvals, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            gvals = np.zeros((0,), dtype=np.float32)

        if team_feature_columns:
            tvals = np.zeros((2, len(team_feature_columns)), dtype=np.float32)
            for side_idx, team_id in enumerate(team_ids.tolist()):
                if team_id <= 0:
                    continue
                mask = pd.to_numeric(game_df["team_id"], errors="coerce") == int(team_id)
                if bool(mask.any()):
                    vals = team_feats_df.loc[mask].mean(axis=0, skipna=True).to_numpy(dtype=np.float32)
                    tvals[side_idx] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            tvals = np.zeros((2, 0), dtype=np.float32)

        examples.append(
            GameLevelExample(
                player_features=player_features,
                player_valid_mask=player_valid,
                player_ids=player_ids,
                team_ids=team_ids,
                y_minutes=y_minutes_arr,
                flow_targets=flow_arr,
                flow_observed_mask=flow_obs_arr,
                lineup_available=lineup_arr,
                game_features=gvals,
                team_features=tvals,
                game_id_norm=str(game_id_norm),
                game_date=str(pd.Timestamp(game_date).date().isoformat()),
            )
        )

    if not examples:
        raise ValueError("No game-level examples were built from the provided frame")
    return examples


def collate_game_level_examples(batch: list[GameLevelExample]) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("batch must be non-empty")

    bsz = len(batch)
    n_feat = batch[0].player_features.shape[2]
    n_flow = batch[0].flow_targets.shape[2]
    n_game_feat = batch[0].game_features.shape[0]
    n_team_feat = batch[0].team_features.shape[1]

    player_features = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_feat), dtype=torch.float32)
    player_valid_mask = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.bool)
    player_ids = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.long)
    team_ids = torch.zeros((bsz, 2), dtype=torch.long)
    y_minutes = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.float32)
    flow_targets = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_flow), dtype=torch.float32)
    flow_observed_mask = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM, n_flow), dtype=torch.bool)
    lineup_available = torch.zeros((bsz, 2, MAX_PLAYERS_PER_TEAM), dtype=torch.bool)
    game_features = torch.zeros((bsz, n_game_feat), dtype=torch.float32)
    team_features = torch.zeros((bsz, 2, n_team_feat), dtype=torch.float32)

    game_ids: list[str] = []
    game_dates: list[str] = []

    for i, ex in enumerate(batch):
        player_features[i] = torch.from_numpy(ex.player_features.astype(np.float32, copy=False))
        player_valid_mask[i] = torch.from_numpy(ex.player_valid_mask.astype(bool, copy=False))
        player_ids[i] = torch.from_numpy(ex.player_ids.astype(np.int64, copy=False))
        team_ids[i] = torch.from_numpy(ex.team_ids.astype(np.int64, copy=False))
        y_minutes[i] = torch.from_numpy(ex.y_minutes.astype(np.float32, copy=False))
        if n_flow > 0:
            flow_targets[i] = torch.from_numpy(ex.flow_targets.astype(np.float32, copy=False))
            flow_observed_mask[i] = torch.from_numpy(ex.flow_observed_mask.astype(bool, copy=False))
        lineup_available[i] = torch.from_numpy(ex.lineup_available.astype(bool, copy=False))
        if n_game_feat > 0:
            game_features[i] = torch.from_numpy(ex.game_features.astype(np.float32, copy=False))
        if n_team_feat > 0:
            team_features[i] = torch.from_numpy(ex.team_features.astype(np.float32, copy=False))
        game_ids.append(ex.game_id_norm)
        game_dates.append(ex.game_date)

    return {
        "player_features": player_features,
        "player_valid_mask": player_valid_mask,
        "player_ids": player_ids,
        "team_ids": team_ids,
        "y_minutes": y_minutes,
        "flow_targets": flow_targets,
        "flow_observed_mask": flow_observed_mask,
        "lineup_available": lineup_available,
        "game_features": game_features,
        "team_features": team_features,
        "game_id_norm": game_ids,
        "game_date": game_dates,
    }


class GameTransformerV2(nn.Module):
    """Cross-team game transformer with joint active-set and minutes heads."""

    def __init__(
        self,
        num_player_features: int,
        *,
        num_game_features: int,
        num_team_features: int,
        d_model: int = 192,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 6,
        dropout: float = 0.1,
        ff_mult: float = 4.0,
        min_active_count: int = 5,
        max_active_count: int = 13,
        total_minutes_per_team: float = 240.0,
        max_minutes_per_player: float = 48.0,
        flow_coupling_type: str = "affine",
        flow_num_blocks: int = 4,
        flow_scale_clip: float = 3.0,
        flow_mean_ctx_weight: float = 1.0,
        flow_context_mode: str = "attention",
        include_pf_in_flow_targets: bool = False,
    ) -> None:
        super().__init__()
        if num_player_features <= 0:
            raise ValueError("num_player_features must be > 0")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model")

        self.num_player_features = int(num_player_features)
        self.num_game_features = int(num_game_features)
        self.num_team_features = int(num_team_features)
        self.d_model = int(d_model)
        self.flow_target_columns = flow_target_columns(include_pf=bool(include_pf_in_flow_targets))
        self.num_flow_stats = int(len(self.flow_target_columns))

        self.player_proj = nn.Linear(int(num_player_features), int(d_model))
        self.game_proj = nn.Linear(int(num_game_features), int(d_model)) if int(num_game_features) > 0 else None
        self.team_proj = nn.Linear(int(num_team_features), int(d_model)) if int(num_team_features) > 0 else None

        self.game_token = nn.Parameter(torch.randn(int(d_model)) * 0.02)
        self.team_tokens = nn.Parameter(torch.randn(2, int(d_model)) * 0.02)

        # token types: 0=game, 1=team, 2=player
        self.token_type_embedding = nn.Embedding(3, int(d_model))
        # side ids: 0=home, 1=away, 2=neutral(game)
        self.side_embedding = nn.Embedding(3, int(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(round(float(ff_mult) * int(d_model))),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))

        self.active_head = JointActiveSetHead(
            d_model=int(d_model),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            min_active_count=int(min_active_count),
            max_active_count=int(max_active_count),
        )
        self.minutes_head = JointMinutesHead(
            d_model=int(d_model),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            total_minutes_per_team=float(total_minutes_per_team),
            max_minutes_per_player=float(max_minutes_per_player),
        )
        self.flow_head = JointGameFlow(
            d_model=int(d_model),
            num_stats=int(self.num_flow_stats),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            num_blocks=int(flow_num_blocks),
            coupling_type=str(flow_coupling_type),
            scale_clip=float(flow_scale_clip),
            mean_ctx_weight=float(flow_mean_ctx_weight),
            context_mode=str(flow_context_mode),
        )

        token_type_ids = [0, 1, *([2] * MAX_PLAYERS_PER_TEAM), 1, *([2] * MAX_PLAYERS_PER_TEAM)]
        side_ids = [2, 0, *([0] * MAX_PLAYERS_PER_TEAM), 1, *([1] * MAX_PLAYERS_PER_TEAM)]
        team_index = [0] * MAX_PLAYERS_PER_TEAM + [1] * MAX_PLAYERS_PER_TEAM
        self.register_buffer("_token_type_ids", torch.tensor(token_type_ids, dtype=torch.long), persistent=False)
        self.register_buffer("_side_ids", torch.tensor(side_ids, dtype=torch.long), persistent=False)
        self.register_buffer("_player_team_index", torch.tensor(team_index, dtype=torch.long), persistent=False)

    def _build_sequence(
        self,
        player_features: torch.Tensor,
        game_features: torch.Tensor | None,
        team_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if player_features.ndim != 4:
            raise ValueError("player_features must have shape (B,2,15,F)")
        if player_features.shape[1] != 2 or player_features.shape[2] != MAX_PLAYERS_PER_TEAM:
            raise ValueError(f"player_features must have shape (B,2,{MAX_PLAYERS_PER_TEAM},F)")

        bsz = player_features.shape[0]
        p_flat = torch.cat([player_features[:, 0], player_features[:, 1]], dim=1)
        p_tok = self.player_proj(p_flat)
        p_home = p_tok[:, :MAX_PLAYERS_PER_TEAM]
        p_away = p_tok[:, MAX_PLAYERS_PER_TEAM:]

        g_tok = self.game_token.unsqueeze(0).unsqueeze(1).expand(bsz, 1, -1)
        if self.game_proj is not None:
            if game_features is None:
                raise ValueError("game_features is required when num_game_features > 0")
            if game_features.ndim != 2 or game_features.shape != (bsz, self.num_game_features):
                raise ValueError("game_features must have shape (B,G)")
            g_tok = g_tok + self.game_proj(game_features).unsqueeze(1)

        t_base = self.team_tokens.unsqueeze(0).expand(bsz, -1, -1)
        if self.team_proj is not None:
            if team_features is None:
                raise ValueError("team_features is required when num_team_features > 0")
            if team_features.ndim != 3 or team_features.shape != (bsz, 2, self.num_team_features):
                raise ValueError("team_features must have shape (B,2,T)")
            t_base = t_base + self.team_proj(team_features)

        seq = torch.cat(
            [
                g_tok,
                t_base[:, 0:1, :],
                p_home,
                t_base[:, 1:2, :],
                p_away,
            ],
            dim=1,
        )
        if seq.shape[1] != 33:
            raise RuntimeError(f"Unexpected sequence length: {seq.shape[1]}")

        type_emb = self.token_type_embedding(self._token_type_ids).unsqueeze(0)
        side_emb = self.side_embedding(self._side_ids).unsqueeze(0)
        seq = seq + type_emb + side_emb
        return self.dropout(seq)

    def forward(
        self,
        player_features: torch.Tensor,
        player_valid_mask: torch.Tensor,
        *,
        game_features: torch.Tensor | None = None,
        team_features: torch.Tensor | None = None,
        sample_active: bool = False,
        active_temperature: float = 1.0,
        target_counts: torch.Tensor | None = None,
        use_target_counts: bool = False,
        target_active_mask: torch.Tensor | None = None,
        use_target_active_mask: bool = False,
        minutes_use_target_active: bool = False,
        run_flow: bool = False,
        flow_targets: torch.Tensor | None = None,
        flow_observed_mask: torch.Tensor | None = None,
    ) -> GameTransformerV2Outputs:
        """Forward pass for one batch of full-game tensors."""

        if player_valid_mask.ndim != 3:
            raise ValueError("player_valid_mask must have shape (B,2,15)")
        if player_valid_mask.shape[:3] != player_features.shape[:3]:
            raise ValueError("player_valid_mask must align with player_features")

        bsz = player_features.shape[0]
        valid_home = player_valid_mask[:, 0].to(dtype=torch.bool)
        valid_away = player_valid_mask[:, 1].to(dtype=torch.bool)
        valid_flat = torch.cat([valid_home, valid_away], dim=1)
        player_team_index = self._player_team_index.unsqueeze(0).expand(bsz, -1)

        seq = self._build_sequence(player_features, game_features, team_features)
        seq_valid = torch.cat(
            [
                torch.ones((bsz, 1), dtype=torch.bool, device=seq.device),
                torch.ones((bsz, 1), dtype=torch.bool, device=seq.device),
                valid_home,
                torch.ones((bsz, 1), dtype=torch.bool, device=seq.device),
                valid_away,
            ],
            dim=1,
        )
        encoded = self.encoder(seq, src_key_padding_mask=~seq_valid)
        encoded = self.final_norm(encoded)

        game_state = encoded[:, 0, :]
        team_states = torch.stack([encoded[:, 1, :], encoded[:, 17, :]], dim=1)
        player_states = torch.cat([encoded[:, 2:17, :], encoded[:, 18:33, :]], dim=1)

        target_active_flat: torch.Tensor | None = None
        if target_active_mask is not None:
            if target_active_mask.ndim != 3 or target_active_mask.shape != player_valid_mask.shape:
                raise ValueError("target_active_mask must have shape (B,2,15)")
            target_active_flat = torch.cat(
                [target_active_mask[:, 0].to(dtype=torch.bool), target_active_mask[:, 1].to(dtype=torch.bool)],
                dim=1,
            )

        active_out = self.active_head(
            player_states,
            team_states,
            player_team_index,
            valid_flat,
            sample=bool(sample_active),
            temperature=float(active_temperature),
            target_counts=target_counts,
            use_target_counts=bool(use_target_counts),
            target_active_mask=target_active_flat,
            use_target_active_mask=bool(use_target_active_mask),
        )

        if bool(minutes_use_target_active):
            if target_active_flat is None:
                raise ValueError("minutes_use_target_active=True requires target_active_mask")
            minutes_active_mask = target_active_flat
        else:
            minutes_active_mask = active_out.active_mask

        minutes_out = self.minutes_head(
            player_states,
            team_states,
            player_team_index,
            valid_flat,
            minutes_active_mask,
        )

        flow_out: JointGameFlowOutputs | None = None
        if bool(run_flow) or flow_targets is not None:
            if flow_targets is None:
                raise ValueError("run_flow=True requires flow_targets")
            if flow_targets.ndim != 4:
                raise ValueError("flow_targets must have shape (B,2,15,S)")
            if flow_targets.shape[:3] != player_features.shape[:3]:
                raise ValueError("flow_targets must align with player_features first 3 dims")
            if flow_targets.shape[3] != self.num_flow_stats:
                raise ValueError(
                    f"flow_targets stat dim mismatch: expected {self.num_flow_stats}, got {flow_targets.shape[3]}"
                )
            flow_flat = torch.cat([flow_targets[:, 0], flow_targets[:, 1]], dim=1)

            flow_obs_flat: torch.Tensor | None = None
            if flow_observed_mask is not None:
                if flow_observed_mask.shape != flow_targets.shape:
                    raise ValueError("flow_observed_mask must match flow_targets shape")
                flow_obs_flat = torch.cat(
                    [
                        flow_observed_mask[:, 0].to(dtype=torch.bool),
                        flow_observed_mask[:, 1].to(dtype=torch.bool),
                    ],
                    dim=1,
                )

            flow_out = self.flow_head(
                flow_flat,
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                valid_mask=valid_flat,
                observed_mask=flow_obs_flat,
            )

        return GameTransformerV2Outputs(
            game_state=game_state,
            team_states=team_states,
            player_states=player_states,
            player_valid_mask=valid_flat,
            player_team_index=player_team_index,
            active=active_out,
            minutes=minutes_out,
            flow=flow_out,
        )


def build_game_transformer_v2(config: GameTransformerV2Config) -> GameTransformerV2:
    return GameTransformerV2(
        num_player_features=len(config.feature_columns),
        num_game_features=len(config.game_feature_columns),
        num_team_features=len(config.team_feature_columns),
        d_model=int(config.d_model),
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        num_heads=int(config.num_heads),
        dropout=float(config.dropout),
        ff_mult=float(config.ff_mult),
        min_active_count=int(config.min_active_count),
        max_active_count=int(config.max_active_count),
        total_minutes_per_team=float(config.total_minutes_per_team),
        max_minutes_per_player=float(config.max_minutes_per_player),
        flow_coupling_type=str(config.flow_coupling_type),
        flow_num_blocks=int(config.flow_num_blocks),
        flow_scale_clip=float(config.flow_scale_clip),
        flow_mean_ctx_weight=float(config.flow_mean_ctx_weight),
        flow_context_mode=str(getattr(config, "flow_context_mode", "mean")),
        include_pf_in_flow_targets=bool(config.include_pf_in_flow_targets),
    )
