"""Guardrails for rotation_set_minutes live inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from projections.models.rotalloc import waterfill_redistribute

KEY_COLS: tuple[str, str, str] = ("game_id", "team_id", "player_id")


@dataclass(frozen=True)
class RotationGuardrailResult:
    minutes_p50: pd.Series
    summary: dict[str, Any]
    team_game_summary: pd.DataFrame
    tail_minutes_top: pd.DataFrame


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(False).astype(bool)
    return series.fillna(False).astype(bool)


def _row_lineup_present(df: pd.DataFrame) -> pd.Series:
    ts = df.get("lineup_timestamp")
    status = df.get("lineup_status")
    roster_status = df.get("lineup_roster_status")
    present = pd.Series(False, index=df.index)
    if ts is not None:
        present = present | pd.to_datetime(ts, errors="coerce", utc=True).notna()
    if status is not None:
        present = present | status.notna()
    if roster_status is not None:
        present = present | roster_status.notna()
    return present


def apply_rotation_minutes_guardrails(
    df: pd.DataFrame,
    *,
    rotation_p50_col: str,
    baseline_p50_col: str,
    minutes_features_row_missing_col: str = "minutes_features_row_missing",
    injury_snapshot_missing_col: str = "injury_snapshot_missing",
    out_flag_col: str = "is_out",
    status_col: str = "status",
    blend_weight: float = 0.2,
    injury_coverage_threshold: float = 0.5,
    dnp_tail_minutes_threshold: float = 8.0,
    cap_max_minutes: float | None = None,
    pathology_max_minutes_threshold: float = 48.0,
    pathology_n_gt_40_threshold: int = 3,
    pathology_top5_sum_threshold: float = 210.0,
    team_target_minutes: float = 240.0,
) -> RotationGuardrailResult:
    """Apply guardrails and return guarded p50 minutes.

    Rules:
      1) Row fallback: minutes_features_row_missing==1 -> use baseline p50 for that row.
      2) Team blend: if lineup_fields_present==False AND injury_coverage_rate < threshold:
         minutes = w*rotation + (1-w)*baseline for the whole team-game.
      3) Tail clamp: if a team-game allocates > threshold minutes to DNP-proxy rows
         (play_prob==0 equivalent, e.g. status OUT), fall back to baseline for that team-game.
      4) Optional cap: when cap_max_minutes is set, cap per-player p50 and redistribute
         within the team-game back to `team_target_minutes` while preserving fallback rows.
      5) Pathology fallback: for obviously impossible allocations (e.g. OT-level minutes or
         extreme concentration), fall back to baseline for the entire team-game.
    """

    if blend_weight < 0.0 or blend_weight > 1.0:
        raise ValueError("blend_weight must be in [0, 1]")

    missing_keys = [c for c in KEY_COLS if c not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing required key columns: {missing_keys}")
    if rotation_p50_col not in df.columns or baseline_p50_col not in df.columns:
        raise ValueError(f"Missing required prediction columns: {rotation_p50_col}, {baseline_p50_col}")

    work = df.copy()
    rot = pd.to_numeric(work[rotation_p50_col], errors="coerce").fillna(0.0).astype(float)
    base = pd.to_numeric(work[baseline_p50_col], errors="coerce").fillna(0.0).astype(float)

    # Treat explicit OUT rows as hard-zero for the rotation overlay. In practice the model
    # may still emit non-zero minutes for OUT players; if we allow those through, the
    # subsequent 240-minute scaling can inflate them dramatically and trip the tail clamp,
    # disabling the overlay for the whole team-game.
    out_mask = pd.Series(False, index=work.index, dtype=bool)
    if out_flag_col in work.columns:
        out_flag = pd.to_numeric(work[out_flag_col], errors="coerce").fillna(0).astype(int)
        out_mask = out_mask | (out_flag.to_numpy(dtype=int) == 1)
    if status_col in work.columns:
        status = work[status_col].astype("string").str.upper()
        out_mask = out_mask | status.eq("OUT").fillna(False).to_numpy(dtype=bool)

    if out_mask.any():
        rot = rot.where(~out_mask, 0.0)

    lineup_present_row = _row_lineup_present(work)

    injury_missing = pd.to_numeric(work.get(injury_snapshot_missing_col, 1), errors="coerce")
    injury_missing = injury_missing.fillna(1).astype(float).clip(lower=0.0, upper=1.0)

    group_cols = ["game_id", "team_id"]
    lineup_present_team = lineup_present_row.groupby([work[c] for c in group_cols], sort=False).any()
    injury_coverage_rate = 1.0 - injury_missing.groupby([work[c] for c in group_cols], sort=False).mean()

    team_index = pd.MultiIndex.from_frame(work.loc[:, group_cols])
    lineup_present_aligned = lineup_present_team.reindex(team_index, fill_value=False).to_numpy(dtype=bool)
    injury_cov_aligned = injury_coverage_rate.reindex(team_index).fillna(0.0).to_numpy(dtype=float)

    blend_team_mask = (~lineup_present_aligned) & (injury_cov_aligned < float(injury_coverage_threshold))

    guarded = rot.copy()
    if blend_team_mask.any():
        w = float(blend_weight)
        guarded = guarded.where(~blend_team_mask, w * rot + (1.0 - w) * base)

    row_missing = pd.to_numeric(work.get(minutes_features_row_missing_col, 0), errors="coerce").fillna(0).astype(int)
    row_fallback_mask = row_missing.to_numpy(dtype=int) == 1
    if row_fallback_mask.any():
        guarded = guarded.where(~row_fallback_mask, base)

    # Enforce per-team 240-minute totals by scaling the non-fallback rows.
    df_team = pd.DataFrame(
        {
            "game_id": work["game_id"],
            "team_id": work["team_id"],
            "guarded": guarded,
            "fixed": row_fallback_mask,
        }
    )
    fixed_sum = df_team.loc[df_team["fixed"]].groupby(group_cols, sort=False)["guarded"].sum()
    total_sum = df_team.groupby(group_cols, sort=False)["guarded"].sum()
    adjustable_sum = (total_sum - fixed_sum).fillna(total_sum)
    target_minus_fixed = (float(team_target_minutes) - fixed_sum).fillna(float(team_target_minutes))
    # Avoid divide-by-zero; if adjustable_sum==0, keep as-is.
    scale = (target_minus_fixed / adjustable_sum.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scale = scale.clip(lower=0.0)
    scale_aligned = scale.reindex(team_index).fillna(1.0).to_numpy(dtype=float)

    # Apply scaling only to non-fixed rows.
    scaled = guarded.to_numpy(dtype=float)
    scaled[~row_fallback_mask] = scaled[~row_fallback_mask] * scale_aligned[~row_fallback_mask]
    guarded = pd.Series(np.clip(scaled, 0.0, None), index=work.index, dtype=float)

    # Tail minutes proxy: minutes assigned to "DNP" rows (explicit OUT flag or status string).
    tail_pre = pd.DataFrame(
        {
            "game_id": work["game_id"],
            "team_id": work["team_id"],
            "tail_minutes_dnp_pre_clamp": np.where(out_mask.to_numpy(dtype=bool), guarded.to_numpy(dtype=float), 0.0),
        }
    )
    tail_pre_team = (
        tail_pre.groupby(group_cols, sort=False)["tail_minutes_dnp_pre_clamp"].sum().reset_index()
    )
    tail_pre_team["tail_clamped"] = tail_pre_team["tail_minutes_dnp_pre_clamp"] > float(dnp_tail_minutes_threshold)

    clamp_index = pd.MultiIndex.from_frame(tail_pre_team.loc[tail_pre_team["tail_clamped"], group_cols])
    clamp_aligned = team_index.isin(clamp_index)
    if clamp_aligned.any():
        guarded = guarded.where(~clamp_aligned, base)

    tail_post = pd.DataFrame(
        {
            "game_id": work["game_id"],
            "team_id": work["team_id"],
            "tail_minutes_dnp_post_clamp": np.where(out_mask.to_numpy(dtype=bool), guarded.to_numpy(dtype=float), 0.0),
        }
    )
    tail_post_team = (
        tail_post.groupby(group_cols, sort=False)["tail_minutes_dnp_post_clamp"].sum().reset_index()
    )

    # Pathology tripwires (team-game fallback to baseline).
    # This is intentionally conservative: when triggered, we revert the whole team-game to baseline.
    metrics = pd.DataFrame(
        {
            "game_id": work["game_id"],
            "team_id": work["team_id"],
            "guarded": guarded.to_numpy(dtype=float),
            "baseline": base.to_numpy(dtype=float),
        }
    )

    def _topk_sum(values: pd.Series, k: int) -> float:
        arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if arr.size == 0:
            return 0.0
        if arr.size <= k:
            return float(arr.sum())
        # partial selection for speed/determinism
        part = np.partition(arr, -k)[-k:]
        return float(part.sum())

    group = metrics.groupby(group_cols, sort=False)
    metrics_team = group["guarded"].agg(
        max_minutes="max",
        n_gt_40=lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 40.0).sum()),
        top5_sum=lambda s: _topk_sum(s, 5),
    ).reset_index()

    max_thr = float(pathology_max_minutes_threshold)
    if not np.isfinite(max_thr) or max_thr <= 0.0:
        raise ValueError("pathology_max_minutes_threshold must be a positive, finite float")
    n_thr = int(pathology_n_gt_40_threshold)
    if n_thr < 1:
        n_thr = 1
    top5_thr = float(pathology_top5_sum_threshold)
    if not np.isfinite(top5_thr) or top5_thr <= 0.0:
        raise ValueError("pathology_top5_sum_threshold must be a positive, finite float")

    max_trigger = metrics_team["max_minutes"] > max_thr
    n_trigger = metrics_team["n_gt_40"] >= n_thr
    top5_trigger = metrics_team["top5_sum"] >= top5_thr

    fallback_team = max_trigger | n_trigger | top5_trigger
    if fallback_team.any():
        def _reason_row(row: pd.Series) -> str:
            parts: list[str] = []
            if bool(row.get("max_trigger", False)):
                parts.append("max_minutes")
            if bool(row.get("n_trigger", False)):
                parts.append("n_gt_40")
            if bool(row.get("top5_trigger", False)):
                parts.append("top5_sum")
            return ",".join(parts) if parts else "unknown"

        metrics_team = metrics_team.copy()
        metrics_team["max_trigger"] = max_trigger.to_numpy(dtype=bool)
        metrics_team["n_trigger"] = n_trigger.to_numpy(dtype=bool)
        metrics_team["top5_trigger"] = top5_trigger.to_numpy(dtype=bool)
        metrics_team["pathology_fallback"] = fallback_team.to_numpy(dtype=bool)
        metrics_team["pathology_reason"] = metrics_team.apply(_reason_row, axis=1)

        fallback_index = pd.MultiIndex.from_frame(metrics_team.loc[metrics_team["pathology_fallback"], group_cols])
        fallback_aligned = pd.MultiIndex.from_frame(work.loc[:, group_cols]).isin(fallback_index)
        guarded = guarded.where(~fallback_aligned, base)
    else:
        metrics_team["pathology_fallback"] = False
        metrics_team["pathology_reason"] = ""

    if cap_max_minutes is not None:
        cap_val = float(cap_max_minutes)

        fixed_mask = row_fallback_mask | clamp_aligned
        # Don't cap/redistribute on team-games that already fell back to baseline.
        if "pathology_fallback" in metrics_team.columns and bool(metrics_team["pathology_fallback"].any()):
            fallbacks = metrics_team.loc[metrics_team["pathology_fallback"], group_cols]
            fallback_index = pd.MultiIndex.from_frame(fallbacks)
            fixed_mask = fixed_mask | pd.MultiIndex.from_frame(work.loc[:, group_cols]).isin(fallback_index)

        out_mask_arr = out_mask.to_numpy(dtype=bool)
        eligible_mask = (~fixed_mask) & (~out_mask_arr)
        if "play_prob" in work.columns:
            play_prob = pd.to_numeric(work["play_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            eligible_mask = eligible_mask & (play_prob > 0.0)

        guarded_arr = guarded.to_numpy(dtype=np.float64)
        weights_arr = np.maximum(guarded_arr, 1e-6)
        pre_cap_max = float(np.nanmax(guarded_arr)) if guarded_arr.size else 0.0
        pre_cap_rate = float((guarded_arr > cap_val).mean()) if guarded_arr.size else 0.0

        keys = work.loc[:, group_cols]
        for _, idx in keys.groupby(group_cols, sort=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            fixed_sum_team = float(np.sum(guarded_arr[idx_arr[fixed_mask[idx_arr]]]))
            target_adj = float(team_target_minutes) - fixed_sum_team
            if target_adj <= 0.0:
                guarded_arr[idx_arr[~fixed_mask[idx_arr]]] = 0.0
                continue

            eligible_team = idx_arr[eligible_mask[idx_arr]]
            if eligible_team.size == 0:
                continue

            noneligible_team = idx_arr[(~fixed_mask[idx_arr]) & (~eligible_mask[idx_arr])]
            if noneligible_team.size:
                guarded_arr[noneligible_team] = 0.0

            minutes_in = guarded_arr[eligible_team]
            weights_in = weights_arr[eligible_team]
            guarded_arr[eligible_team] = waterfill_redistribute(
                minutes_in,
                weights_in,
                np.ones_like(minutes_in, dtype=bool),
                cap_max=cap_val,
                target_sum=target_adj,
            )

        guarded = pd.Series(guarded_arr, index=work.index, dtype=float)

    # Build diagnostics tables.
    team_game = pd.DataFrame(
        {
            "game_id": lineup_present_team.index.get_level_values(0),
            "team_id": lineup_present_team.index.get_level_values(1),
            "lineup_fields_present": lineup_present_team.to_numpy(dtype=bool),
            "injury_coverage_rate": injury_coverage_rate.to_numpy(dtype=float),
        }
    )
    fallback_by_team = (
        pd.Series(row_fallback_mask, index=work.index)
        .groupby([work[c] for c in group_cols], sort=False)
        .agg(["sum", "mean"])
        .rename(columns={"sum": "row_fallback_count", "mean": "row_fallback_rate"})
        .reset_index()
    )
    team_game = team_game.merge(fallback_by_team, on=["game_id", "team_id"], how="left")
    team_game["row_fallback_count"] = pd.to_numeric(team_game["row_fallback_count"], errors="coerce").fillna(0).astype(int)
    team_game["row_fallback_rate"] = pd.to_numeric(team_game["row_fallback_rate"], errors="coerce").fillna(0.0).astype(float)
    team_game["blend_applied"] = (~team_game["lineup_fields_present"]) & (
        team_game["injury_coverage_rate"] < float(injury_coverage_threshold)
    )

    team_game = team_game.merge(tail_pre_team, on=group_cols, how="left")
    team_game = team_game.merge(tail_post_team, on=group_cols, how="left")
    if not metrics_team.empty:
        keep_metrics = [
            "game_id",
            "team_id",
            "max_minutes",
            "n_gt_40",
            "top5_sum",
            "pathology_fallback",
            "pathology_reason",
        ]
        keep_metrics = [c for c in keep_metrics if c in metrics_team.columns]
        team_game = team_game.merge(metrics_team.loc[:, keep_metrics], on=group_cols, how="left")
        team_game["pathology_fallback"] = _coerce_bool_series(team_game.get("pathology_fallback", False))
        if "pathology_reason" not in team_game.columns:
            team_game["pathology_reason"] = ""
        team_game["pathology_reason"] = team_game["pathology_reason"].fillna("").astype("string")
    team_game["tail_minutes_dnp_pre_clamp"] = (
        pd.to_numeric(team_game["tail_minutes_dnp_pre_clamp"], errors="coerce").fillna(0.0).astype(float)
    )
    team_game["tail_minutes_dnp_post_clamp"] = (
        pd.to_numeric(team_game["tail_minutes_dnp_post_clamp"], errors="coerce").fillna(0.0).astype(float)
    )
    team_game["tail_clamped"] = _coerce_bool_series(team_game.get("tail_clamped", False))

    tail_team = (
        team_game.loc[:, ["game_id", "team_id", "tail_minutes_dnp_pre_clamp", "tail_clamped"]]
        .sort_values("tail_minutes_dnp_pre_clamp", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    overlay_row_applied_mask = (~row_fallback_mask) & (~clamp_aligned)

    summary = {
        "rows": int(len(work)),
        "row_fallback_rate": float(row_fallback_mask.mean()) if len(work) else 0.0,
        "team_blend_rate": float(team_game["blend_applied"].mean()) if len(team_game) else 0.0,
        "rows_in_blended_team_games_rate": float(blend_team_mask.mean()) if len(work) else 0.0,
        "overlay_rows_rate": float(overlay_row_applied_mask.mean()) if len(work) else 0.0,
        "tail_clamped_team_games": int(team_game["tail_clamped"].sum()) if len(team_game) else 0,
        "tail_minutes_dnp_pre_clamp_max": float(team_game["tail_minutes_dnp_pre_clamp"].max()) if len(team_game) else 0.0,
        "pathology": {
            "max_minutes_threshold": float(max_thr),
            "n_gt_40_threshold": int(n_thr),
            "top5_sum_threshold": float(top5_thr),
            "fallback_team_games": int(team_game.get("pathology_fallback", pd.Series(dtype=bool)).sum()) if len(team_game) else 0,
        },
    }
    if "pathology_fallback" in team_game.columns and bool(team_game["pathology_fallback"].any()):
        cols = [
            "game_id",
            "team_id",
            "max_minutes",
            "n_gt_40",
            "top5_sum",
            "pathology_reason",
        ]
        cols = [c for c in cols if c in team_game.columns]
        summary["pathology"]["team_games_top"] = (
            team_game.loc[team_game["pathology_fallback"], cols]
            .sort_values(["max_minutes", "top5_sum"], ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
    if cap_max_minutes is not None:
        summary["cap_max_minutes"] = float(cap_max_minutes)
        summary["cap_pre_max_minutes"] = float(pre_cap_max)
        summary["cap_pre_rate"] = float(pre_cap_rate)
        summary["cap_post_max_minutes"] = float(guarded.max()) if len(work) else 0.0

    return RotationGuardrailResult(
        minutes_p50=guarded,
        summary=summary,
        team_game_summary=team_game,
        tail_minutes_top=tail_team,
    )
