"""Rolling calibration offsets for conformal quantiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest, ensure_columns

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RollingCalibrationConfig:
    """Configuration for rolling calibration windows."""

    window_days: int = 21
    min_n: int = 400
    tau: float = 150.0
    tau_bucket: float | None = 180.0
    use_buckets: bool = False
    min_n_bucket: int = 200
    use_spread_buckets: bool = False
    cold_start_min_n: int = 400
    p10_floor_guard: float = 0.10
    p90_floor_guard: float = 0.90
    min_recent_days: int = 0
    bucket_delta_cap: float | None = 0.30
    bucket_floor_relief: float = 0.06
    max_abs_delta_p10: float | None = None
    max_abs_delta_p90: float | None = None
    max_abs_delta_p10_lo: float = 0.20
    max_abs_delta_p10_hi: float = 0.40
    max_abs_delta_p90_lo: float = 0.15
    max_abs_delta_p90_hi: float = 0.30
    max_abs_delta_p10_lo_down: float = 0.10
    max_abs_delta_p10_hi_down: float = 0.15
    max_abs_delta_p10_hi_early: float | None = 0.40
    max_abs_delta_p90_lo_down: float = 0.10
    max_abs_delta_p90_hi_down: float = 0.20
    recent_target_rows: int = 300
    recency_window_days: int = 14
    recency_half_life_days: float = 12.0
    history_months: int = 0
    use_global_p10_delta: bool = False
    use_global_p90_delta: bool = True
    warmup_days_p10: int = 0
    warmup_days_p90: int = 0
    global_recent_target_rows: int = 300
    global_recent_target_rows_p10: int = 180
    global_recent_target_rows_p90: int = 300
    global_min_n: int = 200
    global_baseline_months: int = 2
    global_delta_cap: float = 0.30
    global_strata_cols: tuple[str, ...] = ("injury_snapshot_missing",)
    bucket_min_effective_sample_size: float = 360.0
    bucket_inherit_if_low_n: bool = True
    cap_guardrail_day: int = 7
    cap_guardrail_rate: float | None = None
    monotonicity_repair_threshold: float | None = 0.005
    max_width_delta_pct: float | None = None
    apply_center_width_projection: bool = True
    min_quantile_half_width: float = 0.05
    early_cap_days: int = 7
    max_abs_delta_p90_hi_early: float | None = 0.20
    one_sided_days: int = 0
    delta_smoothing_alpha_p10: float = 0.0
    delta_smoothing_alpha_p90: float = 0.0
    hysteresis_lower: float = 0.09
    hysteresis_upper: float = 0.11

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "window_days": self.window_days,
            "min_n": self.min_n,
            "tau": self.tau,
            "tau_bucket": self.tau_bucket if self.tau_bucket is not None else self.tau,
            "use_buckets": self.use_buckets,
            "min_n_bucket": self.min_n_bucket,
            "use_spread_buckets": self.use_spread_buckets,
            "cold_start_min_n": self.cold_start_min_n,
            "p10_floor_guard": self.p10_floor_guard,
            "p90_floor_guard": self.p90_floor_guard,
            "bucket_delta_cap": self.bucket_delta_cap,
            "bucket_floor_relief": self.bucket_floor_relief,
            "max_abs_delta_p10": self.max_abs_delta_p10,
            "max_abs_delta_p90": self.max_abs_delta_p90,
            "max_abs_delta_p10_lo": self.max_abs_delta_p10_lo,
            "max_abs_delta_p10_hi": self.max_abs_delta_p10_hi,
            "max_abs_delta_p90_lo": self.max_abs_delta_p90_lo,
            "max_abs_delta_p90_hi": self.max_abs_delta_p90_hi,
            "max_abs_delta_p10_lo_down": self.max_abs_delta_p10_lo_down,
            "max_abs_delta_p10_hi_down": self.max_abs_delta_p10_hi_down,
            "max_abs_delta_p90_lo_down": self.max_abs_delta_p90_lo_down,
            "max_abs_delta_p90_hi_down": self.max_abs_delta_p90_hi_down,
            "max_abs_delta_p10_hi_early": self.max_abs_delta_p10_hi_early,
            "recent_target_rows": self.recent_target_rows,
            "recency_window_days": self.recency_window_days,
            "recency_half_life_days": self.recency_half_life_days,
            "history_months": self.history_months,
            "use_global_p10_delta": self.use_global_p10_delta,
            "use_global_p90_delta": self.use_global_p90_delta,
            "warmup_days_p10": self.warmup_days_p10,
            "warmup_days_p90": self.warmup_days_p90,
            "global_recent_target_rows": self.global_recent_target_rows,
            "global_recent_target_rows_p10": self.global_recent_target_rows_p10,
            "global_recent_target_rows_p90": self.global_recent_target_rows_p90,
            "global_min_n": self.global_min_n,
            "global_baseline_months": self.global_baseline_months,
            "global_delta_cap": self.global_delta_cap,
            "global_strata_cols": ",".join(self.global_strata_cols),
            "bucket_min_effective_sample_size": self.bucket_min_effective_sample_size,
            "bucket_inherit_if_low_n": self.bucket_inherit_if_low_n,
            "cap_guardrail_day": self.cap_guardrail_day,
            "cap_guardrail_rate": self.cap_guardrail_rate
            if self.cap_guardrail_rate is not None
            else None,
            "monotonicity_repair_threshold": self.monotonicity_repair_threshold,
            "max_width_delta_pct": self.max_width_delta_pct,
            "apply_center_width_projection": self.apply_center_width_projection,
            "min_quantile_half_width": self.min_quantile_half_width,
            "one_sided_days": self.one_sided_days,
            "hysteresis_lower": self.hysteresis_lower,
            "hysteresis_upper": self.hysteresis_upper,
            "delta_smoothing_alpha_p10": self.delta_smoothing_alpha_p10,
            "delta_smoothing_alpha_p90": self.delta_smoothing_alpha_p90,
            "early_cap_days": self.early_cap_days,
            "max_abs_delta_p90_hi_early": self.max_abs_delta_p90_hi_early,
        }


@dataclass(slots=True)
class _GlobalAnchor:
    delta_p10: float
    delta_p90: float
    weight_recent_p10: float
    weight_recent_p90: float
    recent_rows: int
    baseline_rows: int


@dataclass(slots=True)
class MonotonicityStats:
    total: int
    lower: int
    upper: int


class _GlobalDeltaProvider:
    """Compute leak-safe global deltas with per-stratum crossfading."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_col: str,
        cfg: RollingCalibrationConfig,
        strata_cols: Sequence[str],
        default_delta_p10: float,
        default_delta_p90: float,
    ) -> None:
        self._frame = frame
        self._label_col = label_col
        self._cfg = cfg
        self._strata_cols = tuple(strata_cols)
        self._default_anchor = (
            float(default_delta_p10),
            float(default_delta_p90),
        )
        self._cache: dict[pd.Timestamp, dict[tuple[object, ...], _GlobalAnchor]] = {}

    def anchor(
        self,
        score_date: pd.Timestamp,
        values: dict[str, object] | None = None,
    ) -> _GlobalAnchor:
        normalized_date = pd.Timestamp(score_date).normalize()
        context = self._context_for(normalized_date)
        key = _stratum_key(self._strata_cols, values)
        if key in context:
            return context[key]
        fallback = context.get((), _GlobalAnchor(0.0, 0.0, 0.0, 0.0, 0, 0))
        return _GlobalAnchor(
            delta_p10=fallback.delta_p10,
            delta_p90=fallback.delta_p90,
            weight_recent_p10=0.0,
            weight_recent_p90=0.0,
            recent_rows=0,
            baseline_rows=fallback.baseline_rows,
        )

    def _context_for(self, score_date: pd.Timestamp) -> dict[tuple[object, ...], _GlobalAnchor]:
        cached = self._cache.get(score_date)
        if cached is not None:
            return cached
        context = self._build_context(score_date)
        self._cache[score_date] = context
        return context

    def _build_context(self, score_date: pd.Timestamp) -> dict[tuple[object, ...], _GlobalAnchor]:
        month_start = score_date.replace(day=1)
        baseline_start = month_start - pd.DateOffset(months=self._cfg.global_baseline_months)
        if self._cfg.global_baseline_months <= 0:
            baseline_mask = self._frame["game_date"] < month_start
        else:
            baseline_mask = (self._frame["game_date"] >= baseline_start) & (self._frame["game_date"] < month_start)
        recent_mask = (self._frame["game_date"] >= month_start) & (self._frame["game_date"] < score_date)

        baseline_frame = self._frame.loc[baseline_mask]
        recent_frame = self._frame.loc[recent_mask]

        baseline_stats, baseline_counts = self._fit_frame(baseline_frame)
        recent_stats, recent_counts = self._fit_frame(recent_frame)

        keys: set[tuple[object, ...]] = {()}
        keys |= set(baseline_stats.keys())
        keys |= set(recent_stats.keys())
        keys |= set(baseline_counts.keys())
        keys |= set(recent_counts.keys())

        context: dict[tuple[object, ...], _GlobalAnchor] = {}
        target_rows_p10 = max(1, self._cfg.global_recent_target_rows_p10 or self._cfg.global_recent_target_rows)
        target_rows_p90 = max(1, self._cfg.global_recent_target_rows_p90 or self._cfg.global_recent_target_rows)
        for key in keys:
            baseline_delta = (
                baseline_stats.get(key)
                or baseline_stats.get(())
                or self._default_anchor
            )
            recent_delta = (
                recent_stats.get(key)
                or recent_stats.get(())
                or baseline_delta
            )
            recent_rows = recent_counts.get(key, 0)
            weight_recent_p10 = min(1.0, recent_rows / target_rows_p10)
            weight_recent_p90 = min(1.0, recent_rows / target_rows_p90)
            blended_p10 = _blend_values(baseline_delta[0], recent_delta[0], weight_recent_p10)
            blended_p90 = _blend_values(baseline_delta[1], recent_delta[1], weight_recent_p90)
            blended_p10 = _clamp_abs_delta(blended_p10, self._cfg.global_delta_cap)
            blended_p90 = _clamp_abs_delta(blended_p90, self._cfg.global_delta_cap)
            if abs(blended_p10) - 1e-9 > self._cfg.global_delta_cap or abs(blended_p90) - 1e-9 > self._cfg.global_delta_cap:
                raise ValueError("Global delta exceeded configured cap.")
            context[key] = _GlobalAnchor(
                delta_p10=blended_p10,
                delta_p90=blended_p90,
                weight_recent_p10=weight_recent_p10,
                weight_recent_p90=weight_recent_p90,
                recent_rows=recent_rows,
                baseline_rows=baseline_counts.get(key, baseline_counts.get((), 0)),
            )
        return context

    def _fit_frame(
        self,
        frame: pd.DataFrame,
    ) -> tuple[dict[tuple[object, ...], tuple[float, float]], dict[tuple[object, ...], int]]:
        stats: dict[tuple[object, ...], tuple[float, float]] = {}
        counts: dict[tuple[object, ...], int] = {}
        if frame.empty:
            return stats, counts

        counts[()] = int(len(frame))
        stats[()] = (
            _solve_coverage_target(
                frame,
                0.0,
                label_col=self._label_col,
                quantile_col="p10_raw",
                target=self._cfg.p10_floor_guard,
            )[0],
            _solve_coverage_target(
                frame,
                0.0,
                label_col=self._label_col,
                quantile_col="p90_raw",
                target=self._cfg.p90_floor_guard,
            )[0],
        )

        if not self._strata_cols:
            return stats, counts

        grouped = frame.groupby(list(self._strata_cols), dropna=False)
        for key, group in grouped:
            counts[_ensure_tuple(key)] = int(len(group))
            if len(group) < self._cfg.global_min_n:
                continue
            delta_p10 = _solve_coverage_target(
                group,
                0.0,
                label_col=self._label_col,
                quantile_col="p10_raw",
                target=self._cfg.p10_floor_guard,
            )[0]
            delta_p90 = _solve_coverage_target(
                group,
                0.0,
                label_col=self._label_col,
                quantile_col="p90_raw",
                target=self._cfg.p90_floor_guard,
            )[0]
            stats[_ensure_tuple(key)] = (
                _clamp_abs_delta(delta_p10, self._cfg.global_delta_cap),
                _clamp_abs_delta(delta_p90, self._cfg.global_delta_cap),
            )
        return stats, counts


def compute_rolling_offsets(
    df: pd.DataFrame,
    *,
    global_delta_p10: float,
    global_delta_p90: float,
    label_col: str = "minutes",
    config: RollingCalibrationConfig | None = None,
    score_dates: Sequence[pd.Timestamp | str] | None = None,
) -> pd.DataFrame:
    """Compute per-day rolling offsets for lower/upper quantiles.

    The calibration window looks back ``window_days`` days using ``tip_ts`` while
    excluding the scoring date (tip < D). Each day's offsets are shrunk toward
    the global conformal deltas when the window has limited coverage.
    """

    cfg = config or RollingCalibrationConfig()

    required_cols: Iterable[str] = (
        list(KEY_COLUMNS)
        + [
            label_col,
            "game_date",
            "tip_ts",
            "feature_as_of_ts",
            "p10_raw",
            "p90_raw",
        ]
    )
    if cfg.use_buckets:
        required_cols = list(required_cols) + ["injury_snapshot_missing"]
        if cfg.use_spread_buckets:
            required_cols.append("spread_home")
    ensure_columns(df, required_cols)
    working = df.copy()
    working["feature_as_of_ts"] = _to_naive_ts(working["feature_as_of_ts"])
    working["tip_ts"] = _to_naive_ts(working["tip_ts"])
    working["game_date"] = _normalize_dates(working["game_date"])
    working = deduplicate_latest(working, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    working = _ensure_strata_columns(working, cfg.global_strata_cols)

    bucket_cols: list[str] = []
    if cfg.use_buckets:
        working, bucket_cols, _ = _attach_bucket_metadata(working, cfg)

    target_dates = _resolve_score_dates(score_dates, working["game_date"])

    results: list[dict[str, object]] = []
    smoothing_state: dict[str | None, tuple[float, float]] = {}
    window_length = pd.Timedelta(days=cfg.window_days)
    global_provider = _GlobalDeltaProvider(
        working,
        label_col,
        cfg,
        cfg.global_strata_cols,
        default_delta_p10=global_delta_p10,
        default_delta_p90=global_delta_p90,
    )
    tau_bucket = cfg.tau if cfg.tau_bucket is None else cfg.tau_bucket
    effective_min_n_bucket = cfg.min_n_bucket

    for score_date in target_dates:
        window_start = score_date - window_length
        window_mask = (working["tip_ts"] >= window_start) & (working["tip_ts"] < score_date)
        window = working.loc[window_mask].dropna(subset=[label_col, "p10_raw", "p90_raw"])
        n = int(len(window))
        shrinkage = _effective_shrinkage(n, cfg)
        cold_start_used = int(n < cfg.cold_start_min_n)

        anchor_overall = global_provider.anchor(score_date)
        delta_p10_anchor = anchor_overall.delta_p10 if cfg.use_global_p10_delta else 0.0
        delta_p90_anchor = anchor_overall.delta_p90 if cfg.use_global_p90_delta else 0.0

        window_p10_global = window["p10_raw"] + delta_p10_anchor
        window_p90_global = window["p90_raw"] + delta_p90_anchor
        delta10_local = _safe_quantile(window[label_col] - window_p10_global, 0.10)
        delta90_local = _safe_quantile(window[label_col] - window_p90_global, 0.90)
        delta10_local = 0.0 if not np.isfinite(delta10_local) else delta10_local
        delta90_local = 0.0 if not np.isfinite(delta90_local) else delta90_local

        age_days = _age_in_days(score_date, window["tip_ts"])
        recency_weights = _exp_decay_weights(age_days, cfg.recency_half_life_days)
        window_ess = _effective_sample_size_from_weights(recency_weights)
        recent_rows = _count_recent_rows(age_days, cfg.recency_window_days)

        trust_p10 = _warmup_trust(score_date, cfg.warmup_days_p10)
        trust_p90 = _warmup_trust(score_date, cfg.warmup_days_p90)

        delta_p10 = delta_p10_anchor + shrinkage * delta10_local
        guard_hit = False
        coverage_before = float("nan")
        if cold_start_used:
            delta_p10 = delta_p10_anchor
        else:
            delta_p10, guard_hit, coverage_before, coverage_after = _solve_coverage_target(
                window,
                delta_p10,
                label_col=label_col,
                quantile_col="p10_raw",
                target=cfg.p10_floor_guard,
                weights=recency_weights,
            )
            if guard_hit:
                logger.info(
                    "p10 floor solve date=%s n=%d coverage_before=%.4f coverage_after=%.4f delta=%.4f cold_start=%d",
                    score_date.date(),
                    n,
                    coverage_before if np.isfinite(coverage_before) else float("nan"),
                    coverage_after if np.isfinite(coverage_after) else float("nan"),
                    delta_p10,
                    cold_start_used,
                )
        cap_up_p10, cap_down_p10 = _resolve_cap_limit(cfg, recent_rows, quantile="p10", score_date=score_date)
        delta_p10, cap_bound_p10 = _clamp_signed_delta(delta_p10, cap_up_p10, cap_down_p10)
        delta_p10 = _blend_with_raw(delta_p10, trust_p10)
        delta_p10, cap_bound_post_p10 = _clamp_signed_delta(delta_p10, cap_up_p10, cap_down_p10)
        cap_bound_p10 = cap_bound_p10 or cap_bound_post_p10

        delta_p90 = delta_p90_anchor + shrinkage * delta90_local
        delta_p90, guard_hit_p90, coverage_p90_before, coverage_p90_after = _solve_coverage_target(
            window,
            delta_p90,
            label_col=label_col,
            quantile_col="p90_raw",
            target=cfg.p90_floor_guard,
            weights=recency_weights,
        )
        cap_up_p90, cap_down_p90 = _resolve_cap_limit(cfg, recent_rows, quantile="p90", score_date=score_date)
        delta_p90, cap_bound_p90 = _clamp_signed_delta(delta_p90, cap_up_p90, cap_down_p90)
        delta_p90 = _blend_with_raw(delta_p90, trust_p90)
        delta_p90, cap_bound_post_p90 = _clamp_signed_delta(delta_p90, cap_up_p90, cap_down_p90)
        cap_bound_p90 = cap_bound_p90 or cap_bound_post_p90

        window_residual_p10 = delta_p10 - delta_p10_anchor
        window_residual_p90 = delta_p90 - delta_p90_anchor

        enforce_one_sided = cfg.one_sided_days > 0 and score_date.day <= cfg.one_sided_days
        delta_p10 = _apply_one_sided_delta(
            delta_p10,
            direction="p10",
            enforce=enforce_one_sided,
            coverage_before=coverage_before,
            lower_threshold=cfg.hysteresis_lower,
            upper_threshold=cfg.hysteresis_upper,
        )
        delta_p90 = _apply_one_sided_delta(
            delta_p90,
            direction="p90",
            enforce=enforce_one_sided,
            coverage_before=coverage_p90_before,
            lower_threshold=1.0 - cfg.hysteresis_upper,
            upper_threshold=1.0 - cfg.hysteresis_lower,
        )

        prev_pair = smoothing_state.get(None)
        if cfg.delta_smoothing_alpha_p10 > 0.0:
            prev_p10 = None if prev_pair is None else prev_pair[0]
            delta_p10 = _smooth_delta(delta_p10, previous=prev_p10, alpha=cfg.delta_smoothing_alpha_p10)
        if cfg.delta_smoothing_alpha_p90 > 0.0:
            prev_p90 = None if prev_pair is None else prev_pair[1]
            delta_p90 = _smooth_delta(delta_p90, previous=prev_p90, alpha=cfg.delta_smoothing_alpha_p90)
        smoothing_state[None] = (delta_p10, delta_p90)

        base_payload: dict[str, object] = {
            "score_date": score_date,
            "window_start": window_start,
            "window_end": score_date,
            "n": n,
            "n_bucket": None,
            "shrinkage": shrinkage,
            "delta_p10": delta_p10,
            "delta_p90": delta_p90,
            "strategy": "rolling",
            "bucket_name": None,
            "bucket_source": "window",
            "cold_start_used": cold_start_used,
            "recent_rows": recent_rows,
            "window_ess": window_ess,
            "cap_bound_p10": int(bool(cap_bound_p10)),
            "cap_bound_p90": int(bool(cap_bound_p90)),
            "cap_value_p10_up": cap_up_p10,
            "cap_value_p10_down": cap_down_p10,
            "cap_value_p90_up": cap_up_p90,
            "cap_value_p90_down": cap_down_p90,
            "cap_value_p10": cap_up_p10,
            "cap_value_p90": cap_up_p90,
            "global_recent_weight_p10": anchor_overall.weight_recent_p10,
            "global_recent_weight_p90": anchor_overall.weight_recent_p90,
            "global_recent_rows": anchor_overall.recent_rows,
            "bucket_inherited": None,
            "bucket_ess": None,
        }

        if not cfg.use_buckets or not bucket_cols:
            results.append(base_payload)
            continue

        score_rows = working.loc[working["game_date"] == score_date]
        bucket_records = _enumerate_bucket_records(score_rows, bucket_cols)
        for bucket_values in bucket_records:
            bucket_window = window
            for col, value in bucket_values.items():
                bucket_window = bucket_window.loc[bucket_window[col] == value]
            n_bucket = int(len(bucket_window))

            recent_enough = True
            if cfg.min_recent_days > 0:
                month_start = score_date.replace(day=1)
                recent_rows_bucket = bucket_window.loc[bucket_window["game_date"] >= month_start]
                if recent_rows_bucket["game_date"].nunique() < cfg.min_recent_days:
                    recent_enough = False

            bucket_anchor = global_provider.anchor(score_date, bucket_values)
            delta_p10_used = (
                (bucket_anchor.delta_p10 if cfg.use_global_p10_delta else 0.0) + window_residual_p10
            )
            delta_p90_used = (
                (bucket_anchor.delta_p90 if cfg.use_global_p90_delta else 0.0) + window_residual_p90
            )

            bucket_weights = recency_weights.reindex(bucket_window.index) if not bucket_window.empty else pd.Series(dtype=float)
            bucket_ess = _effective_sample_size_from_weights(bucket_weights)
            inherit_bucket = cfg.bucket_inherit_if_low_n and (
                n_bucket < effective_min_n_bucket
                or bucket_ess < cfg.bucket_min_effective_sample_size
                or not recent_enough
            )

            bucket_source = "bucket"
            bucket_shrinkage = shrinkage
            bucket_guard_frame = None
            bucket_cov_p10_before = float("nan")
            bucket_cov_p90_before = float("nan")
            if not inherit_bucket and n_bucket > 0:
                bucket_shrinkage = _effective_shrinkage(
                    n_bucket,
                    cfg,
                    min_n_floor=effective_min_n_bucket,
                    tau_override=tau_bucket,
                )
                bucket_p10_global = bucket_window["p10_raw"] + bucket_anchor.delta_p10
                bucket_p90_global = bucket_window["p90_raw"] + bucket_anchor.delta_p90
                delta10_bucket = _safe_quantile(bucket_window[label_col] - bucket_p10_global, 0.10)
                delta90_bucket = _safe_quantile(bucket_window[label_col] - bucket_p90_global, 0.90)
                if np.isfinite(delta10_bucket):
                    delta_p10_used += bucket_shrinkage * delta10_bucket
                if np.isfinite(delta90_bucket):
                    delta_p90_used += bucket_shrinkage * delta90_bucket
                if cfg.bucket_delta_cap and cfg.bucket_delta_cap > 0:
                    delta_p10_used = _clamp_delta_deviation(delta_p10_used, delta_p10, cfg.bucket_delta_cap)
                    delta_p90_used = _clamp_delta_deviation(delta_p90_used, delta_p90, cfg.bucket_delta_cap)
                bucket_guard_frame = bucket_window
            else:
                bucket_source = "inherit"

            bucket_cold_start = int(bool(cold_start_used or inherit_bucket))

            if bucket_guard_frame is not None:
                delta_p10_used, _, bucket_cov_p10_before, _ = _solve_coverage_target(
                    bucket_guard_frame,
                    delta_p10_used,
                    label_col=label_col,
                    quantile_col="p10_raw",
                    target=cfg.p10_floor_guard,
                    weights=bucket_weights,
                )
                delta_p10_used = _blend_with_raw(
                    _clamp_signed_delta(delta_p10_used, cap_up_p10, cap_down_p10)[0],
                    trust_p10,
                )
                delta_p90_used, _, bucket_cov_p90_before, _ = _solve_coverage_target(
                    bucket_guard_frame,
                    delta_p90_used,
                    label_col=label_col,
                    quantile_col="p90_raw",
                    target=cfg.p90_floor_guard,
                    weights=bucket_weights,
                )
                delta_p90_used = _blend_with_raw(
                    _clamp_signed_delta(delta_p90_used, cap_up_p90, cap_down_p90)[0],
                    trust_p90,
                )

                if (
                    cfg.bucket_floor_relief > 0
                    and _simulate_coverage(
                        bucket_guard_frame,
                        delta_p10_used,
                        label_col,
                        "p10_raw",
                        weights=bucket_weights,
                    )
                    < cfg.bucket_floor_relief
                ):
                    delta_p10_used, _, _, _ = _solve_coverage_target(
                        bucket_guard_frame,
                        delta_p10_used,
                        label_col=label_col,
                        quantile_col="p10_raw",
                        target=cfg.p10_floor_guard,
                        weights=bucket_weights,
                    )

            bucket_key = _format_bucket_name(bucket_cols, bucket_values)
            coverage_p10_for_constraints = bucket_cov_p10_before
            if not np.isfinite(coverage_p10_for_constraints):
                coverage_p10_for_constraints = coverage_before
            coverage_p90_for_constraints = bucket_cov_p90_before
            if not np.isfinite(coverage_p90_for_constraints):
                coverage_p90_for_constraints = coverage_p90_before
            delta_p10_used = _apply_one_sided_delta(
                delta_p10_used,
                direction="p10",
                enforce=enforce_one_sided,
                coverage_before=coverage_p10_for_constraints,
                lower_threshold=cfg.hysteresis_lower,
                upper_threshold=cfg.hysteresis_upper,
            )
            delta_p90_used = _apply_one_sided_delta(
                delta_p90_used,
                direction="p90",
                enforce=enforce_one_sided,
                coverage_before=coverage_p90_for_constraints,
                lower_threshold=1.0 - cfg.hysteresis_upper,
                upper_threshold=1.0 - cfg.hysteresis_lower,
            )
            prev_pair = smoothing_state.get(bucket_key)
            if cfg.delta_smoothing_alpha_p10 > 0.0:
                prev_p10 = None if prev_pair is None else prev_pair[0]
                delta_p10_used = _smooth_delta(delta_p10_used, previous=prev_p10, alpha=cfg.delta_smoothing_alpha_p10)
            if cfg.delta_smoothing_alpha_p90 > 0.0:
                prev_p90 = None if prev_pair is None else prev_pair[1]
                delta_p90_used = _smooth_delta(delta_p90_used, previous=prev_p90, alpha=cfg.delta_smoothing_alpha_p90)
            smoothing_state[bucket_key] = (delta_p10_used, delta_p90_used)

            payload = base_payload.copy()
            payload.update(
                {
                    "n_bucket": n_bucket,
                    "bucket_name": bucket_key,
                    "bucket_source": bucket_source,
                    "delta_p10": delta_p10_used,
                    "delta_p90": delta_p90_used,
                    "shrinkage": bucket_shrinkage,
                    "cold_start_used": bucket_cold_start,
                    "bucket_inherited": int(inherit_bucket),
                    "bucket_ess": bucket_ess,
                    "global_recent_weight_p10": bucket_anchor.weight_recent_p10,
                    "global_recent_weight_p90": bucket_anchor.weight_recent_p90,
                    "global_recent_rows": bucket_anchor.recent_rows,
                }
            )
            payload.update(bucket_values)
            results.append(payload)

    offsets = pd.DataFrame(results)
    if cfg.cap_guardrail_rate is not None and not offsets.empty:
        window_rows = offsets.loc[offsets["bucket_name"].isna()]
        guard_mask = window_rows["score_date"].dt.day > cfg.cap_guardrail_day
        if guard_mask.any():
            clamp_rate = float(window_rows.loc[guard_mask, "cap_bound_p10"].mean())
            if clamp_rate > cfg.cap_guardrail_rate:
                raise RuntimeError(
                    f"P10 clamp binding rate {clamp_rate:.3f} exceeds guardrail {cfg.cap_guardrail_rate:.3f}"
                )

    return offsets


def apply_rolling_offsets(
    df: pd.DataFrame,
    offsets: pd.DataFrame,
    *,
    date_col: str = "game_date",
    config: RollingCalibrationConfig | None = None,
) -> pd.DataFrame:
    """Apply per-date offsets to a dataframe containing raw quantiles."""

    cfg = config or RollingCalibrationConfig()
    working = df.copy()
    working[date_col] = _normalize_dates(working[date_col])
    offsets = offsets.copy()
    offsets["score_date"] = _normalize_dates(offsets["score_date"])

    merge_left = [date_col]
    merge_right = ["score_date"]
    added_cols: list[str] = []
    bucket_cols: list[str] = []
    if cfg.use_buckets:
        working, bucket_cols, added_cols = _attach_bucket_metadata(working, cfg)
        for col in bucket_cols:
            if col not in offsets:
                raise ValueError(f"Rolling offsets missing bucket column '{col}' required for merge")
        merge_left.extend(bucket_cols)
        merge_right.extend(bucket_cols)

    offsets_subset = offsets.loc[:, merge_right + ["delta_p10", "delta_p90"]]
    merged = working.merge(
        offsets_subset,
        left_on=merge_left,
        right_on=merge_right,
        how="left",
    )
    missing_mask = merged["delta_p10"].isna() | merged["delta_p90"].isna()
    if missing_mask.any():
        missing = merged.loc[missing_mask, date_col].unique().tolist()
        raise ValueError(f"Missing rolling offsets for dates: {missing}")

    merged["p10"] = merged["p10_raw"] + merged["delta_p10"]
    merged["p90"] = merged["p90_raw"] + merged["delta_p90"]
    width_pre_projection = merged["p90"] - merged["p10"]
    projection_adjustments = 0
    if cfg.apply_center_width_projection:
        projection_adjustments = _project_minimal_tails(
            merged,
            median_col="p50_raw",
            min_half_width=max(cfg.min_quantile_half_width, 1e-6),
        )
        if projection_adjustments:
            logger.debug("Center-width projection adjusted %s quantile rows.", projection_adjustments)
    width_before = merged["p90_raw"] - merged["p10_raw"]
    width_after = merged["p90"] - merged["p10"]
    median_before = float(width_before.median(skipna=True))
    median_after = float(width_after.median(skipna=True))
    width_delta_pct = float("nan")
    if np.isfinite(median_before) and median_before != 0:
        width_delta_pct = (median_after - median_before) / median_before
        logger.debug("Quantile width median shift=%.3f%%", width_delta_pct * 100.0)
        if (
            cfg.max_width_delta_pct is not None
            and np.isfinite(width_delta_pct)
            and abs(width_delta_pct) > cfg.max_width_delta_pct
        ):
            raise RuntimeError(
                f"Median quantile width changed by {width_delta_pct:.3f}, exceeding guardrail {cfg.max_width_delta_pct:.3f}"
            )

    clamp_stats = _enforce_monotonicity(merged, median_col="p50_raw")
    if clamp_stats.total:
        logger.debug(
            "Clamped %s quantile rows (p10>p50=%s, p90<p50=%s).",
            clamp_stats.total,
            clamp_stats.lower,
            clamp_stats.upper,
        )
    if (
        cfg.monotonicity_repair_threshold is not None
        and merged.shape[0] > 0
        and clamp_stats.total / merged.shape[0] > cfg.monotonicity_repair_threshold
    ):
        raise RuntimeError(
            f"Monotonicity repairs {clamp_stats.total} exceed threshold "
            f"{cfg.monotonicity_repair_threshold:.3f}"
        )

    drop_cols = ["score_date", "delta_p10", "delta_p90"]
    merged = merged.drop(columns=drop_cols)
    for col in added_cols:
        if col in merged.columns:
            merged = merged.drop(columns=col)
    return merged


def _clamp_abs_delta(value: float, limit: float | None) -> float:
    if limit is None or limit <= 0 or not np.isfinite(value):
        return float(value)
    return float(np.clip(value, -limit, limit))


def _clamp_delta_deviation(value: float, reference: float, cap: float | None) -> float:
    if cap is None or cap <= 0 or not np.isfinite(value) or not np.isfinite(reference):
        return float(value)
    lower = reference - cap
    upper = reference + cap
    return float(np.clip(value, lower, upper))


def _warmup_trust(score_date: pd.Timestamp, warmup_days: int) -> float:
    if warmup_days <= 0:
        return 1.0
    month_start = score_date.replace(day=1)
    day_offset = max(0, (score_date - month_start).days)
    return float(min(1.0, day_offset / float(warmup_days)))


def _blend_with_raw(delta: float, trust: float) -> float:
    if not np.isfinite(delta):
        return float(delta)
    trust = float(np.clip(trust, 0.0, 1.0))
    return float(delta * trust)


def _ensure_strata_columns(df: pd.DataFrame, strata_cols: Sequence[str]) -> pd.DataFrame:
    if not strata_cols:
        return df
    for col in strata_cols:
        if col not in df.columns:
            default = 1 if col == "injury_snapshot_missing" else 0
            df[col] = default
        if col == "injury_snapshot_missing":
            df[col] = df[col].fillna(1).astype(int)
    return df


def _stratum_key(strata_cols: Sequence[str], values: dict[str, object] | None) -> tuple[object, ...]:
    if not strata_cols or not values:
        return ()
    return tuple(values.get(col) for col in strata_cols)


def _ensure_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _blend_values(baseline: float, recent: float, weight: float) -> float:
    if not np.isfinite(baseline):
        baseline = 0.0
    if not np.isfinite(recent):
        recent = baseline
    weight = float(np.clip(weight, 0.0, 1.0))
    return float((1.0 - weight) * baseline + weight * recent)


def _age_in_days(score_date: pd.Timestamp, tips: pd.Series) -> pd.Series:
    if tips.empty:
        return pd.Series(dtype=float)
    deltas = pd.Series(pd.Timestamp(score_date) - tips, index=tips.index)
    return deltas.dt.total_seconds() / 86_400.0


def _exp_decay_weights(age_days: pd.Series, half_life_days: float) -> pd.Series:
    if age_days.empty:
        return pd.Series(dtype=float)
    if half_life_days <= 0:
        return pd.Series(1.0, index=age_days.index, dtype=float)
    decay = 0.5 ** (age_days / max(half_life_days, 1e-6))
    decay = np.clip(decay, 0.0, None)
    return pd.Series(decay, index=age_days.index, dtype=float)


def _effective_sample_size_from_weights(weights: pd.Series | None) -> float:
    if weights is None or weights.empty:
        return 0.0
    arr = weights.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    total = float(arr.sum())
    sum_sq = float(np.square(arr).sum())
    if sum_sq <= 0:
        return 0.0
    return float((total**2) / sum_sq)


def _count_recent_rows(age_days: pd.Series, window_days: int) -> int:
    if age_days.empty or window_days <= 0:
        return int(len(age_days))
    return int((age_days <= window_days).sum())


def _resolve_cap_limit(
    cfg: RollingCalibrationConfig,
    recent_rows: int,
    *,
    quantile: str,
    score_date: pd.Timestamp,
) -> float | None:
    if quantile == "p10":
        fixed = cfg.max_abs_delta_p10
        cap_lo_up = cfg.max_abs_delta_p10_lo
        cap_hi_up = cfg.max_abs_delta_p10_hi
        cap_lo_down = cfg.max_abs_delta_p10_lo_down
        cap_hi_down = cfg.max_abs_delta_p10_hi_down
        if (
            cfg.max_abs_delta_p10_hi_early is not None
            and cfg.early_cap_days > 0
            and score_date.day <= cfg.early_cap_days
        ):
            cap_hi_up = max(cap_lo_up, min(cap_hi_up, cfg.max_abs_delta_p10_hi_early))
    else:
        fixed = cfg.max_abs_delta_p90
        cap_lo_up = cfg.max_abs_delta_p90_lo
        cap_hi_up = cfg.max_abs_delta_p90_hi
        cap_lo_down = cfg.max_abs_delta_p90_lo_down
        cap_hi_down = cfg.max_abs_delta_p90_hi_down
        if (
            cfg.max_abs_delta_p90_hi_early is not None
            and cfg.early_cap_days > 0
            and score_date.day <= cfg.early_cap_days
        ):
            cap_hi_up = min(cap_hi_up, cfg.max_abs_delta_p90_hi_early)
    if fixed is not None:
        if fixed <= 0:
            return float("inf"), float("inf")
        value = float(fixed)
        return value, value
    cap_hi_up = max(cap_hi_up, cap_lo_up)
    cap_hi_down = max(cap_hi_down, cap_lo_down)
    target_rows = max(1, cfg.recent_target_rows)
    trust = min(1.0, max(0.0, recent_rows / target_rows))
    cap_up = float(cap_lo_up + (cap_hi_up - cap_lo_up) * (1.0 - trust))
    cap_down = float(cap_lo_down + (cap_hi_down - cap_lo_down) * (1.0 - trust))
    return cap_up, cap_down


def _clamp_signed_delta(value: float, cap_up: float, cap_down: float) -> tuple[float, bool]:
    if not np.isfinite(value):
        return float(value), False
    upper = max(0.0, cap_up)
    lower = max(0.0, cap_down)
    bounded = value
    if value > 0:
        bounded = min(value, upper)
    else:
        bounded = max(value, -lower)
    return float(bounded), not np.isclose(bounded, value, atol=1e-9)


def _effective_shrinkage(
    n: int,
    cfg: RollingCalibrationConfig,
    *,
    min_n_floor: int | None = None,
    tau_override: float | None = None,
) -> float:
    if n <= 0:
        return 0.0
    threshold = cfg.min_n if min_n_floor is None else min_n_floor
    if n < threshold:
        return 0.0
    tau_value = cfg.tau if tau_override is None else tau_override
    if tau_value <= 0:
        return 1.0
    return n / (n + tau_value)


def _safe_quantile(series: pd.Series, q: float) -> float:
    arr = series.to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _to_naive_ts(values: Iterable[object]) -> pd.Series:
    series = pd.Series(pd.to_datetime(values, utc=True))
    return series.dt.tz_localize(None)


def _normalize_dates(values: Iterable[object]) -> pd.Series:
    series = pd.Series(pd.to_datetime(values))
    tz = getattr(series.dt, "tz", None)
    if tz is not None:
        series = pd.Series(series.dt.tz_convert("UTC"))
    return series.dt.tz_localize(None).dt.normalize()


def _resolve_score_dates(
    score_dates: Sequence[pd.Timestamp | str] | None,
    default_series: pd.Series,
) -> list[pd.Timestamp]:
    if score_dates is None:
        normalized = _normalize_dates(default_series)
    else:
        normalized = _normalize_dates(score_dates)
    unique = normalized.dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(ts) for ts in unique.tolist()]


def _attach_bucket_metadata(
    df: pd.DataFrame,
    cfg: RollingCalibrationConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    working = df.copy()
    bucket_cols: list[str] = []
    added_cols: list[str] = []
    if not cfg.use_buckets:
        return working, bucket_cols, added_cols

    ensure_columns(working, {"injury_snapshot_missing"})
    working["injury_snapshot_missing"] = working["injury_snapshot_missing"].fillna(1).astype(int)
    bucket_cols.append("injury_snapshot_missing")

    if cfg.use_spread_buckets:
        ensure_columns(working, {"spread_home"})
        spread_bucket = _categorize_spread_bucket(working["spread_home"])
        working["_spread_bucket"] = spread_bucket
        working = working.rename(columns={"_spread_bucket": "spread_bucket"})
        bucket_cols.append("spread_bucket")
        added_cols.append("spread_bucket")

    return working, bucket_cols, added_cols


def _categorize_spread_bucket(series: pd.Series) -> pd.Series:
    abs_spread = series.astype(float).abs()
    abs_spread = abs_spread.fillna(0.0)
    labels = []
    for value in abs_spread.to_numpy():
        if value >= 10.0:
            labels.append("abs_spread_ge_10")
        elif value >= 5.0:
            labels.append("abs_spread_5_9.5")
        else:
            labels.append("abs_spread_0_4.5")
    return pd.Series(labels, index=series.index, dtype="string")


def _enumerate_bucket_records(df: pd.DataFrame, bucket_cols: Sequence[str]) -> list[dict[str, object]]:
    if not bucket_cols:
        return [{}]
    unique = df.loc[:, bucket_cols].drop_duplicates().sort_values(bucket_cols)
    records: list[dict[str, object]] = []
    for _, row in unique.iterrows():
        record = {col: row[col] for col in bucket_cols}
        records.append(record)
    return records


def _format_bucket_name(bucket_cols: Sequence[str], values: dict[str, object]) -> str | None:
    if not bucket_cols:
        return None
    parts = [f"{col}={values[col]}" for col in bucket_cols]
    return "|".join(parts)


def _enforce_monotonicity(df: pd.DataFrame, *, median_col: str) -> MonotonicityStats:
    if median_col not in df:
        return MonotonicityStats(total=0, lower=0, upper=0)
    lower = df["p10"].to_numpy(dtype=float)
    median = df[median_col].to_numpy(dtype=float)
    upper = df["p90"].to_numpy(dtype=float)
    low_violation = lower > median
    high_violation = upper < median
    lower_count = int(low_violation.sum())
    upper_count = int(high_violation.sum())
    adjustments = lower_count + upper_count
    if adjustments:
        if low_violation.any():
            df.loc[low_violation, "p10"] = df.loc[low_violation, median_col]
        if high_violation.any():
            df.loc[high_violation, "p90"] = df.loc[high_violation, median_col]
    return MonotonicityStats(total=adjustments, lower=lower_count, upper=upper_count)


def _project_minimal_tails(
    df: pd.DataFrame,
    *,
    median_col: str,
    min_half_width: float,
) -> int:
    if median_col not in df:
        return 0
    eps = max(min_half_width, 1e-6)
    centers = df[median_col].to_numpy(dtype=float)
    lowers = df["p10"].to_numpy(dtype=float)
    uppers = df["p90"].to_numpy(dtype=float)
    adjustments = 0
    for idx in range(len(df)):
        center = centers[idx]
        low = lowers[idx]
        high = uppers[idx]
        changed = False
        desired_low = center - eps
        desired_high = center + eps
        if low > desired_low:
            low = desired_low
            changed = True
        if high < desired_high:
            high = desired_high
            changed = True
        if low > high:
            mid = center
            half = max(eps, abs(high - low) / 2.0)
            low = mid - half
            high = mid + half
            changed = True
        lowers[idx] = low
        uppers[idx] = high
        if changed:
            adjustments += 1
    df["p10"] = lowers
    df["p90"] = uppers
    return adjustments


def _apply_one_sided_delta(
    delta: float,
    *,
    direction: str,
    enforce: bool,
    coverage_before: float,
    lower_threshold: float,
    upper_threshold: float,
) -> float:
    if not enforce or not np.isfinite(coverage_before):
        return float(delta)
    if direction == "p10":
        if coverage_before < lower_threshold and delta < 0:
            return 0.0
        if coverage_before > upper_threshold and delta > 0:
            return 0.0
    else:
        if coverage_before < lower_threshold and delta < 0:
            return 0.0
        if coverage_before > upper_threshold and delta > 0:
            return 0.0
    return float(delta)


def _smooth_delta(
    current: float,
    *,
    previous: float | None,
    alpha: float,
) -> float:
    if not np.isfinite(current):
        return float(current)
    if previous is None or not np.isfinite(previous):
        return float(current)
    if alpha <= 0.0 or alpha >= 1.0:
        return float(current)
    return float(alpha * previous + (1.0 - alpha) * current)


def _simulate_coverage(
    frame: pd.DataFrame,
    delta: float,
    label_col: str,
    quantile_col: str,
    weights: pd.Series | None = None,
) -> float:
    if frame.empty or not np.isfinite(delta):
        return float("nan")
    threshold = frame[quantile_col] + delta
    hits = frame[label_col] <= threshold
    if hits.empty:
        return float("nan")
    if weights is None or weights.empty:
        return float(hits.mean())
    aligned = weights.reindex(hits.index)
    valid = aligned.notna()
    if not valid.any():
        return float("nan")
    weight_values = aligned.loc[valid].to_numpy(dtype=float)
    weight_values = weight_values[np.isfinite(weight_values)]
    hits_values = hits.loc[valid].to_numpy(dtype=float)
    if weight_values.size == 0:
        return float("nan")
    total = float(weight_values.sum())
    if total <= 0:
        return float("nan")
    return float(np.dot(weight_values, hits_values) / total)


def _solve_coverage_target(
    frame: pd.DataFrame,
    delta: float,
    *,
    label_col: str,
    quantile_col: str,
    target: float,
    weights: pd.Series | None = None,
) -> tuple[float, bool, float, float]:
    coverage_before = _simulate_coverage(frame, delta, label_col, quantile_col, weights=weights)
    if target <= 0 or frame.empty or not np.isfinite(delta):
        return delta, False, coverage_before, coverage_before
    if not np.isfinite(coverage_before) or coverage_before >= target:
        return delta, False, coverage_before, coverage_before
    baseline_delta = 0.0
    coverage_baseline = _simulate_coverage(frame, baseline_delta, label_col, quantile_col, weights=weights)
    if not np.isfinite(coverage_baseline) or coverage_baseline < target:
        return delta, False, coverage_before, coverage_before
    low, high = sorted((delta, baseline_delta))
    best = high
    for _ in range(25):
        mid = (low + high) / 2.0
        mid_coverage = _simulate_coverage(frame, mid, label_col, quantile_col, weights=weights)
        if not np.isfinite(mid_coverage):
            break
        if mid_coverage >= target:
            best = mid
            high = mid
        else:
            low = mid
    coverage_after = _simulate_coverage(frame, best, label_col, quantile_col, weights=weights)
    return best, True, coverage_before, coverage_after
