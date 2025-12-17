"""Evaluation utilities for ownership_v1 predictions.

This module is intentionally dependency-light (numpy/pandas/scipy) and is designed
to be used both from scripts and tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from projections.paths import data_path


NormalizationMethod = Literal["none", "scale_to_sum"]


@dataclass(frozen=True)
class OwnershipEvalSlice:
    """A fixed validation slice definition (do not move goalposts)."""

    name: str
    slate_ids: list[str]
    data_source: str | None = None
    target_sum_pct: float = 800.0

    def filter_df(self, df: pd.DataFrame, slate_id_col: str = "slate_id") -> pd.DataFrame:
        if slate_id_col not in df.columns:
            raise KeyError(f"Missing required column: {slate_id_col}")
        filtered = df[df[slate_id_col].astype(str).isin(set(self.slate_ids))].copy()
        if self.data_source is not None and "data_source" in filtered.columns:
            filtered = filtered[filtered["data_source"].astype(str) == self.data_source].copy()
        return filtered

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OwnershipEvalSlice":
        return cls(
            name=str(payload["name"]),
            slate_ids=[str(x) for x in payload["slate_ids"]],
            data_source=str(payload["data_source"]) if payload.get("data_source") is not None else None,
            target_sum_pct=float(payload.get("target_sum_pct", 800.0)),
        )


def load_eval_slice(path: Path | str) -> OwnershipEvalSlice:
    """Load a slice definition from JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval slice config not found: {p}")
    import json

    payload = json.loads(p.read_text(encoding="utf-8"))
    return OwnershipEvalSlice.from_dict(payload)


def default_eval_slice_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "ownership_eval_slice.json"


def _clip_pct_to_prob(pct: np.ndarray, *, eps: float) -> np.ndarray:
    return np.clip(pct / 100.0, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class RegressionMetrics:
    mae_pct: float
    rmse_pct: float
    mae_logit: float
    rmse_logit: float


@dataclass(frozen=True)
class RankMetrics:
    spearman_pooled: float
    spearman_per_slate_mean: float
    spearman_per_slate_std: float
    spearman_top10_mean: float
    spearman_top20_mean: float
    recall_at_10: float
    recall_at_20: float


@dataclass(frozen=True)
class CalibrationMetrics:
    ece_pct: float
    bins: list[dict[str, Any]]


@dataclass(frozen=True)
class SumMetrics:
    target_sum_pct: float
    sum_actual_mean: float
    sum_actual_std: float
    sum_actual_min: float
    sum_actual_max: float
    mean_abs_actual_sum_error_to_target: float
    sum_pred_mean: float
    sum_pred_std: float
    sum_pred_min: float
    sum_pred_max: float
    mean_abs_sum_error_to_target: float
    max_pred_mean: float
    max_pred_p95: float
    n_pred_over_60: int
    n_pred_over_70: int
    n_pred_over_100: int


@dataclass(frozen=True)
class SegmentBiasMetrics:
    top10_mean_bias_pct: float
    top20_mean_bias_pct: float
    tail_le_5_mean_bias_pct: float
    tail_le_1_mean_bias_pct: float


@dataclass(frozen=True)
class OwnershipEvaluationResult:
    slice_name: str
    n_rows: int
    n_slates: int
    date_min: str | None
    date_max: str | None
    normalization: NormalizationMethod
    regression: RegressionMetrics
    ranking: RankMetrics
    calibration: CalibrationMetrics
    sums: SumMetrics
    segment_bias: SegmentBiasMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else float("nan")


def _compute_recall_at_k(group: pd.DataFrame, *, k: int, actual_col: str, pred_col: str, id_col: str | None) -> float:
    if len(group) < k:
        return float("nan")
    if id_col is not None and id_col in group.columns:
        ids = group[id_col].astype(str)
    else:
        ids = group.index.astype(str)
    actual_top = set(ids.loc[group[actual_col].nlargest(k).index].tolist())
    pred_top = set(ids.loc[group[pred_col].nlargest(k).index].tolist())
    return len(actual_top & pred_top) / float(k)


def _default_calibration_bin_edges() -> np.ndarray:
    # Higher resolution near common DFS ranges + chalk region.
    return np.array([0, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 100], dtype=float)


def compute_calibration_table(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    bin_edges: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    if bin_edges is None:
        edges = _default_calibration_bin_edges()
    else:
        edges = np.array(list(bin_edges), dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges must be a 1D sequence with >= 2 edges")

    work = df[[actual_col, pred_col]].copy()
    work = work.dropna(subset=[actual_col, pred_col])
    if work.empty:
        empty = pd.DataFrame(columns=["bin", "n", "avg_pred", "avg_actual", "bias"])
        return float("nan"), empty

    pred = work[pred_col].astype(float).to_numpy()
    actual = work[actual_col].astype(float).to_numpy()

    # pd.cut includes right edge by default; use right=False for half-open [a,b).
    bins = pd.cut(pred, bins=edges, right=False, include_lowest=True)
    work["bin"] = bins.astype(str)

    grouped = (
        work.groupby("bin", observed=True)
        .agg(n=(pred_col, "size"), avg_pred=(pred_col, "mean"), avg_actual=(actual_col, "mean"))
        .reset_index()
    )
    grouped["bias"] = grouped["avg_pred"] - grouped["avg_actual"]

    total = grouped["n"].sum()
    if total <= 0:
        return float("nan"), grouped
    ece = float((grouped["n"] * grouped["bias"].abs()).sum() / total)
    return ece, grouped


def _normalize_by_group_sum(
    df: pd.DataFrame,
    *,
    pred_col: str,
    group_col: str,
    target_sum_pct: float,
    method: NormalizationMethod,
) -> pd.Series:
    if method == "none":
        return df[pred_col].astype(float)

    if method != "scale_to_sum":
        raise ValueError(f"Unknown normalization method: {method}")

    # Mirror production behavior: predictions are clipped to valid percent space
    # before any downstream normalization.
    base = df[pred_col].astype(float).clip(lower=0.0, upper=100.0)
    sums = base.groupby(df[group_col]).transform("sum").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = target_sum_pct / sums
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return base * scale


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    slice_name: str,
    actual_col: str = "actual_own_pct",
    pred_col: str = "pred_own_pct",
    slate_id_col: str = "slate_id",
    player_id_col: str = "player_id",
    target_sum_pct: float = 800.0,
    normalization: NormalizationMethod = "none",
    logit_eps: float = 0.005,
) -> OwnershipEvaluationResult:
    required = [actual_col, pred_col, slate_id_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df.copy()
    work[slate_id_col] = work[slate_id_col].astype(str)
    work[pred_col] = work[pred_col].astype(float)
    work[actual_col] = work[actual_col].astype(float)

    # Apply normalization (if any) in percent space.
    work["_pred_eval"] = _normalize_by_group_sum(
        work,
        pred_col=pred_col,
        group_col=slate_id_col,
        target_sum_pct=target_sum_pct,
        method=normalization,
    )
    # Keep outputs in percent space. For "none", the upstream model is expected to
    # already be in [0, 100]. For sum-normalization methods, we avoid hard upper
    # clipping here so the sum constraint can be evaluated honestly; spike checks
    # are reported separately.
    if normalization == "none":
        work["_pred_eval"] = work["_pred_eval"].clip(0.0, 100.0)
    else:
        work["_pred_eval"] = work["_pred_eval"].clip(lower=0.0)

    # Basic metadata
    n_rows = int(len(work))
    n_slates = int(work[slate_id_col].nunique())
    date_min = str(work["game_date"].min()) if "game_date" in work.columns and n_rows > 0 else None
    date_max = str(work["game_date"].max()) if "game_date" in work.columns and n_rows > 0 else None

    # Regression metrics (percent)
    err = work["_pred_eval"].to_numpy() - work[actual_col].to_numpy()
    mae_pct = float(np.mean(np.abs(err))) if n_rows else float("nan")
    rmse_pct = float(np.sqrt(np.mean(err**2))) if n_rows else float("nan")

    # Logit-space metrics
    p_pred = _clip_pct_to_prob(work["_pred_eval"].to_numpy(), eps=logit_eps)
    p_act = _clip_pct_to_prob(work[actual_col].to_numpy(), eps=logit_eps)
    logit_pred = _logit(p_pred)
    logit_act = _logit(p_act)
    logit_err = logit_pred - logit_act
    mae_logit = float(np.mean(np.abs(logit_err))) if n_rows else float("nan")
    rmse_logit = float(np.sqrt(np.mean(logit_err**2))) if n_rows else float("nan")

    regression = RegressionMetrics(
        mae_pct=mae_pct,
        rmse_pct=rmse_pct,
        mae_logit=mae_logit,
        rmse_logit=rmse_logit,
    )

    # Ranking metrics
    spearman_pooled = _safe_spearman(work[actual_col].to_numpy(), work["_pred_eval"].to_numpy())

    per_slate = []
    top10_spearman = []
    top20_spearman = []
    recall10 = []
    recall20 = []

    for _, g in work.groupby(slate_id_col):
        rho = _safe_spearman(g[actual_col].to_numpy(), g["_pred_eval"].to_numpy())
        if not np.isnan(rho):
            per_slate.append(rho)

        # Top-k subset metrics on actual top-k players
        for k, acc in [(10, top10_spearman), (20, top20_spearman)]:
            if len(g) < k:
                continue
            top = g.nlargest(k, actual_col)
            rho_k = _safe_spearman(top[actual_col].to_numpy(), top["_pred_eval"].to_numpy())
            if not np.isnan(rho_k):
                acc.append(rho_k)

        # Recall@k (top actual vs top predicted)
        r10 = _compute_recall_at_k(g, k=10, actual_col=actual_col, pred_col="_pred_eval", id_col=player_id_col)
        r20 = _compute_recall_at_k(g, k=20, actual_col=actual_col, pred_col="_pred_eval", id_col=player_id_col)
        if not np.isnan(r10):
            recall10.append(r10)
        if not np.isnan(r20):
            recall20.append(r20)

    spearman_mean = float(np.mean(per_slate)) if per_slate else float("nan")
    spearman_std = float(np.std(per_slate)) if per_slate else float("nan")
    spearman_top10_mean = float(np.mean(top10_spearman)) if top10_spearman else float("nan")
    spearman_top20_mean = float(np.mean(top20_spearman)) if top20_spearman else float("nan")
    recall_at_10 = float(np.mean(recall10)) if recall10 else float("nan")
    recall_at_20 = float(np.mean(recall20)) if recall20 else float("nan")

    ranking = RankMetrics(
        spearman_pooled=float(spearman_pooled),
        spearman_per_slate_mean=spearman_mean,
        spearman_per_slate_std=spearman_std,
        spearman_top10_mean=spearman_top10_mean,
        spearman_top20_mean=spearman_top20_mean,
        recall_at_10=recall_at_10,
        recall_at_20=recall_at_20,
    )

    # Calibration metrics (ECE in pct points)
    ece, cal_table = compute_calibration_table(work, actual_col=actual_col, pred_col="_pred_eval")
    calibration = CalibrationMetrics(
        ece_pct=float(ece),
        bins=cal_table.to_dict(orient="records"),
    )

    # Segment bias metrics (mean over slates for top-k and tail)
    top10_bias = []
    top20_bias = []
    tail5_bias = []
    tail1_bias = []
    for _, g in work.groupby(slate_id_col):
        if len(g) >= 10:
            top10 = g.nlargest(10, actual_col)
            top10_bias.append(float((top10["_pred_eval"] - top10[actual_col]).mean()))
        if len(g) >= 20:
            top20 = g.nlargest(20, actual_col)
            top20_bias.append(float((top20["_pred_eval"] - top20[actual_col]).mean()))
        tail5 = g[g[actual_col] <= 5.0]
        if not tail5.empty:
            tail5_bias.append(float((tail5["_pred_eval"] - tail5[actual_col]).mean()))
        tail1 = g[g[actual_col] <= 1.0]
        if not tail1.empty:
            tail1_bias.append(float((tail1["_pred_eval"] - tail1[actual_col]).mean()))

    segment_bias = SegmentBiasMetrics(
        top10_mean_bias_pct=float(np.mean(top10_bias)) if top10_bias else float("nan"),
        top20_mean_bias_pct=float(np.mean(top20_bias)) if top20_bias else float("nan"),
        tail_le_5_mean_bias_pct=float(np.mean(tail5_bias)) if tail5_bias else float("nan"),
        tail_le_1_mean_bias_pct=float(np.mean(tail1_bias)) if tail1_bias else float("nan"),
    )

    # Sum constraint metrics
    sums_actual_by_slate = work.groupby(slate_id_col)[actual_col].sum().astype(float)
    abs_actual_err_to_target = (sums_actual_by_slate - float(target_sum_pct)).abs()

    sums_by_slate = work.groupby(slate_id_col)["_pred_eval"].sum().astype(float)
    abs_err_to_target = (sums_by_slate - float(target_sum_pct)).abs()
    max_by_slate = work.groupby(slate_id_col)["_pred_eval"].max().astype(float)
    sums = SumMetrics(
        target_sum_pct=float(target_sum_pct),
        sum_actual_mean=float(sums_actual_by_slate.mean()) if not sums_actual_by_slate.empty else float("nan"),
        sum_actual_std=float(sums_actual_by_slate.std()) if not sums_actual_by_slate.empty else float("nan"),
        sum_actual_min=float(sums_actual_by_slate.min()) if not sums_actual_by_slate.empty else float("nan"),
        sum_actual_max=float(sums_actual_by_slate.max()) if not sums_actual_by_slate.empty else float("nan"),
        mean_abs_actual_sum_error_to_target=float(abs_actual_err_to_target.mean())
        if not abs_actual_err_to_target.empty
        else float("nan"),
        sum_pred_mean=float(sums_by_slate.mean()) if not sums_by_slate.empty else float("nan"),
        sum_pred_std=float(sums_by_slate.std()) if not sums_by_slate.empty else float("nan"),
        sum_pred_min=float(sums_by_slate.min()) if not sums_by_slate.empty else float("nan"),
        sum_pred_max=float(sums_by_slate.max()) if not sums_by_slate.empty else float("nan"),
        mean_abs_sum_error_to_target=float(abs_err_to_target.mean()) if not abs_err_to_target.empty else float("nan"),
        max_pred_mean=float(max_by_slate.mean()) if not max_by_slate.empty else float("nan"),
        max_pred_p95=float(np.percentile(max_by_slate.to_numpy(), 95)) if len(max_by_slate) else float("nan"),
        n_pred_over_60=int((work["_pred_eval"] > 60.0).sum()),
        n_pred_over_70=int((work["_pred_eval"] > 70.0).sum()),
        n_pred_over_100=int((work["_pred_eval"] > 100.0).sum()),
    )

    return OwnershipEvaluationResult(
        slice_name=slice_name,
        n_rows=n_rows,
        n_slates=n_slates,
        date_min=date_min,
        date_max=date_max,
        normalization=normalization,
        regression=regression,
        ranking=ranking,
        calibration=calibration,
        sums=sums,
        segment_bias=segment_bias,
    )


def load_run_val_predictions(run_id: str, *, root: Path | None = None) -> pd.DataFrame:
    """Load `val_predictions.csv` for an ownership_v1 run_id."""
    base = root if root is not None else data_path()
    path = base / "artifacts" / "ownership_v1" / "runs" / run_id / "val_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"val_predictions.csv not found: {path}")
    return pd.read_csv(path)


__all__ = [
    "CalibrationMetrics",
    "OwnershipEvalSlice",
    "OwnershipEvaluationResult",
    "RankMetrics",
    "RegressionMetrics",
    "SegmentBiasMetrics",
    "SumMetrics",
    "NormalizationMethod",
    "compute_calibration_table",
    "default_eval_slice_path",
    "evaluate_predictions",
    "load_eval_slice",
    "load_run_val_predictions",
]
