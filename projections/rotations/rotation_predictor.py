from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


_DIGITS_RE = re.compile(r"[^0-9]+")


def canonicalize_game_id(game_id: object) -> str:
    """Return a canonical 10-digit NBA game_id string.

    Accepts values like:
    - "0022400001" (already canonical)
    - 22400001 or "22400001" (left-pad -> "0022400001")
    - 2_240_000_1-like strings with punctuation (digits extracted)
    """
    if game_id is None:
        return ""
    s = str(game_id).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    # Handle common float artifacts like "22400001.0".
    if s.endswith(".0"):
        s = s[:-2]
    s = _DIGITS_RE.sub("", s)
    if not s:
        return ""
    # Canonical NBA ids are 10 digits; some sources drop leading zeros -> 8 digits.
    if len(s) > 10:
        # Best-effort: interpret as int to strip any leading zeros beyond 10, then re-pad.
        try:
            s = str(int(s))
        except Exception:
            return ""
    return s.zfill(10)


def season_start_year_from_game_id(game_id: str) -> Optional[int]:
    gid = canonicalize_game_id(game_id)
    if len(gid) != 10 or not gid.isdigit():
        return None
    season_two = gid[3:5]
    if not season_two.isdigit():
        return None
    return 2000 + int(season_two)


def _resolve_bundle_dir(path: Path) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    if p.is_file():
        run_id = p.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError(f"Empty rotation predictor bundle pointer: {p}")
        resolved = p.parent / run_id
        if not resolved.exists():
            raise FileNotFoundError(f"Pointer {p} -> {resolved} does not exist")
        return resolved
    raise FileNotFoundError(f"rotation predictor bundle not found: {p}")


@dataclass(frozen=True)
class RotationPredictorBundle:
    bundle_dir: Path
    meta: dict[str, Any]
    feature_columns: tuple[str, ...]
    dataset_dir: Optional[Path]

    @property
    def model_ge5_path(self) -> Path:
        return self.bundle_dir / "model_ge5.lgb"

    @property
    def model_ge15_path(self) -> Path:
        return self.bundle_dir / "model_ge15.lgb"

    @property
    def predictions_test_path(self) -> Path:
        return self.bundle_dir / "predictions_test.parquet"

    @property
    def predictions_all_path(self) -> Path:
        return self.bundle_dir / "predictions_all.parquet"


def load_rotation_predictor_bundle(path: Path) -> RotationPredictorBundle:
    bundle_dir = _resolve_bundle_dir(Path(path))
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in rotation predictor bundle: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_columns = tuple(str(c) for c in meta.get("feature_columns", []) if str(c))
    dataset_dir = meta.get("dataset_dir", None)
    dataset_path = Path(dataset_dir) if dataset_dir else None
    return RotationPredictorBundle(
        bundle_dir=bundle_dir,
        meta=meta,
        feature_columns=feature_columns,
        dataset_dir=dataset_path,
    )


def _map_person_id_to_internal(
    person_ids: Iterable[object],
    *,
    person_id_to_internal_id: Optional[dict[int, int]],
) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    mapping = person_id_to_internal_id or {}
    for v in person_ids:
        try:
            pid = int(v)
        except Exception:
            out.append(None)
            continue
        # Heuristic: rot_v1 internal IDs are small contiguous ints. When a value is already in that
        # range, treat it as internal.
        if pid <= 2000:
            out.append(pid)
            continue
        out.append(int(mapping.get(pid)) if pid in mapping else None)
    return out


def load_cached_predictions(
    bundle: RotationPredictorBundle,
    *,
    person_id_to_internal_id: Optional[dict[int, int]] = None,
    game_id_allow: Optional[set[str]] = None,
    team_id_allow: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Load cached predictions_test.parquet and normalize keys to rot_v1 conventions.

    Returns columns: game_id, team_id, player_id, p_ge5_pred, p_ge15_pred, pred_source.
    - game_id is canonicalized to 10-digit string.
    - player_id is converted into internal id space (when mapping provided), else left as-is for ids<=2000.
    """
    p = bundle.predictions_test_path
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions_test.parquet: {p}")

    df = pd.read_parquet(p)
    required = {"game_id", "team_id", "player_id", "p_ge5", "p_ge15"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"predictions_test.parquet missing required columns: {missing}. Got columns={list(df.columns)}")

    out = df[["game_id", "team_id", "player_id", "p_ge5", "p_ge15"]].copy()
    out["game_id"] = out["game_id"].map(canonicalize_game_id).astype("string")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["p_ge5_pred"] = pd.to_numeric(out["p_ge5"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
    out["p_ge15_pred"] = pd.to_numeric(out["p_ge15"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
    out = out.drop(columns=["p_ge5", "p_ge15"])

    out = out.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    out["team_id"] = out["team_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)

    out["player_id"] = pd.Series(
        _map_person_id_to_internal(out["player_id"].tolist(), person_id_to_internal_id=person_id_to_internal_id),
        index=out.index,
        dtype="Int64",
    )
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    if game_id_allow is not None:
        out = out[out["game_id"].isin({str(g) for g in game_id_allow})].copy()
    if team_id_allow is not None:
        out = out[out["team_id"].isin({int(t) for t in team_id_allow})].copy()

    out["pred_source"] = "cached_preds"
    out = out.sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    out = out.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()
    return out[["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred", "pred_source"]].copy()


def load_cached_all_predictions(
    bundle: RotationPredictorBundle,
    *,
    person_id_to_internal_id: Optional[dict[int, int]] = None,
    game_id_allow: Optional[set[str]] = None,
    team_id_allow: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Load predictions_all.parquet and normalize keys to rot_v1 conventions.

    Returns columns: game_id, team_id, player_id, p_ge5_pred, p_ge15_pred, pred_source.
    """
    p = bundle.predictions_all_path
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions_all.parquet: {p}")

    df = pd.read_parquet(p)
    required = {"game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"predictions_all.parquet missing required columns: {missing}. Got columns={list(df.columns)}")

    out = df[["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"]].copy()
    out["game_id"] = out["game_id"].map(canonicalize_game_id).astype("string")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["p_ge5_pred"] = pd.to_numeric(out["p_ge5_pred"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
    out["p_ge15_pred"] = pd.to_numeric(out["p_ge15_pred"], errors="coerce").astype(np.float64).clip(0.0, 1.0)
    out = out.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    out["team_id"] = out["team_id"].astype(int)
    out["player_id"] = out["player_id"].astype(int)

    out["player_id"] = pd.Series(
        _map_person_id_to_internal(out["player_id"].tolist(), person_id_to_internal_id=person_id_to_internal_id),
        index=out.index,
        dtype="Int64",
    )
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    if game_id_allow is not None:
        out = out[out["game_id"].isin({str(g) for g in game_id_allow})].copy()
    if team_id_allow is not None:
        out = out[out["team_id"].isin({int(t) for t in team_id_allow})].copy()

    out["pred_source"] = "cached_all"
    out = out.sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    out = out.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()
    return out[["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred", "pred_source"]].copy()


def predict_probs_from_features(
    bundle: RotationPredictorBundle,
    *,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run LightGBM predictors on a feature frame.

    Expects columns: game_id, team_id, player_id (personId or internal) plus bundle.feature_columns.
    Returns input columns plus p_ge5_pred/p_ge15_pred.
    """
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "lightgbm is required for predict_probs_from_features() but could not be imported. "
            "Install deps (uv sync) or use --gate-feature-source cached_all/cached_preds."
        ) from e

    missing = [c for c in bundle.feature_columns if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature_df missing required feature columns: {missing[:20]} (and {max(0, len(missing) - 20)} more)")

    X = feature_df[list(bundle.feature_columns)].astype(float)
    ge5 = lgb.Booster(model_file=str(bundle.model_ge5_path))
    ge15 = lgb.Booster(model_file=str(bundle.model_ge15_path))

    out = feature_df.copy()
    out["p_ge5_pred"] = pd.Series(ge5.predict(X), index=out.index, dtype=np.float64).clip(0.0, 1.0)
    out["p_ge15_pred"] = pd.Series(ge15.predict(X), index=out.index, dtype=np.float64).clip(0.0, 1.0)
    return out


def load_cached_train_predictions(
    bundle: RotationPredictorBundle,
    *,
    person_id_to_internal_id: Optional[dict[int, int]] = None,
    game_id_allow: Optional[set[str]] = None,
    team_id_allow: Optional[set[int]] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load cached training-dataset features and run predictor models to generate probabilities.

    This is intended for eval-only staged integration until live features are computed in-pipeline.
    """
    if bundle.dataset_dir is None:
        raise ValueError("rotation predictor meta.json is missing dataset_dir (required for cached_train)")
    features_path = bundle.dataset_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing predictor dataset features.parquet: {features_path}")

    cols = ["game_id", "team_id", "player_id", *bundle.feature_columns]
    cols = list(dict.fromkeys(cols))  # preserve order, drop dupes
    df = pd.read_parquet(features_path, columns=cols)
    if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
        # Deterministic downsample: stable sort then head.
        df = df.sort_values(["game_id", "team_id", "player_id"], kind="mergesort").head(int(max_rows)).copy()

    df = df.copy()
    df["game_id"] = df["game_id"].map(canonicalize_game_id).astype("string")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    if game_id_allow is not None:
        df = df[df["game_id"].isin({str(g) for g in game_id_allow})].copy()
    if team_id_allow is not None:
        df = df[df["team_id"].isin({int(t) for t in team_id_allow})].copy()

    if df.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred", "pred_source"])

    pred = predict_probs_from_features(bundle, feature_df=df)
    pred = pred[["game_id", "team_id", "player_id", "p_ge5_pred", "p_ge15_pred"]].copy()

    pred["player_id"] = pd.Series(
        _map_person_id_to_internal(pred["player_id"].tolist(), person_id_to_internal_id=person_id_to_internal_id),
        index=pred.index,
        dtype="Int64",
    )
    pred = pred.dropna(subset=["player_id"]).copy()
    pred["player_id"] = pred["player_id"].astype(int)

    pred["pred_source"] = "cached_train"
    pred = pred.sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    pred = pred.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()
    return pred
