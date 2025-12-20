"""Fit a lineup popularity model from historical DK contest results (bronze)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from projections.paths import data_path

from .field_weight_model import FieldWeightModel, FieldWeightModelBucket, save_field_weight_model

logger = logging.getLogger(__name__)

CONTEST_INDEX_PATTERN = re.compile(r"^nba_gpp_(\d{4}-\d{2}-\d{2})\.csv$")
RESULT_FILE_PATTERN = re.compile(r"^contest_(\d+)_results\.csv$")


@dataclass(frozen=True)
class FieldWeightCalibrationConfig:
    min_field_size: int = 5000
    max_field_size: int = 250000
    max_dates: int = 30
    contests_per_date: int = 2
    max_unique_lineups_per_contest: int = 50000
    ridge_alpha: float = 1.0
    random_seed: int = 7

    # Ownership source (bronze)
    use_own_pct_weighted: bool = True


def _gpp_root(data_root: Optional[Path] = None) -> Path:
    root = data_root or data_path()
    return root / "bronze" / "dk_contests" / "nba_gpp_data"


def _ownership_by_date_path(game_date: str, data_root: Optional[Path] = None) -> Path:
    root = data_root or data_path()
    return root / "bronze" / "dk_contests" / "ownership_by_date" / f"{game_date}.parquet"


def iter_available_dates(data_root: Optional[Path] = None) -> List[str]:
    root = _gpp_root(data_root)
    if not root.exists():
        return []
    dates = [p.name for p in root.iterdir() if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name)]
    return sorted(dates, reverse=True)


def load_contest_index(game_date: str, data_root: Optional[Path] = None) -> pd.DataFrame:
    root = _gpp_root(data_root)
    path = root / game_date / f"nba_gpp_{game_date}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "").str.replace("ï»¿", "")
    return df


def iter_result_files(game_date: str, data_root: Optional[Path] = None) -> Iterator[Tuple[str, int, Path]]:
    root = _gpp_root(data_root) / game_date / "results"
    if not root.exists():
        return
    for path in sorted(root.glob("contest_*_results.csv")):
        if path.name.startswith("._"):
            continue
        match = RESULT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        contest_id = int(match.group(1))
        yield game_date, contest_id, path


def _read_csv_maybe_zipped(path: Path) -> pd.DataFrame:
    """Read a contest standings CSV.

    Some historical downloads are saved as ZIP archives with a .csv suffix.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        pass

    try:
        with zipfile.ZipFile(path) as zf:
            # Prefer the standings file if present
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV members found in {path}")
            preferred = next((n for n in names if "standings" in n.lower()), names[0])
            with zf.open(preferred) as f:
                return pd.read_csv(f, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Failed to read results file {path}: {exc}") from exc


def load_contest_results(contest_results_path: Path) -> pd.DataFrame:
    df = _read_csv_maybe_zipped(contest_results_path)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "").str.replace("ï»¿", "")
    return df


def load_ownership_lookup(game_date: str, *, cfg: FieldWeightCalibrationConfig, data_root: Optional[Path] = None) -> Dict[str, float]:
    path = _ownership_by_date_path(game_date, data_root)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    col = "own_pct_weighted" if cfg.use_own_pct_weighted and "own_pct_weighted" in df.columns else "own_pct"
    lookup: Dict[str, float] = {}
    for _, row in df.iterrows():
        name = row.get("Player")
        if pd.isna(name):
            continue
        lookup[str(name)] = float(row.get(col) or 0.0)
    return lookup


def parse_lineup_players(lineup_str: object) -> List[str]:
    # Reuse the same parsing logic as contest analysis utilities.
    from projections.api.contest_service import parse_lineup  # local import to keep CLI light

    if lineup_str is None or (isinstance(lineup_str, float) and math.isnan(lineup_str)):
        return []
    return parse_lineup(str(lineup_str))


def field_size_bucket(field_size: int) -> str:
    if field_size < 5000:
        return "<5k"
    if field_size < 20000:
        return "5k-20k"
    if field_size < 50000:
        return "20k-50k"
    return "50k+"


def contest_is_gpp_classic(contest_name: str) -> bool:
    name = contest_name.lower()
    if "showdown" in name or "single game" in name or "tiers" in name:
        return False
    if "in-game" in name or "2h" in name or "1h" in name or "2nd half" in name:
        return False
    if "late night" in name or "(night)" in name:
        return False
    return True


def build_training_rows_for_contest(
    *,
    game_date: str,
    contest_id: int,
    contest_results_path: Path,
    contest_meta: Dict[str, object],
    cfg: FieldWeightCalibrationConfig,
    data_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    df = load_contest_results(contest_results_path)
    if "Lineup" not in df.columns:
        return []

    # Standings can include late entries with missing lineup; drop those.
    entries = df[df["Lineup"].notna()].copy()
    if entries.empty:
        return []

    own_lookup = load_ownership_lookup(game_date, cfg=cfg, data_root=data_root)

    lineup_counts: Counter[Tuple[str, ...]] = Counter()
    lineup_players: Dict[Tuple[str, ...], List[str]] = {}

    for lineup_str in entries["Lineup"].tolist():
        players = parse_lineup_players(lineup_str)
        if len(players) < 8:
            continue
        key = tuple(sorted(players))
        lineup_counts[key] += 1
        lineup_players.setdefault(key, players)

    if not lineup_counts:
        return []

    # Optionally subsample to keep training bounded.
    items = list(lineup_counts.items())
    if len(items) > cfg.max_unique_lineups_per_contest:
        rng = np.random.default_rng(cfg.random_seed)
        keep_idx = rng.choice(len(items), size=cfg.max_unique_lineups_per_contest, replace=False)
        items = [items[int(i)] for i in keep_idx]

    field_size = int(contest_meta.get("current_entries") or 0)
    max_entries = int(contest_meta.get("max_entries") or 0)
    draft_group_id = int(contest_meta.get("draft_group_id") or 0)

    rows: List[Dict[str, object]] = []
    for key, entry_count in items:
        players = lineup_players.get(key) or list(key)
        owns = [float(own_lookup.get(p, 0.0)) for p in players]
        missing_own = sum(1 for o in owns if o <= 0)
        # Skip if ownership lookup isn't covering the lineup.
        if missing_own > 0:
            continue

        sum_own = float(sum(owns))
        row = {
            "game_date": game_date,
            "contest_id": int(contest_id),
            "draft_group_id": draft_group_id,
            "field_size": field_size,
            "max_entries": max_entries,
            "field_bucket": field_size_bucket(field_size),
            "entry_count": int(entry_count),
            "log_entry_count": float(math.log(max(1, int(entry_count)))),
            "sum_own": sum_own,
            "num_under_5": int(sum(1 for o in owns if o < 5)),
            "num_under_10": int(sum(1 for o in owns if o < 10)),
            "num_over_50": int(sum(1 for o in owns if o > 50)),
        }
        rows.append(row)
    return rows


def collect_training_frame(
    cfg: FieldWeightCalibrationConfig,
    *,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    all_dates = iter_available_dates(data_root)
    dates: List[str] = []
    for day in all_dates:
        if _ownership_by_date_path(day, data_root).exists():
            dates.append(day)
        if len(dates) >= cfg.max_dates:
            break

    all_rows: List[Dict[str, object]] = []
    for game_date in dates:
        try:
            contests = load_contest_index(game_date, data_root)
        except FileNotFoundError:
            continue

        # Filter to large-field classic GPPs.
        contests = contests.copy()
        contests["contest_name"] = contests["contest_name"].astype(str)
        contests = contests[contests["game_type"].astype(str).str.lower() == "classic"]
        contests = contests[contests["current_entries"].fillna(0).astype(int).between(cfg.min_field_size, cfg.max_field_size)]
        contests = contests[contests["contest_name"].apply(contest_is_gpp_classic)]
        if contests.empty:
            continue

        contests = contests.sort_values("current_entries", ascending=False).head(cfg.contests_per_date)

        meta_by_id = {
            int(row["contest_id"]): row.to_dict()
            for _, row in contests.iterrows()
            if pd.notna(row.get("contest_id"))
        }

        for _date, contest_id, path in iter_result_files(game_date, data_root):
            meta = meta_by_id.get(int(contest_id))
            if meta is None:
                continue
            try:
                rows = build_training_rows_for_contest(
                    game_date=game_date,
                    contest_id=contest_id,
                    contest_results_path=path,
                    contest_meta=meta,
                    cfg=cfg,
                    data_root=data_root,
                )
                all_rows.extend(rows)
            except Exception as exc:
                logger.warning("Skipping contest %s/%s (%s): %s", game_date, contest_id, path.name, exc)
                continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def fit_weight_model(
    training: pd.DataFrame,
    *,
    cfg: FieldWeightCalibrationConfig,
) -> FieldWeightModel:
    features = ["sum_own", "num_under_5", "num_under_10", "num_over_50"]
    buckets = sorted(training["field_bucket"].dropna().unique().tolist())

    model_buckets: Dict[str, FieldWeightModelBucket] = {}
    for bucket in buckets:
        sub = training[training["field_bucket"] == bucket].copy()
        if len(sub) < 1000:
            continue
        X = sub[features].astype(float).values
        y = sub["log_entry_count"].astype(float).values
        sample_weight = np.sqrt(sub["entry_count"].astype(float).values)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=float(cfg.ridge_alpha), fit_intercept=True)
        ridge.fit(Xs, y, sample_weight=sample_weight)

        coef = {name: float(val) for name, val in zip(features, ridge.coef_, strict=True)}
        means = {name: float(val) for name, val in zip(features, scaler.mean_, strict=True)}
        stds = {name: float(val) for name, val in zip(features, scaler.scale_, strict=True)}
        model_buckets[bucket] = FieldWeightModelBucket(
            intercept=float(ridge.intercept_),
            coef=coef,
            feature_means=means,
            feature_stds=stds,
        )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "ridge_alpha": float(cfg.ridge_alpha),
        "min_field_size": int(cfg.min_field_size),
        "max_field_size": int(cfg.max_field_size),
        "max_dates": int(cfg.max_dates),
        "contests_per_date": int(cfg.contests_per_date),
        "max_unique_lineups_per_contest": int(cfg.max_unique_lineups_per_contest),
        "use_own_pct_weighted": bool(cfg.use_own_pct_weighted),
        "n_rows": int(len(training)),
        "bucket_counts": training["field_bucket"].value_counts().to_dict(),
    }
    return FieldWeightModel(buckets=model_buckets, meta=meta)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate field lineup weights from bronze contest results.")
    parser.add_argument("--min-field-size", type=int, default=5000)
    parser.add_argument("--max-field-size", type=int, default=250000)
    parser.add_argument("--max-dates", type=int, default=30)
    parser.add_argument("--contests-per-date", type=int, default=2)
    parser.add_argument("--max-unique-lineups-per-contest", type=int, default=50000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--out", type=str, default=str(data_path("gold", "field_weight_model_v1.json")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = FieldWeightCalibrationConfig(
        min_field_size=args.min_field_size,
        max_field_size=args.max_field_size,
        max_dates=args.max_dates,
        contests_per_date=args.contests_per_date,
        max_unique_lineups_per_contest=args.max_unique_lineups_per_contest,
        ridge_alpha=args.ridge_alpha,
    )

    training = collect_training_frame(cfg)
    if training.empty:
        raise SystemExit("No training rows found (check bronze paths and filters).")

    model = fit_weight_model(training, cfg=cfg)
    out_path = Path(args.out).expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"out": str(out_path), "meta": model.meta, "buckets": list(model.buckets.keys())}, indent=2))
        return 0

    save_field_weight_model(model, out_path)
    print(f"Wrote field weight model -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
