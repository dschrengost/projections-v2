"""Evaluate ownership projections produced via the production (live) path.

This evaluator joins:
  - DK actual ownership (bronze/dk_contests/ownership_by_slate/*.parquet)
  - Live ownership predictions (silver/ownership_predictions/<date>/*.parquet)

Because DK "slate_id" (from contest exports) is not the same as DK "draft_group_id"
(from salary/draftables), we map slates by player-pool overlap before joining rows.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from projections.ownership_v1.evaluation import evaluate_predictions
from projections.paths import data_path


_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(val: object, *, strip_suffix: bool = False) -> str:
    if val is None or pd.isna(val):
        return ""
    normalized = unicodedata.normalize("NFKD", str(val))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_only = re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()
    if not ascii_only:
        return ""
    tokens = [t for t in ascii_only.split() if t]
    if strip_suffix and tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens = tokens[:-1]
    return "".join(tokens)


def _candidate_game_dates(game_date_str: str, *, max_day_offset: int = 0) -> list[str]:
    base = date.fromisoformat(game_date_str)
    candidates = [base + timedelta(days=d) for d in range(-max_day_offset, max_day_offset + 1)]
    return [d.isoformat() for d in candidates]


@dataclass(frozen=True)
class SlateMatch:
    source_slate_id: str
    source_game_date: str
    target_slate_id: str | None
    target_game_date: str | None
    source_players: int
    target_players: int
    intersection: int
    recall_source: float
    recall_target: float
    overlap_coeff: float
    jaccard: float
    date_offset_days: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_slate_index(
    df: pd.DataFrame,
    *,
    slate_id_col: str,
    game_date_col: str,
    player_col: str,
) -> pd.DataFrame:
    required = {slate_id_col, game_date_col, player_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    working = df[[slate_id_col, game_date_col, player_col]].copy()
    working[slate_id_col] = working[slate_id_col].astype(str)
    working[game_date_col] = working[game_date_col].astype(str)
    working[player_col] = working[player_col].astype(str)
    working = working[working[player_col].ne("")].copy()

    def _to_set(s: pd.Series) -> set[str]:
        return set(s.tolist())

    idx = (
        working.groupby(slate_id_col, sort=False)
        .agg(
            game_date=(game_date_col, "first"),
            player_set=(player_col, _to_set),
            n_players=(player_col, "size"),
        )
        .reset_index()
    )
    return idx


def match_slates_by_player_overlap(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_slate_id_col: str,
    target_slate_id_col: str,
    game_date_col: str = "game_date",
    player_col: str = "player_name_norm",
    max_day_offset: int = 0,
    min_overlap_coeff: float = 0.85,
    min_intersection: int = 50,
) -> tuple[pd.DataFrame, list[SlateMatch]]:
    """Map each source slate_id to the best matching target slate_id by overlap.

    Uses overlap coefficient = |A∩B| / min(|A|,|B|) to allow subset slates.
    Keeps at most one source slate per target slate (best match wins).
    """

    source_idx = _build_slate_index(
        source,
        slate_id_col=source_slate_id_col,
        game_date_col=game_date_col,
        player_col=player_col,
    )
    target_idx = _build_slate_index(
        target,
        slate_id_col=target_slate_id_col,
        game_date_col=game_date_col,
        player_col=player_col,
    )

    by_date: dict[str, list[tuple[str, set[str], int]]] = {}
    for _, row in target_idx.iterrows():
        gd = str(row["game_date"])
        if not gd or gd.lower() == "nan":
            continue
        by_date.setdefault(gd, []).append((str(row[target_slate_id_col]), set(row["player_set"]), int(row["n_players"])))

    mapping: dict[str, str] = {}
    matches: list[SlateMatch] = []

    for _, row in source_idx.iterrows():
        src_id = str(row[source_slate_id_col])
        src_date = str(row["game_date"])
        try:
            _ = date.fromisoformat(src_date)
        except ValueError:
            matches.append(
                SlateMatch(
                    source_slate_id=src_id,
                    source_game_date=src_date,
                    target_slate_id=None,
                    target_game_date=None,
                    source_players=int(row["n_players"]),
                    target_players=0,
                    intersection=0,
                    recall_source=0.0,
                    recall_target=0.0,
                    overlap_coeff=0.0,
                    jaccard=0.0,
                    date_offset_days=None,
                )
            )
            continue

        src_set = set(row["player_set"])
        src_n = len(src_set)

        best: tuple[float, int, float, str, str, int, float, float, int] | None = None
        # (
        #   overlap_coeff, intersection, jaccard,
        #   tgt_id, tgt_date, tgt_n,
        #   recall_src, recall_tgt, date_offset
        # )

        for cand_date in _candidate_game_dates(src_date, max_day_offset=max_day_offset):
            for tgt_id, tgt_set, tgt_n in by_date.get(cand_date, []):
                inter = len(src_set & tgt_set)
                if inter == 0:
                    continue
                denom = min(src_n, tgt_n)
                overlap = inter / denom if denom else 0.0
                union = len(src_set | tgt_set)
                jacc = inter / union if union else 0.0
                rec_src = inter / src_n if src_n else 0.0
                rec_tgt = inter / tgt_n if tgt_n else 0.0
                offset = (date.fromisoformat(cand_date) - date.fromisoformat(src_date)).days

                cand = (overlap, inter, jacc, tgt_id, cand_date, tgt_n, rec_src, rec_tgt, offset)
                if best is None:
                    best = cand
                    continue
                # Prefer overlap, then intersection, then jaccard.
                if cand[:3] > best[:3]:
                    best = cand

        if best is None:
            matches.append(
                SlateMatch(
                    source_slate_id=src_id,
                    source_game_date=src_date,
                    target_slate_id=None,
                    target_game_date=None,
                    source_players=src_n,
                    target_players=0,
                    intersection=0,
                    recall_source=0.0,
                    recall_target=0.0,
                    overlap_coeff=0.0,
                    jaccard=0.0,
                    date_offset_days=None,
                )
            )
            continue

        overlap, inter, jacc, tgt_id, tgt_date, tgt_n, rec_src, rec_tgt, offset = best
        if overlap < min_overlap_coeff or inter < min_intersection:
            matches.append(
                SlateMatch(
                    source_slate_id=src_id,
                    source_game_date=src_date,
                    target_slate_id=None,
                    target_game_date=None,
                    source_players=src_n,
                    target_players=int(tgt_n),
                    intersection=int(inter),
                    recall_source=float(rec_src),
                    recall_target=float(rec_tgt),
                    overlap_coeff=float(overlap),
                    jaccard=float(jacc),
                    date_offset_days=int(offset),
                )
            )
            continue

        mapping[src_id] = tgt_id
        matches.append(
            SlateMatch(
                source_slate_id=src_id,
                source_game_date=src_date,
                target_slate_id=str(tgt_id),
                target_game_date=str(tgt_date),
                source_players=src_n,
                target_players=int(tgt_n),
                intersection=int(inter),
                recall_source=float(rec_src),
                recall_target=float(rec_tgt),
                overlap_coeff=float(overlap),
                jaccard=float(jacc),
                date_offset_days=int(offset),
            )
        )

    # Keep only the best-overlap source slate per target slate id.
    best_by_target: dict[str, SlateMatch] = {}
    for m in matches:
        if m.target_slate_id is None:
            continue
        prev = best_by_target.get(m.target_slate_id)
        if prev is None:
            best_by_target[m.target_slate_id] = m
            continue
        score = (m.overlap_coeff, m.intersection, m.jaccard)
        prev_score = (prev.overlap_coeff, prev.intersection, prev.jaccard)
        if score > prev_score:
            best_by_target[m.target_slate_id] = m

    allowed_sources = {m.source_slate_id for m in best_by_target.values()}
    mapping = {src: tgt for src, tgt in mapping.items() if src in allowed_sources}

    out = source.copy()
    out[source_slate_id_col] = out[source_slate_id_col].astype(str)
    out[target_slate_id_col] = out[source_slate_id_col].map(mapping).astype("string")
    return out, matches


def _load_dk_actual_for_date(actual_root: Path, game_date: date) -> pd.DataFrame:
    files = sorted(actual_root.glob(f"{game_date.isoformat()}_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if "slate_id" not in df.columns:
        df["slate_id"] = df.get("slateId")
    df["game_date"] = df.get("game_date", game_date.isoformat()).astype(str)
    df["player_name_norm"] = df["Player"].map(lambda x: normalize_name(x, strip_suffix=False))
    df["player_name_norm_base"] = df["Player"].map(lambda x: normalize_name(x, strip_suffix=True))
    df["actual_own_pct"] = pd.to_numeric(df["own_pct"], errors="coerce")
    return df


def _load_live_preds_for_date(pred_root: Path, game_date: date, *, snapshot: str = "locked") -> pd.DataFrame:
    day_dir = pred_root / game_date.isoformat()
    if day_dir.is_file() and day_dir.suffix == ".parquet":
        df = pd.read_parquet(day_dir)
        df = df.copy()
        df["draft_group_id"] = "legacy"
        df["game_date"] = df.get("game_date", game_date).astype(str)
        df["player_name_norm"] = df["player_name"].map(lambda x: normalize_name(x, strip_suffix=False))
        df["player_name_norm_base"] = df["player_name"].map(lambda x: normalize_name(x, strip_suffix=True))
        return df

    if not day_dir.exists():
        return pd.DataFrame()

    parquet_files = sorted([p for p in day_dir.glob("*.parquet") if p.name != "slates.json"])
    if not parquet_files:
        return pd.DataFrame()

    # Prefer locked snapshot per draft_group_id when available.
    chosen: dict[str, Path] = {}
    for p in parquet_files:
        name = p.name
        locked = name.endswith("_locked.parquet")
        if locked:
            dg = name[: -len("_locked.parquet")]
        else:
            if not name.endswith(".parquet"):
                continue
            dg = name[: -len(".parquet")]
        if not dg.isdigit():
            continue
        if snapshot == "locked":
            if locked:
                chosen[dg] = p
            else:
                chosen.setdefault(dg, p)
        else:
            if not locked:
                chosen[dg] = p

    dfs: list[pd.DataFrame] = []
    for dg, path in sorted(chosen.items()):
        df = pd.read_parquet(path)
        df = df.copy()
        df["draft_group_id"] = df.get("draft_group_id", dg).astype(str)
        df["game_date"] = df.get("game_date", game_date).astype(str)
        if "pred_own_pct_raw" not in df.columns:
            df["pred_own_pct_raw"] = df["pred_own_pct"].astype(float)
        df["player_name_norm"] = df["player_name"].map(lambda x: normalize_name(x, strip_suffix=False))
        df["player_name_norm_base"] = df["player_name"].map(lambda x: normalize_name(x, strip_suffix=True))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _select_actual_slates(actual: pd.DataFrame, *, selector: str) -> list[str]:
    if actual.empty:
        return []
    if selector == "all":
        return sorted(actual["slate_id"].astype(str).unique().tolist())
    if selector == "largest_entries":
        by = actual.groupby("slate_id", sort=False)["slate_entries"].first()
        return [str(by.idxmax())] if not by.empty else []
    if selector == "largest_size":
        by = actual.groupby("slate_id", sort=False)["slate_size"].first()
        return [str(by.idxmax())] if not by.empty else []
    raise ValueError(f"Unknown slate selector: {selector}")


def _join_slate_pair(
    actual_slate: pd.DataFrame,
    pred_slate: pd.DataFrame,
    *,
    slate_id: str,
    draft_group_id: str,
) -> pd.DataFrame:
    a = actual_slate.copy()
    p = pred_slate.copy()
    a = a.dropna(subset=["actual_own_pct"]).copy()

    # First pass: strict match.
    a1 = a[["player_name_norm", "player_name_norm_base", "actual_own_pct", "own_pct_simple"]].copy()
    out = p.merge(a1, how="left", on="player_name_norm", suffixes=("", "_actual"))
    out["_match_type"] = np.where(out["actual_own_pct"].notna(), "exact", "none")

    # Fallback: base key (suffix-stripped).
    need = out["actual_own_pct"].isna()
    if need.any():
        a2 = (
            a[["player_name_norm_base", "actual_own_pct", "own_pct_simple"]]
            .rename(columns={"player_name_norm_base": "_k"})
            .drop_duplicates(subset=["_k"])
        )
        out2 = out.loc[need].drop(columns=["actual_own_pct", "own_pct_simple"], errors="ignore").copy()
        out2 = out2.merge(a2, how="left", left_on="player_name_norm_base", right_on="_k")
        out.loc[need, "actual_own_pct"] = out2["actual_own_pct"].to_numpy()
        out.loc[need, "own_pct_simple"] = out2["own_pct_simple"].to_numpy()
        out.loc[need & out["actual_own_pct"].notna(), "_match_type"] = "base"

    out["slate_id"] = str(slate_id)
    out["draft_group_id"] = str(draft_group_id)
    return out


def _render_markdown(
    title: str,
    *,
    date_min: str,
    date_max: str,
    n_slates: int,
    n_rows: int,
    join_rate_mean: float,
    raw: dict,
    scaled: dict,
    prod: dict | None,
) -> str:
    def fmt(x: float) -> str:
        if x != x:
            return "NaN"
        return f"{x:.4f}"

    def get(d: dict, path: str) -> float:
        cur = d
        for part in path.split("."):
            cur = cur[part]
        return float(cur)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Dates: {date_min}..{date_max}")
    lines.append(f"- n_slates: {n_slates}, n_rows (joined): {n_rows}")
    lines.append(f"- Mean join rate (min-set overlap): {join_rate_mean:.4f}")
    lines.append("")

    lines.append("## Raw Model (pred_own_pct_raw)")
    lines.append(f"- Raw MAE/RMSE: {fmt(get(raw, 'regression.mae_pct'))} / {fmt(get(raw, 'regression.rmse_pct'))}")
    lines.append(f"- Spearman pooled: {fmt(get(raw, 'ranking.spearman_pooled'))}")
    lines.append(f"- Spearman top10/top20: {fmt(get(raw, 'ranking.spearman_top10_mean'))} / {fmt(get(raw, 'ranking.spearman_top20_mean'))}")
    lines.append(f"- Recall@10/20: {fmt(get(raw, 'ranking.recall_at_10'))} / {fmt(get(raw, 'ranking.recall_at_20'))}")
    lines.append(f"- ECE: {fmt(get(raw, 'calibration.ece_pct'))}")
    lines.append("")

    lines.append("## Scaled-to-Sum (scale pred_own_pct_raw to 800%)")
    lines.append(f"- MAE/RMSE: {fmt(get(scaled, 'regression.mae_pct'))} / {fmt(get(scaled, 'regression.rmse_pct'))}")
    lines.append(f"- Spearman pooled: {fmt(get(scaled, 'ranking.spearman_pooled'))}")
    lines.append(f"- Spearman top10/top20: {fmt(get(scaled, 'ranking.spearman_top10_mean'))} / {fmt(get(scaled, 'ranking.spearman_top20_mean'))}")
    lines.append(f"- Recall@10/20: {fmt(get(scaled, 'ranking.recall_at_10'))} / {fmt(get(scaled, 'ranking.recall_at_20'))}")
    lines.append(f"- ECE: {fmt(get(scaled, 'calibration.ece_pct'))}")
    lines.append("")

    if prod is not None:
        lines.append("## Production Output (pred_own_pct)")
        lines.append(f"- MAE/RMSE: {fmt(get(prod, 'regression.mae_pct'))} / {fmt(get(prod, 'regression.rmse_pct'))}")
        lines.append(f"- Spearman pooled: {fmt(get(prod, 'ranking.spearman_pooled'))}")
        lines.append(f"- Spearman top10/top20: {fmt(get(prod, 'ranking.spearman_top10_mean'))} / {fmt(get(prod, 'ranking.spearman_top20_mean'))}")
        lines.append(f"- Recall@10/20: {fmt(get(prod, 'ranking.recall_at_10'))} / {fmt(get(prod, 'ranking.recall_at_20'))}")
        lines.append(f"- ECE: {fmt(get(prod, 'calibration.ece_pct'))}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate production-path ownership predictions vs DK actuals")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to projections.paths.data_path()")
    parser.add_argument("--pred-root", type=Path, default=None, help="Defaults to <data_root>/silver/ownership_predictions")
    parser.add_argument(
        "--actual-root",
        type=Path,
        default=None,
        help="Defaults to <data_root>/bronze/dk_contests/ownership_by_slate",
    )
    parser.add_argument(
        "--slate-selector",
        choices=["all", "largest_entries", "largest_size"],
        default="largest_entries",
        help="Which actual DK slates to evaluate per date (default: largest_entries, approximates main slate).",
    )
    parser.add_argument("--pred-snapshot", choices=["locked", "latest"], default="locked")
    parser.add_argument("--min-overlap-coeff", type=float, default=0.85)
    parser.add_argument("--min-intersection", type=int, default=50)
    parser.add_argument("--max-day-offset", type=int, default=0)
    parser.add_argument("--target-sum-pct", type=float, default=800.0)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-parquet", type=Path, default=None, help="Optional: write joined rows to parquet")
    args = parser.parse_args()

    root = args.data_root if args.data_root is not None else data_path()
    pred_root = args.pred_root if args.pred_root is not None else root / "silver" / "ownership_predictions"
    actual_root = (
        args.actual_root if args.actual_root is not None else root / "bronze" / "dk_contests" / "ownership_by_slate"
    )

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be >= --start-date")

    joined: list[pd.DataFrame] = []
    all_matches: list[SlateMatch] = []
    join_rates: list[float] = []

    d = start
    while d <= end:
        actual = _load_dk_actual_for_date(actual_root, d)
        preds = _load_live_preds_for_date(pred_root, d, snapshot=args.pred_snapshot)
        if actual.empty or preds.empty:
            d = d + timedelta(days=1)
            continue

        actual_slates = _select_actual_slates(actual, selector=args.slate_selector)
        if not actual_slates:
            d = d + timedelta(days=1)
            continue

        # Restrict to selected actual slates, then map to predicted draft groups.
        actual_sel = actual[actual["slate_id"].astype(str).isin(set(actual_slates))].copy()
        actual_long = actual_sel[["slate_id", "game_date", "player_name_norm"]].copy()
        preds_long = preds[["draft_group_id", "game_date", "player_name_norm"]].copy()

        mapped, matches = match_slates_by_player_overlap(
            actual_long,
            preds_long,
            source_slate_id_col="slate_id",
            target_slate_id_col="draft_group_id",
            game_date_col="game_date",
            player_col="player_name_norm",
            max_day_offset=args.max_day_offset,
            min_overlap_coeff=args.min_overlap_coeff,
            min_intersection=args.min_intersection,
        )
        all_matches.extend(matches)

        for slate_id, g in mapped.groupby("slate_id", sort=False):
            dg = g["draft_group_id"].dropna().astype(str).unique().tolist()
            if not dg:
                continue
            dg_id = dg[0]
            actual_slate = actual_sel[actual_sel["slate_id"].astype(str) == str(slate_id)].copy()
            pred_slate = preds[preds["draft_group_id"].astype(str) == str(dg_id)].copy()
            if actual_slate.empty or pred_slate.empty:
                continue

            inter = len(set(actual_slate["player_name_norm"].tolist()) & set(pred_slate["player_name_norm"].tolist()))
            denom = min(actual_slate["player_name_norm"].nunique(), pred_slate["player_name_norm"].nunique())
            join_rates.append(inter / denom if denom else 0.0)

            joined.append(_join_slate_pair(actual_slate, pred_slate, slate_id=str(slate_id), draft_group_id=str(dg_id)))

        d = d + timedelta(days=1)

    if not joined:
        raise SystemExit("No joined rows found (missing preds/actuals or mapping thresholds too strict).")

    df = pd.concat(joined, ignore_index=True)
    df = df.dropna(subset=["actual_own_pct"]).copy()

    raw_res = evaluate_predictions(
        df,
        slice_name=f"prod_{args.start_date}_to_{args.end_date}",
        target_sum_pct=float(args.target_sum_pct),
        pred_col="pred_own_pct_raw",
        normalization="none",
    )
    scaled_res = evaluate_predictions(
        df,
        slice_name=f"prod_{args.start_date}_to_{args.end_date}",
        target_sum_pct=float(args.target_sum_pct),
        pred_col="pred_own_pct_raw",
        normalization="scale_to_sum",
    )

    prod_res = None
    if "pred_own_pct" in df.columns:
        prod_res = evaluate_predictions(
            df,
            slice_name=f"prod_{args.start_date}_to_{args.end_date}",
            target_sum_pct=float(args.target_sum_pct),
            pred_col="pred_own_pct",
            normalization="none",
        )

    payload = {
        "date_range": {"start": args.start_date, "end": args.end_date},
        "target_sum_pct": float(args.target_sum_pct),
        "slate_selector": args.slate_selector,
        "pred_snapshot": args.pred_snapshot,
        "mapping": {
            "n_match_records": len(all_matches),
            "matches": [m.to_dict() for m in all_matches],
        },
        "join": {
            "n_rows": int(len(df)),
            "n_slates": int(df["slate_id"].nunique()),
            "join_rate_mean": float(np.mean(join_rates)) if join_rates else float("nan"),
            "join_rate_min": float(np.min(join_rates)) if join_rates else float("nan"),
        },
        "raw_model": raw_res.to_dict(),
        "scaled_to_sum": scaled_res.to_dict(),
        "production_output": prod_res.to_dict() if prod_res is not None else None,
    }

    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        md = _render_markdown(
            "Ownership Production-Path Eval",
            date_min=args.start_date,
            date_max=args.end_date,
            n_slates=int(df["slate_id"].nunique()),
            n_rows=int(len(df)),
            join_rate_mean=float(payload["join"]["join_rate_mean"]),
            raw=raw_res.to_dict(),
            scaled=scaled_res.to_dict(),
            prod=prod_res.to_dict() if prod_res is not None else None,
        )
        args.out_md.write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
