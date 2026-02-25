"""Action Network player-props feature extraction for minutes/rates builders."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from unidecode import unidecode

_TEAM_ABBR_TO_NBA: dict[str, str] = {
    "PHO": "PHX",
    "GS": "GSW",
    "NO": "NOP",
    "SA": "SAS",
    "NY": "NYK",
}

_SUPPORTED_PROP_KEYS: tuple[str, ...] = ("pts", "reb", "ast", "3pm", "pra", "pr", "pa", "ra")
_NAME_SUFFIXES: set[str] = {"jr", "sr", "ii", "iii", "iv", "v"}


def _coerce_ts(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _normalize_team_abbr(value: object) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    return _TEAM_ABBR_TO_NBA.get(raw, raw)


def _normalize_player_name(value: object) -> str:
    text = unidecode(str(value or "")).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token and token not in _NAME_SUFFIXES]
    return " ".join(tokens)


def _american_to_implied_prob(odds: float) -> float | None:
    if not np.isfinite(odds) or odds == 0:
        return None
    if odds > 0:
        return float(100.0 / (odds + 100.0))
    return float((-odds) / ((-odds) + 100.0))


def _canonical_prop_key(raw_name: object) -> str | None:
    normalized = str(raw_name or "").strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace(" ", "")
    mapping = {
        "pts": "pts",
        "points": "pts",
        "rebs": "reb",
        "reb": "reb",
        "rebounds": "reb",
        "ast": "ast",
        "assists": "ast",
        "3ptm": "3pm",
        "3pm": "3pm",
        "threesmade": "3pm",
        "pts+rebs+ast": "pra",
        "points+rebounds+assists": "pra",
        "pts+rebs": "pr",
        "points+rebounds": "pr",
        "pts+ast": "pa",
        "points+assists": "pa",
        "rebs+ast": "ra",
        "rebounds+assists": "ra",
    }
    return mapping.get(normalized)


def _team_tricode_from_display_text(value: object) -> str:
    text = str(value or "")
    prefix = text.split("-", 1)[0].strip().upper()
    return _normalize_team_abbr(prefix)


def _team_lookup_from_payload(payload: dict) -> dict[int, str]:
    lookup: dict[int, str] = {}
    teams = payload.get("teams") or []
    away_id = pd.to_numeric(payload.get("away_team_id"), errors="coerce")
    home_id = pd.to_numeric(payload.get("home_team_id"), errors="coerce")
    if len(teams) >= 2:
        if pd.notna(away_id):
            lookup[int(away_id)] = _normalize_team_abbr(teams[0])
        if pd.notna(home_id):
            lookup[int(home_id)] = _normalize_team_abbr(teams[1])
    return lookup


def _summarize_market_from_lines(lines_by_book: dict | None) -> dict[str, float] | None:
    if not isinstance(lines_by_book, dict) or not lines_by_book:
        return None

    per_book_rows: list[tuple[float, float]] = []
    for _, raw_lines in lines_by_book.items():
        if not isinstance(raw_lines, list) or not raw_lines:
            continue

        by_value: dict[float, dict[str, float]] = {}
        for raw in raw_lines:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("period", "")).lower() != "event":
                continue
            side = str(raw.get("side", "")).lower()
            if side not in {"over", "under"}:
                continue
            odds = pd.to_numeric(raw.get("odds"), errors="coerce")
            value = pd.to_numeric(raw.get("value"), errors="coerce")
            if not np.isfinite(odds) or not np.isfinite(value):
                continue
            bucket = by_value.setdefault(float(value), {})
            bucket[side] = float(odds)

        candidates: list[tuple[float, float]] = []
        for line_value, sides in by_value.items():
            over_odds = sides.get("over")
            under_odds = sides.get("under")
            if over_odds is None or under_odds is None:
                continue
            p_over = _american_to_implied_prob(over_odds)
            p_under = _american_to_implied_prob(under_odds)
            if p_over is None or p_under is None:
                continue
            denom = p_over + p_under
            if denom <= 0:
                continue
            p_over_novig = p_over / denom
            candidates.append((line_value, p_over_novig))

        if not candidates:
            continue
        # Main line heuristic: closest no-vig split to 50/50 for each book.
        per_book_rows.append(min(candidates, key=lambda item: abs(item[1] - 0.5)))

    if not per_book_rows:
        return None
    lines = np.array([row[0] for row in per_book_rows], dtype=float)
    probs = np.array([row[1] for row in per_book_rows], dtype=float)
    return {
        "line": float(np.mean(lines)),
        "p_over": float(np.mean(probs)),
        "line_std": float(np.std(lines, ddof=0)) if len(lines) > 1 else 0.0,
        "books": float(len(per_book_rows)),
    }


def load_action_props_long_from_bronze(*, props_dir: Path, game_date: pd.Timestamp) -> pd.DataFrame:
    """Load normalized per-player market rows from Action bronze JSON files for one date."""

    day = pd.Timestamp(game_date).normalize().date().isoformat()
    day_ts_utc = pd.Timestamp(day).tz_localize("UTC")
    files = sorted(props_dir.glob(f"{day}_*.json"))
    if not files:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )

    rows: list[dict[str, object]] = []
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        props_payload = payload.get("props") or {}
        player_props = props_payload.get("player_props") or {}
        players_lookup = props_payload.get("players") or {}
        team_lookup = _team_lookup_from_payload(payload)
        game_id = pd.to_numeric(payload.get("game_id"), errors="coerce")
        as_of_ts = _coerce_ts(payload.get("fetched_at"))
        # Historical backfills can stamp fetched_at months after the game date.
        # Clamp clearly stale timestamps to game-day midnight UTC so strict as-of
        # filtering remains anti-leak while preserving usable training signal.
        if pd.isna(as_of_ts) or as_of_ts > (day_ts_utc + pd.Timedelta(days=2)):
            as_of_ts = day_ts_utc

        for prop_entries in player_props.values():
            if not isinstance(prop_entries, list):
                continue
            for entry in prop_entries:
                if not isinstance(entry, dict):
                    continue
                raw_prop_name = (
                    entry.get("custom_pick_type_name")
                    or entry.get("custom_pick_type_display_name")
                    or entry.get("type")
                )
                prop_key = _canonical_prop_key(raw_prop_name)
                if prop_key is None:
                    continue

                market = _summarize_market_from_lines(entry.get("lines"))
                if market is None:
                    continue

                action_player_id = str(entry.get("player_id") or "")
                player_meta = players_lookup.get(action_player_id, {}) if action_player_id else {}
                player_name = (
                    player_meta.get("full_name")
                    or entry.get("player_full_name")
                    or entry.get("player_abbr")
                    or player_meta.get("abbr")
                )
                if not player_name:
                    continue
                player_name_norm = _normalize_player_name(player_name)
                if not player_name_norm:
                    continue

                team_tricode = _team_tricode_from_display_text(player_meta.get("display_text"))
                if not team_tricode:
                    team_id_raw = player_meta.get("team_id", entry.get("team_id"))
                    team_id = pd.to_numeric(team_id_raw, errors="coerce")
                    if pd.notna(team_id):
                        team_tricode = team_lookup.get(int(team_id), "")
                if not team_tricode:
                    continue

                rows.append(
                    {
                        "game_date": pd.Timestamp(day),
                        "team_tricode": team_tricode,
                        "player_name": str(player_name),
                        "player_name_norm": player_name_norm,
                        "prop_key": prop_key,
                        "line": market["line"],
                        "p_over": market["p_over"],
                        "line_std": market["line_std"],
                        "books": market["books"],
                        "action_props_as_of_ts": as_of_ts,
                        "action_game_id": int(game_id) if pd.notna(game_id) else pd.NA,
                        "source_file": file_path.name,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )
    frame = pd.DataFrame.from_records(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
    frame["action_props_as_of_ts"] = pd.to_datetime(frame["action_props_as_of_ts"], utc=True, errors="coerce")
    frame["team_tricode"] = frame["team_tricode"].map(_normalize_team_abbr)
    return frame


def _market_feature_defaults() -> dict[str, float | int | pd.Timestamp]:
    defaults: dict[str, float | int | pd.Timestamp] = {
        "action_props_as_of_ts": pd.NaT,
        "an_has_any_props": 0,
        "an_props_market_count": 0,
        "an_props_books_total": 0.0,
        "an_implied_activity_line": 0.0,
    }
    for key in _SUPPORTED_PROP_KEYS:
        defaults[f"an_{key}_line"] = 0.0
        defaults[f"an_{key}_p_over"] = 0.5
        defaults[f"an_{key}_line_std"] = 0.0
        defaults[f"an_{key}_books"] = 0.0
        defaults[f"an_has_{key}"] = 0
    return defaults


ACTION_MARKET_FEATURE_COLUMNS: tuple[str, ...] = tuple(_market_feature_defaults().keys())


def build_action_props_feature_snapshots(action_long: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized long Action markets into per-player snapshot feature rows."""

    if action_long.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                *ACTION_MARKET_FEATURE_COLUMNS,
            ]
        )

    required = {"game_date", "team_tricode", "player_name", "player_name_norm", "prop_key", "line", "p_over", "line_std", "books", "action_props_as_of_ts"}
    if not required.issubset(action_long.columns):
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                *ACTION_MARKET_FEATURE_COLUMNS,
            ]
        )

    base = action_long.copy()
    base = base[base["prop_key"].isin(_SUPPORTED_PROP_KEYS)].copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                *ACTION_MARKET_FEATURE_COLUMNS,
            ]
        )
    for col in ("line", "p_over", "line_std", "books"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
    base["action_props_as_of_ts"] = pd.to_datetime(base["action_props_as_of_ts"], utc=True, errors="coerce")
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce").dt.normalize()
    base = base.dropna(subset=["game_date", "team_tricode", "player_name_norm", "prop_key"])
    if base.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                *ACTION_MARKET_FEATURE_COLUMNS,
            ]
        )

    idx_cols = ["game_date", "team_tricode", "player_name", "player_name_norm", "action_props_as_of_ts"]
    grouped = (
        base.groupby(idx_cols + ["prop_key"], as_index=False)
        .agg(
            line=("line", "mean"),
            p_over=("p_over", "mean"),
            line_std=("line_std", "mean"),
            books=("books", "max"),
        )
    )

    out = grouped[idx_cols].drop_duplicates().reset_index(drop=True)

    def _pivot(metric: str) -> pd.DataFrame:
        wide = grouped.pivot_table(index=idx_cols, columns="prop_key", values=metric, aggfunc="mean")
        return wide.reset_index()

    line_wide = _pivot("line")
    over_wide = _pivot("p_over")
    std_wide = _pivot("line_std")
    books_wide = _pivot("books")

    for key in _SUPPORTED_PROP_KEYS:
        out = out.merge(
            line_wide[idx_cols + [key]] if key in line_wide.columns else line_wide[idx_cols],
            on=idx_cols,
            how="left",
            suffixes=("", "_dup"),
        )
        if key in out.columns:
            out.rename(columns={key: f"an_{key}_line"}, inplace=True)
        else:
            out[f"an_{key}_line"] = np.nan

        out = out.merge(
            over_wide[idx_cols + [key]] if key in over_wide.columns else over_wide[idx_cols],
            on=idx_cols,
            how="left",
            suffixes=("", "_dup"),
        )
        if key in out.columns:
            out.rename(columns={key: f"an_{key}_p_over"}, inplace=True)
        else:
            out[f"an_{key}_p_over"] = np.nan

        out = out.merge(
            std_wide[idx_cols + [key]] if key in std_wide.columns else std_wide[idx_cols],
            on=idx_cols,
            how="left",
            suffixes=("", "_dup"),
        )
        if key in out.columns:
            out.rename(columns={key: f"an_{key}_line_std"}, inplace=True)
        else:
            out[f"an_{key}_line_std"] = np.nan

        out = out.merge(
            books_wide[idx_cols + [key]] if key in books_wide.columns else books_wide[idx_cols],
            on=idx_cols,
            how="left",
            suffixes=("", "_dup"),
        )
        if key in out.columns:
            out.rename(columns={key: f"an_{key}_books"}, inplace=True)
        else:
            out[f"an_{key}_books"] = np.nan

        out[f"an_has_{key}"] = pd.to_numeric(out[f"an_{key}_books"], errors="coerce").fillna(0).gt(0).astype(int)

    has_cols = [f"an_has_{key}" for key in _SUPPORTED_PROP_KEYS]
    book_cols = [f"an_{key}_books" for key in _SUPPORTED_PROP_KEYS]
    out["an_props_market_count"] = out[has_cols].sum(axis=1).astype(int)
    out["an_props_books_total"] = out[book_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    out["an_has_any_props"] = out["an_props_market_count"].gt(0).astype(int)

    pra = pd.to_numeric(out["an_pra_line"], errors="coerce")
    fallback = out[["an_pts_line", "an_reb_line", "an_ast_line"]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=2)
    out["an_implied_activity_line"] = pra.combine_first(fallback).fillna(0.0).astype(float)

    defaults = _market_feature_defaults()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)

    return out[["game_date", "team_tricode", "player_name", "player_name_norm", *ACTION_MARKET_FEATURE_COLUMNS]].copy()


def load_action_props_feature_snapshots_for_date(*, props_dir: Path, game_date: pd.Timestamp) -> pd.DataFrame:
    """High-level helper for date-scoped Action player-prop snapshots."""

    long_df = load_action_props_long_from_bronze(props_dir=props_dir, game_date=game_date)
    return build_action_props_feature_snapshots(long_df)


def _canonical_rotowire_prop_key(raw_name: object) -> str | None:
    normalized = str(raw_name or "").strip().lower()
    mapping = {
        "pts": "pts",
        "reb": "reb",
        "ast": "ast",
        "threes": "3pm",
        "ptsrebast": "pra",
        "ptsreb": "pr",
        "ptsast": "pa",
        "rebast": "ra",
    }
    return mapping.get(normalized)


def load_rotowire_props_long_from_bronze(
    *,
    rotowire_props_root: Path,
    game_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load normalized per-player market rows from Rotowire bronze parquet for one date."""
    day = pd.Timestamp(game_date).normalize().date().isoformat()
    partition = Path(rotowire_props_root) / f"game_date={day}"
    files = sorted(partition.glob("*.parquet")) if partition.exists() else []
    if not files:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )

    raw = pd.concat(frames, ignore_index=True)
    required_cols = {"player_name", "team", "prop_type", "line", "book", "scraped_at"}
    if not required_cols.issubset(raw.columns):
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )

    work = raw.copy()
    work["prop_key"] = work["prop_type"].map(_canonical_rotowire_prop_key)
    work = work.loc[work["prop_key"].isin(_SUPPORTED_PROP_KEYS)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )

    work["line"] = pd.to_numeric(work["line"], errors="coerce")
    work["over_odds"] = pd.to_numeric(work.get("over_odds"), errors="coerce")
    implied_over = pd.to_numeric(work.get("implied_over_prob"), errors="coerce")
    work["p_over"] = implied_over
    missing_over = work["p_over"].isna()
    if missing_over.any():
        work.loc[missing_over, "p_over"] = work.loc[missing_over, "over_odds"].map(_american_to_implied_prob)
    work["p_over"] = pd.to_numeric(work["p_over"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    work["action_props_as_of_ts"] = pd.to_datetime(work["scraped_at"], utc=True, errors="coerce")
    work["team_tricode"] = work["team"].map(_normalize_team_abbr)
    work["player_name_norm"] = work["player_name"].map(_normalize_player_name)
    work["game_date"] = pd.Timestamp(day)
    work["action_game_id"] = pd.to_numeric(work.get("game_id"), errors="coerce").astype("Int64")
    work = work.dropna(subset=["line", "action_props_as_of_ts"])
    work = work.loc[
        work["team_tricode"].astype(str).str.len().gt(0)
        & work["player_name_norm"].astype(str).str.len().gt(0)
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "line",
                "p_over",
                "line_std",
                "books",
                "action_props_as_of_ts",
                "action_game_id",
                "source_file",
            ]
        )

    grouped = (
        work.groupby(
            [
                "game_date",
                "team_tricode",
                "player_name",
                "player_name_norm",
                "prop_key",
                "action_props_as_of_ts",
            ],
            as_index=False,
        )
        .agg(
            line=("line", "mean"),
            p_over=("p_over", "mean"),
            line_std=("line", lambda s: float(np.std(pd.to_numeric(s, errors="coerce"), ddof=0))),
            books=("book", "nunique"),
            action_game_id=("action_game_id", "first"),
            source_file=("source_file", "first"),
        )
    )
    grouped["line_std"] = pd.to_numeric(grouped["line_std"], errors="coerce").fillna(0.0)
    grouped["books"] = pd.to_numeric(grouped["books"], errors="coerce").fillna(0.0)
    return grouped[
        [
            "game_date",
            "team_tricode",
            "player_name",
            "player_name_norm",
            "prop_key",
            "line",
            "p_over",
            "line_std",
            "books",
            "action_props_as_of_ts",
            "action_game_id",
            "source_file",
        ]
    ].copy()


def load_action_props_feature_snapshots_for_date_live(
    *,
    action_props_dir: Path,
    game_date: pd.Timestamp,
    allow_rotowire_fallback: bool = False,
    rotowire_props_root: Path | None = None,
    expected_team_tricodes: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load live action-props snapshots with optional Rotowire fallback.

    Returns (snapshots, source), where source is one of:
    - "action_network"
    - "rotowire_fallback"
    - "none"
    """
    action_long = load_action_props_long_from_bronze(
        props_dir=Path(action_props_dir),
        game_date=game_date,
    )
    action_snapshots = build_action_props_feature_snapshots(action_long)
    expected_teams_norm = {
        _normalize_team_abbr(team)
        for team in (expected_team_tricodes or [])
        if str(team or "").strip()
    }
    if not action_snapshots.empty:
        if expected_teams_norm:
            action_teams = {
                _normalize_team_abbr(team)
                for team in action_snapshots["team_tricode"].tolist()
                if str(team or "").strip()
            }
            if action_teams and not action_teams.isdisjoint(expected_teams_norm):
                return action_snapshots, "action_network"
            action_snapshots = action_snapshots.iloc[0:0].copy()
        else:
            return action_snapshots, "action_network"

    if not allow_rotowire_fallback:
        return action_snapshots, "none"

    root = (
        Path(rotowire_props_root)
        if rotowire_props_root is not None
        else Path(action_props_dir).parents[1] / "props"
    )
    rotowire_long = load_rotowire_props_long_from_bronze(
        rotowire_props_root=root,
        game_date=game_date,
    )
    if rotowire_long.empty:
        return action_snapshots, "none"
    return build_action_props_feature_snapshots(rotowire_long), "rotowire_fallback"


def attach_action_props_features(
    base_df: pd.DataFrame,
    action_snapshots: pd.DataFrame,
    *,
    strict_asof: bool = True,
    as_of_col: str = "feature_as_of_ts",
    tip_col: str = "tip_ts",
    game_date_offsets: tuple[int, ...] = (0,),
    clamp_late_asof_to_game_date: bool = False,
) -> pd.DataFrame:
    """Attach Action player-prop features to a base player-game frame."""

    out = base_df.copy()
    defaults = _market_feature_defaults()
    if out.empty:
        for col, default in defaults.items():
            out[col] = default
        return out

    if action_snapshots is None or action_snapshots.empty:
        for col, default in defaults.items():
            out[col] = default
        return out

    required_base = {"game_date", "team_tricode", "player_name"}
    required_snap = {"game_date", "team_tricode", "player_name_norm", *ACTION_MARKET_FEATURE_COLUMNS}
    if not required_base.issubset(out.columns) or not required_snap.issubset(action_snapshots.columns):
        for col, default in defaults.items():
            out[col] = default
        return out

    working = out.copy()
    working["_row_id"] = np.arange(len(working), dtype=int)
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce").dt.normalize()
    working["_team_tricode_norm"] = working["team_tricode"].map(_normalize_team_abbr)
    working["_player_name_norm"] = working["player_name"].map(_normalize_player_name)

    snaps = action_snapshots.copy()
    snaps["game_date"] = pd.to_datetime(snaps["game_date"], errors="coerce").dt.normalize()
    snaps["_team_tricode_norm"] = snaps["team_tricode"].map(_normalize_team_abbr)
    snaps["_player_name_norm"] = snaps["player_name_norm"].map(_normalize_player_name)
    snaps["action_props_as_of_ts"] = pd.to_datetime(snaps["action_props_as_of_ts"], utc=True, errors="coerce")

    offsets: list[int] = []
    for raw in game_date_offsets:
        try:
            offsets.append(int(raw))
        except Exception:
            continue
    if not offsets:
        offsets = [0]

    snap_frames: list[pd.DataFrame] = []
    for offset in sorted(set(offsets)):
        snap_view = snaps.copy()
        if offset != 0:
            snap_view["game_date"] = snap_view["game_date"] + pd.Timedelta(days=offset)
        snap_view["_game_date_offset_days"] = offset
        snap_frames.append(snap_view)
    snaps_join = pd.concat(snap_frames, ignore_index=True) if snap_frames else snaps

    merge_cols = ["game_date", "_team_tricode_norm", "_player_name_norm"]
    snapshot_feature_cols = ["action_props_as_of_ts", *[col for col in ACTION_MARKET_FEATURE_COLUMNS if col != "action_props_as_of_ts"]]
    candidate = working[
        ["_row_id", *merge_cols, tip_col] + ([as_of_col] if as_of_col in working.columns else [])
    ].merge(
        snaps_join[merge_cols + ["_game_date_offset_days", *snapshot_feature_cols]],
        on=merge_cols,
        how="left",
    )
    candidate = candidate[candidate["an_has_any_props"].fillna(0).astype(float) > 0].copy()

    if strict_asof and not candidate.empty:
        tip_ts = pd.to_datetime(candidate.get(tip_col), utc=True, errors="coerce")
        if as_of_col in candidate.columns:
            feature_as_of_ts = pd.to_datetime(candidate.get(as_of_col), utc=True, errors="coerce")
            ceiling = pd.concat([tip_ts, feature_as_of_ts], axis=1).min(axis=1)
        else:
            ceiling = tip_ts
        props_asof = pd.to_datetime(candidate["action_props_as_of_ts"], utc=True, errors="coerce")
        if clamp_late_asof_to_game_date:
            game_day_floor = pd.to_datetime(candidate["game_date"], utc=True, errors="coerce")
            late_mask = props_asof.notna() & ceiling.notna() & (props_asof > ceiling)
            props_asof = props_asof.where(~late_mask, game_day_floor)
            candidate["action_props_as_of_ts"] = props_asof
        candidate = candidate.loc[
            props_asof.notna() & ceiling.notna() & (props_asof <= ceiling)
        ].copy()

    if candidate.empty:
        for col, default in defaults.items():
            working[col] = default
        return working.drop(columns=["_row_id", "_team_tricode_norm", "_player_name_norm"], errors="ignore")

    candidate["_offset_abs"] = pd.to_numeric(candidate["_game_date_offset_days"], errors="coerce").fillna(0).abs()
    min_offset = candidate.groupby("_row_id")["_offset_abs"].transform("min")
    candidate = candidate.loc[candidate["_offset_abs"] == min_offset].copy()
    candidate = candidate.sort_values(
        ["_row_id", "action_props_as_of_ts"],
        kind="mergesort",
    )
    latest = candidate.groupby("_row_id", as_index=False).tail(1)
    latest = latest[["_row_id", *ACTION_MARKET_FEATURE_COLUMNS]]

    working = working.merge(latest, on="_row_id", how="left", suffixes=("", "_an"))
    for col in ACTION_MARKET_FEATURE_COLUMNS:
        incoming_col = f"{col}_an"
        if incoming_col in working.columns:
            if col in working.columns:
                working[col] = working[incoming_col].combine_first(working[col])
            else:
                working[col] = working[incoming_col]
            working.drop(columns=[incoming_col], inplace=True)
    for col, default in defaults.items():
        if col not in working.columns:
            working[col] = default
        working[col] = working[col].fillna(default)

    return working.drop(columns=["_row_id", "_team_tricode_norm", "_player_name_norm"], errors="ignore")
