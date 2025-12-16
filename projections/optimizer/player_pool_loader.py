"""Helpers to build optimizer-ready player pools from local gold artifacts."""

from __future__ import annotations

import glob
import os
import re
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from projections.dk.salaries_schema import dk_salaries_gold_path


def get_projections_data_root() -> str:
    """
    Return the base projections data root, using the PROJECTIONS_DATA_ROOT
    env var if set, otherwise defaulting to /home/daniel/projections-data.
    """
    return os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data")


def _load_parquet_glob(path_glob: str, kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No {kind} parquet files found under {path_glob}")
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def _load_parquet_from_candidates(globs: list[str], kind: str) -> pd.DataFrame:
    last_err: Exception | None = None
    for pat in globs:
        try:
            return _load_parquet_glob(pat, kind)
        except FileNotFoundError as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    raise FileNotFoundError(f"No {kind} parquet files found for patterns={globs}")


def load_minutes_for_date(game_date: str, root: Optional[str] = None) -> pd.DataFrame:
    """
    Load minutes projections for the given game_date from:
        {root}/gold/projections_minutes_v1/game_date={game_date}/*.parquet

    Returns a DataFrame. If no files are found, raises FileNotFoundError.
    """

    base = root or get_projections_data_root()
    globs = [
        os.path.join(
            base,
            "gold",
            "projections_minutes_v1",
            f"game_date={game_date}",
            "*.parquet",
        ),
        os.path.join(
            base,
            "gold",
            "projections_minutes_v1",
            game_date,
            "*.parquet",
        ),
    ]
    return _load_parquet_from_candidates(globs, kind="minutes projections")


def load_fpts_for_date(game_date: str, root: Optional[str] = None) -> pd.DataFrame:
    """
    Load fantasy points projections for the given game_date from:
        {root}/gold/projections_fpts_v1/game_date={game_date}/*.parquet

    Returns a DataFrame. If no files are found, raises FileNotFoundError.
    """

    base = root or get_projections_data_root()
    globs = [
        os.path.join(
            base,
            "gold",
            "projections_fpts_v1",
            f"game_date={game_date}",
            "*.parquet",
        ),
        os.path.join(
            base,
            "gold",
            "projections_fpts_v1",
            game_date,
            "run=*",
            "*.parquet",
        ),
    ]
    return _load_parquet_from_candidates(globs, kind="fantasy points projections")


def load_salaries_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a DK salaries CSV for this slate.

    DK-specific normalization for now; extend later for other sites.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Salaries CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names
    norm_cols = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=norm_cols, inplace=True)

    # Map common DK headers to normalized names
    rename_map = {
        "name": "name",
        "player": "name",
        "id": "dk_id",
        "salary": "salary",
        "teamabbrev": "team",
        "team": "team",
        "position": "positions",
        "positions": "positions",
        "game info": "game_info",
    }
    for src, dest in rename_map.items():
        if src in df.columns and dest not in df.columns:
            df.rename(columns={src: dest}, inplace=True)

    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    if "positions" in df.columns:
        df["positions"] = df["positions"].astype(str)
    return df


def _pick_join_key(left: pd.DataFrame, right: pd.DataFrame) -> Optional[str]:
    for key in ("player_id", "dk_id", "name", "player_name"):
        if key in left.columns and key in right.columns:
            return key
    return None


def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_positions(val: object) -> list[str]:
    if isinstance(val, str):
        text = val.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)
                return _normalize_positions(parsed)
            except Exception:
                pass
        return [p.strip() for p in re.split(r"[\/,]", text) if p.strip()]
    if isinstance(val, np.ndarray):
        return [str(p).strip() for p in val.tolist() if str(p).strip()]
    if isinstance(val, (list, tuple, set)):
        return [str(p).strip() for p in val if str(p).strip()]
    if isinstance(val, Iterable) and not isinstance(val, (bytes, bytearray)):
        try:
            return [str(p).strip() for p in list(val) if str(p).strip()]
        except Exception:
            return []
    return []


def _normalize_name_value(val: object) -> str:
    if val is None:
        return ""
    # Strip accents so "Jokić" matches "Jokic" when joining on names.
    normalized = unicodedata.normalize("NFKD", str(val))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().lower()


def _normalize_team_value(val: object) -> str:
    if val is None:
        return ""
    return str(val).strip().upper()


def build_player_pool_from_gold(
    game_date: str,
    site: str,
    *,
    draft_group_id: int | None = None,
    root: Path | None = None,
    salaries_csv: str | None = None,
) -> pd.DataFrame:
    """
    Build a player pool DataFrame for the optimizer by joining salaries
    with minutes and FPTS projections for the given game_date.

    If salaries_csv is not provided, uses the gold dk_salaries parquet for
    the (game_date, site, draft_group_id).

    Expected columns (best effort, will raise if absent):
        - player_id or name (join key)
        - salary
        - positions (pipe/slash/comma separated or list)
        - projection column such as dk_fpts_mean / proj / fpts_mean
    """

    data_root = Path(root) if root is not None else Path(get_projections_data_root())
    minutes_df = load_minutes_for_date(game_date, root=str(data_root))
    try:
        fpts_df = load_fpts_for_date(game_date, root=str(data_root))
        # Prefer the latest FPTS run when multiple runs are present for the same date.
        if "minutes_run_id" in fpts_df.columns:
            fpts_df = (
                fpts_df.sort_values("minutes_run_id")
                .drop_duplicates(subset=["player_id", "game_id"], keep="last")
            )
        elif "run_id" in fpts_df.columns:
            fpts_df = (
                fpts_df.sort_values("run_id")
                .drop_duplicates(subset=["player_id", "game_id"], keep="last")
            )
    except FileNotFoundError as exc:
        print(f"[player-pool] warning: {exc}; falling back to minutes-only projections", file=sys.stderr)
        fpts_df = minutes_df.copy()
    if minutes_df.empty:
        try:
            print(f"[player-pool] minutes data empty for {game_date}; falling back to fpts only")
        except Exception:
            pass
        minutes_df = fpts_df.copy()
    if salaries_csv:
        salaries_df = load_salaries_from_csv(salaries_csv)
    else:
        if draft_group_id is None:
            raise ValueError("draft_group_id is required when salaries_csv is not provided")
        salaries_path = dk_salaries_gold_path(
            data_root, site=site, game_date=game_date, draft_group_id=draft_group_id
        )
        if not salaries_path.exists():
            raise FileNotFoundError(f"Salaries parquet not found: {salaries_path}")
        salaries_df = pd.read_parquet(salaries_path)
        # Align column names for downstream join logic
        if "dk_player_id" in salaries_df.columns and "dk_id" not in salaries_df.columns:
            salaries_df = salaries_df.assign(dk_id=salaries_df["dk_player_id"])
        if "display_name" in salaries_df.columns and "name" not in salaries_df.columns:
            salaries_df = salaries_df.assign(name=salaries_df["display_name"])
        if "team_abbrev" in salaries_df.columns and "team" not in salaries_df.columns:
            salaries_df = salaries_df.assign(team=salaries_df["team_abbrev"])

    key_minutes_fpts = _pick_join_key(minutes_df, fpts_df)
    if key_minutes_fpts is None:
        raise ValueError(
            "Unable to join minutes and fpts projections; expected a shared column like "
            "'player_id' or 'name'."
        )
    if minutes_df is fpts_df or minutes_df.equals(fpts_df):
        projections_df = minutes_df.copy()
    else:
        projections_df = minutes_df.merge(fpts_df, on=key_minutes_fpts, how="inner", suffixes=("", "_fpts"))

    # Build normalized join keys on name + team
    proj_name_col = _first_present(projections_df, ["player_name", "name", "display_name", "Name"])
    sal_name_col = _first_present(salaries_df, ["display_name", "name", "player_name", "Name"])
    proj_team_col = _first_present(projections_df, ["team_tricode", "team_abbrev", "team", "team_name", "Team"])
    sal_team_col = _first_present(salaries_df, ["team_abbrev", "team_tricode", "team", "Team"])

    if not proj_name_col or not sal_name_col or not proj_team_col or not sal_team_col:
        raise ValueError(
            "Unable to join projections with salaries; missing name/team columns. "
            f"projection columns={sorted(projections_df.columns.tolist())} "
            f"salaries columns={sorted(salaries_df.columns.tolist())}"
        )

    projections_df = projections_df.copy()
    salaries_df = salaries_df.copy()
    projections_df["__join_name"] = projections_df[proj_name_col].apply(_normalize_name_value)
    salaries_df["__join_name"] = salaries_df[sal_name_col].apply(_normalize_name_value)
    projections_df["__join_team"] = projections_df[proj_team_col].apply(_normalize_team_value)
    salaries_df["__join_team"] = salaries_df[sal_team_col].apply(_normalize_team_value)

    merged = projections_df.merge(
        salaries_df, on=["__join_name", "__join_team"], how="inner", suffixes=("", "_salary")
    )
    merged.drop(columns=["__join_name", "__join_team"], inplace=True, errors="ignore")

    sal_salary_col = _first_present(salaries_df, ["salary", "Salary", "dk_salary", "DK Salary", "salary_salary"])
    proj_rows = len(projections_df)
    sal_rows = len(salaries_df)
    joined_rows = len(merged)
    try:
        print(
            f"[player-pool] projections={proj_rows} salaries={sal_rows} joined={joined_rows} key=name+team"
        )
        if proj_rows > 0 and joined_rows < 0.8 * proj_rows:
            print(
                f"[player-pool] warning: join dropped {proj_rows - joined_rows} rows ({(1 - joined_rows / proj_rows):.1%})"
            )
        if sal_salary_col and "_merge" not in salaries_df.columns:
            missing_sal = salaries_df.merge(
                projections_df[["__join_name", "__join_team"]].drop_duplicates(),
                on=["__join_name", "__join_team"],
                how="left",
                indicator=True,
            )
            missing_sal = missing_sal[missing_sal["_merge"] == "left_only"]
            if not missing_sal.empty:
                missing_sal["_salary_numeric"] = pd.to_numeric(
                    missing_sal[sal_salary_col], errors="coerce"
                )
                top_missing = missing_sal.sort_values("_salary_numeric", ascending=False).head(5)
                missing_names: list[str] = []
                for _, row in top_missing.iterrows():
                    name_val = row.get(sal_name_col, row.get("display_name", ""))
                    sal_val = row.get("_salary_numeric")
                    sal_str = f"${int(sal_val):,}" if pd.notna(sal_val) else "unknown"
                    missing_names.append(f"{name_val} ({sal_str})")
                if missing_names:
                    print(
                        "[player-pool] missing high-salary names (no projection match): "
                        + ", ".join(missing_names)
                    )
    except Exception:
        pass

    # Coalesce duplicated name columns if created by the join
    for base in ("name", "display_name", "player_name"):
        x_col, y_col = f"{base}_x", f"{base}_y"
        if x_col in merged.columns or y_col in merged.columns:
            merged[base] = merged.get(x_col, merged.get(y_col))
            if y_col in merged.columns:
                merged[base] = merged[base].fillna(merged[y_col])
            merged.drop(columns=[c for c in (x_col, y_col) if c in merged.columns], inplace=True)

    player_id_col = _first_present(
        merged,
        [
            "player_id",
            "player_id_salary",
            "dk_id",
            "dk_player_id",
            "dk_id_salary",
            "id",
        ],
    )
    name_col = _first_present(
        merged,
        [
            "display_name",
            "display_name_salary",
            "name_salary",
            "name",
            "player_name",
            "Name",
        ],
    )
    team_col = _first_present(
        merged,
        [
            "team_abbrev",
            "team_abbrev_salary",
            "team_tricode",
            "team",
            "team_salary",
            "Team",
        ],
    )
    pos_col = _first_present(
        merged,
        [
            "positions",
            "positions_x",
            "positions_y",
            "positions_salary",
            "position",
            "position_x",
            "position_y",
            "Position",
            "Pos",
        ],
    )
    if pos_col:
        normalized_positions = merged[pos_col].apply(_normalize_positions)
    else:
        normalized_positions = pd.Series([[] for _ in range(len(merged))], index=merged.index)

    if normalized_positions.apply(len).max() == 0:
        for alt_col in ("positions_salary", "position", "positions_y", "Pos"):
            if alt_col in merged.columns:
                alt_norm = merged[alt_col].apply(_normalize_positions)
                if alt_norm.apply(len).max() > 0:
                    normalized_positions = alt_norm
                    break
    salary_col = _first_present(
        merged,
        [
            "salary_salary",
            "salary",
            "Salary",
            "dk_salary",
            "DK Salary",
        ],
    )
    proj_col = _first_present(
        merged,
        [
            "proj_fpts",
            "dk_fpts_mean",
            "fpts_mean",
            "fpts",
            "proj",
            "projection",
            "dk_proj",
            "minutes_p50",
            "minutes",
        ],
    )
    dk_player_col = _first_present(
        merged, ["dk_player_id", "dk_player_id_salary", "dk_id", "dk_id_salary"]
    )

    if not player_id_col or not salary_col or not proj_col:
        raise ValueError(
            "Player pool is missing required columns; need at least player_id (or name), salary, and a projection column."
        )

    try:
        median_proj = pd.to_numeric(merged[proj_col], errors="coerce").median()
        if median_proj > 200:
            print(
                f"[player-pool] WARNING: projection column {proj_col} has suspicious median={median_proj:.2f}; check FPTS semantics."
            )
    except Exception:
        pass

    result = pd.DataFrame()
    result["player_id"] = pd.to_numeric(merged[player_id_col], errors="coerce").astype("Int64")
    result["name"] = merged[name_col] if name_col else merged[player_id_col]
    if team_col:
        result["team"] = merged[team_col]

    # Positions: prefer the normalized positions, but if still empty, try to
    # re-parse the raw salaries column directly.
    positions_series = normalized_positions.reset_index(drop=True)
    if positions_series.apply(len).max() == 0 and "positions" in merged.columns:
        positions_series = merged["positions"].apply(_normalize_positions).reset_index(drop=True)
    result["positions"] = positions_series

    result["salary"] = pd.to_numeric(merged[salary_col], errors="coerce")
    result["dk_fpts_mean"] = pd.to_numeric(merged[proj_col], errors="coerce")
    if dk_player_col:
        result["dk_player_id"] = pd.to_numeric(merged[dk_player_col], errors="coerce").astype("Int64")

    for minutes_col in ("minutes_p10", "minutes_p50", "minutes_p90", "minutes"):
        if minutes_col in merged.columns:
            result[minutes_col] = merged[minutes_col]

    for optional in ("own_proj", "stddev", "dk_id", "status", "is_disabled"):
        if optional in merged.columns:
            result[optional] = merged[optional]

    # Keep the site in the payload for downstream consumers
    result["site"] = site
    return result
