from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds
import typer
from rich.console import Console

from projections import paths
from projections.rotations.player_map import build_person_id_to_internal_id_map

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Build minutes priors in rot_v1 internal player_id space.")


REQUIRED_MINUTES_COLS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "player_id",
    "minutes_pred_p10",
    "minutes_pred_p50",
    "minutes_pred_p90",
    "minutes_pred_play_prob",
)


def _format_game_id(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{int(value):010d}"
    except Exception:
        s = str(value).strip()
        return s.zfill(10) if s.isdigit() else s


def _discover_minutes_for_rates_paths(*, data_root: Path, season_start_year: int) -> list[Path]:
    base = data_root / "gold" / "minutes_for_rates_reconciled" / f"season={int(season_start_year)}"
    if not base.exists():
        raise FileNotFoundError(f"Missing expected minutes_for_rates_reconciled season dir: {base}")
    paths_list = sorted(base.glob("game_date=*/minutes_for_rates.parquet"))
    if not paths_list:
        raise FileNotFoundError(f"No minutes_for_rates.parquet files found under: {base}")
    return paths_list


def build_minutes_prior_internal_df(
    *,
    season_start_year: int,
    data_root: Path | None = None,
    person_source: Path | None = None,
    limit_paths: int | None = None,
    emit_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (priors_df, unmapped_rows_df).

    - `priors_df` uses INTERNAL player_id space (rot_v1 / pbp_v1 internal IDs).
    - `unmapped_rows_df` contains rows from minutes_for_rates that could not be mapped.
    """
    season_start_year = int(season_start_year)
    root = (data_root or paths.get_data_root()).expanduser().resolve()

    priors_dir = root / "artifacts" / "rot_eval_v1" / "_priors"
    diagnostics_dir = priors_dir if emit_diagnostics else None

    map_result = build_person_id_to_internal_id_map(
        season_start_year=season_start_year,
        person_source=person_source,
        data_root=root,
        diagnostics_dir=diagnostics_dir,
    )
    person_to_internal = map_result.person_id_to_internal_id

    paths_list = _discover_minutes_for_rates_paths(data_root=root, season_start_year=season_start_year)
    if limit_paths is not None:
        paths_list = paths_list[: int(limit_paths)]

    dataset = ds.dataset([str(p) for p in paths_list], format="parquet")
    schema_names = set(dataset.schema.names)
    missing = [c for c in REQUIRED_MINUTES_COLS if c not in schema_names]
    if missing:
        raise ValueError(
            f"minutes_for_rates_reconciled missing required columns {missing}. "
            f"Found schema={sorted(schema_names)}"
        )

    table = dataset.to_table(columns=list(REQUIRED_MINUTES_COLS))
    df = table.to_pandas()

    df["game_id"] = df["game_id"].map(_format_game_id).astype("string")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["nba_person_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["minutes_pred_p10"] = pd.to_numeric(df["minutes_pred_p10"], errors="coerce").fillna(0.0).astype("float64")
    df["minutes_pred_p50"] = pd.to_numeric(df["minutes_pred_p50"], errors="coerce").fillna(0.0).astype("float64")
    df["minutes_pred_p90"] = pd.to_numeric(df["minutes_pred_p90"], errors="coerce").fillna(0.0).astype("float64")
    df["minutes_pred_play_prob"] = (
        pd.to_numeric(df["minutes_pred_play_prob"], errors="coerce")
        .fillna(0.0)
        .astype("float64")
        .clip(0.0, 1.0)
    )
    df = df.dropna(subset=["game_id", "team_id", "nba_person_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["nba_person_id"] = df["nba_person_id"].astype(int)

    df["player_id"] = df["nba_person_id"].map(person_to_internal).astype("Int64")
    unmapped_rows = df[df["player_id"].isna()].copy()
    unmapped_rows = unmapped_rows.sort_values(["game_id", "team_id", "nba_person_id"], kind="mergesort").reset_index(drop=True)

    df = df.dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype("string"),
            "team_id": df["team_id"].astype("int64"),
            "player_id": df["player_id"].astype("int64"),
            "minutes_prior": df["minutes_pred_p50"].astype("float64").clip(lower=0.0),
            "minutes_p10": df["minutes_pred_p10"].astype("float64").clip(lower=0.0),
            "minutes_p90": df["minutes_pred_p90"].astype("float64").clip(lower=0.0),
            "play_prob": df["minutes_pred_play_prob"].astype("float64").clip(0.0, 1.0),
        }
    )
    out = out.sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)

    return out, unmapped_rows


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


@app.command()
def main(
    *,
    season_start_year: int = typer.Option(..., "--season-start-year", help="Season start year (e.g. 2024 for 2024-25)."),
    person_source: Optional[Path] = typer.Option(
        None,
        "--person-source",
        help="Optional override path for personId/name source (defaults to silver/nba_daily_lineups/season=YYYY).",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output parquet if it exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build and validate but do not write outputs."),
) -> None:
    root = paths.get_data_root()
    priors_dir = root / "artifacts" / "rot_eval_v1" / "_priors"
    out_path = priors_dir / f"minutes_prior_internal_season={int(season_start_year)}.parquet"

    if out_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    df, unmapped = build_minutes_prior_internal_df(
        season_start_year=season_start_year,
        data_root=root,
        person_source=person_source,
        emit_diagnostics=not dry_run,
    )

    console.print(
        f"built priors: rows={len(df):,} games={df['game_id'].nunique():,} players={df['player_id'].nunique():,}"
    )
    console.print(
        f"unmapped rows dropped: {len(unmapped):,} (unique personIds={unmapped['nba_person_id'].nunique() if not unmapped.empty else 0:,})"
    )
    if not df.empty:
        console.print(f"player_id max: {int(df['player_id'].max())}")

    if dry_run:
        console.print("dry-run: skipping parquet write")
        return

    if not unmapped.empty:
        unmapped_path = priors_dir / f"minutes_prior_internal_unmapped_rows_season={int(season_start_year)}.csv"
        priors_dir.mkdir(parents=True, exist_ok=True)
        unmapped.to_csv(unmapped_path, index=False)
        console.print(f"wrote unmapped rows: {unmapped_path}")

    _atomic_write_parquet(df, out_path)
    console.print(f"wrote: {out_path}")


if __name__ == "__main__":
    app()

