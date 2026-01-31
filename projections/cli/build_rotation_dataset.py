"""Build + publish a rot_v1 bundle from Phase 1 PBP stints."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from projections.rotations.dataset import build_rotation_dataset
from projections.rotations.manifest import (
    build_manifest,
    sha256_file,
    write_json,
    write_latest_published_run_id,
)
from projections.rotations.qa import run_qa_gates
from projections.rotations.schemas import ROT_V1_SCHEMA_VERSION

console = Console()
app = typer.Typer(help="Build and publish rot_v1 rotation datasets (rotation_events + rotation_labels).")


DEFAULT_PBP_BUNDLE = Path("/home/daniel/projections-data/artifacts/pbp_v1/LATEST_PUBLISHED")
DEFAULT_ROT_ARTIFACTS_ROOT = Path("/home/daniel/projections-data/artifacts/rot_v1")
DEFAULT_SCHEDULE_ROOT = Path("/home/daniel/projections-data/silver/schedule")


def _resolve_bundle_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file():
        run_id = path.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError(f"Empty bundle pointer: {path}")
        resolved = path.parent / run_id
        if not resolved.exists():
            raise FileNotFoundError(f"Pointer {path} -> {resolved} does not exist")
        return resolved
    raise FileNotFoundError(f"Bundle path not found: {path}")


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

def _infer_season_start_year(game_id: str) -> int | None:
    s = str(game_id).strip()
    # canonical examples:
    # - regular: 0022400061 (season=2024-25)
    # - playoffs: 0042400101 (season=2024-25)
    if len(s) >= 5 and s[3:5].isdigit():
        return 2000 + int(s[3:5])
    return None


def _schedule_paths_for_bundle(*, schedule_root: Path, sample_game_id: str) -> list[Path]:
    year = _infer_season_start_year(sample_game_id)
    if year is not None:
        season_dir = schedule_root / f"season={year}"
        if season_dir.exists():
            return sorted(season_dir.glob("month=*/schedule.parquet"))
    return sorted(schedule_root.glob("season=*/month=*/schedule.parquet"))


def _load_schedule_all(*, schedule_paths: list[Path]) -> pd.DataFrame:
    if not schedule_paths:
        raise FileNotFoundError("No schedule parquet files found.")
    frames = [
        pd.read_parquet(
            path,
            columns=[
                "game_id",
                "season",
                "home_team_id",
                "away_team_id",
                "home_team_tricode",
                "away_team_tricode",
            ],
        )
        for path in schedule_paths
    ]
    schedule = pd.concat(frames, ignore_index=True)
    schedule["game_id"] = pd.to_numeric(schedule["game_id"], errors="coerce").astype("Int64")
    schedule = schedule.dropna(subset=["game_id"]).copy()
    schedule["game_id"] = schedule["game_id"].astype(int)
    schedule["home_team_tricode"] = schedule["home_team_tricode"].astype("string").str.upper()
    schedule["away_team_tricode"] = schedule["away_team_tricode"].astype("string").str.upper()
    schedule["home_team_id"] = pd.to_numeric(schedule["home_team_id"], errors="coerce").astype("Int64")
    schedule["away_team_id"] = pd.to_numeric(schedule["away_team_id"], errors="coerce").astype("Int64")
    schedule = schedule.dropna(subset=["home_team_id", "away_team_id"]).copy()
    schedule["home_team_id"] = schedule["home_team_id"].astype(int)
    schedule["away_team_id"] = schedule["away_team_id"].astype(int)
    schedule["season"] = schedule["season"].astype("string")
    schedule = schedule.drop_duplicates(subset=["game_id"], keep="last")
    return schedule


def _build_tricode_to_team_id(schedule_all: pd.DataFrame) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for tricode_col, team_id_col in [
        ("home_team_tricode", "home_team_id"),
        ("away_team_tricode", "away_team_id"),
    ]:
        pairs = schedule_all[[tricode_col, team_id_col]].dropna().drop_duplicates()
        for tricode, team_id in zip(pairs[tricode_col].tolist(), pairs[team_id_col].tolist()):
            if tricode is None:
                continue
            mapping[str(tricode).upper()] = int(team_id)
    return mapping


def _infer_away_home_tricodes_from_filename(path: Path) -> tuple[str, str] | None:
    base = path.name.rsplit(".", 1)[0]
    parts = base.rsplit("-", 2)
    if len(parts) != 3:
        return None
    maybe_matchup = parts[2]
    if "@" not in maybe_matchup:
        return None
    away, home = maybe_matchup.split("@", 1)
    away = away.strip().upper()
    home = home.strip().upper()
    if not away or not home:
        return None
    return away, home


def _infer_away_home_tricodes_from_scores(path: Path) -> tuple[str, str] | None:
    df = pd.read_parquet(path, columns=["event_index", "team", "home_score", "away_score"])
    if df.empty:
        return None
    work = df.copy()
    work["event_index"] = pd.to_numeric(work["event_index"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["event_index"]).copy()
    if work.empty:
        return None
    work = work.sort_values("event_index", kind="mergesort")
    work["team"] = work["team"].astype("string").str.upper()
    work["home_score"] = pd.to_numeric(work["home_score"], errors="coerce").fillna(0).astype(int)
    work["away_score"] = pd.to_numeric(work["away_score"], errors="coerce").fillna(0).astype(int)
    work["home_delta"] = work["home_score"].diff().fillna(0).astype(int)
    work["away_delta"] = work["away_score"].diff().fillna(0).astype(int)

    scoring = work[(work["home_delta"] > 0) | (work["away_delta"] > 0)].copy()
    if scoring.empty:
        return None

    home_votes = scoring.loc[(scoring["home_delta"] > 0) & (scoring["away_delta"] == 0), "team"].dropna()
    away_votes = scoring.loc[(scoring["away_delta"] > 0) & (scoring["home_delta"] == 0), "team"].dropna()
    if home_votes.empty or away_votes.empty:
        return None

    home = str(home_votes.value_counts().idxmax()).upper()
    away = str(away_votes.value_counts().idxmax()).upper()
    if not home or not away:
        return None
    return away, home


def _find_pbp_part_for_game(*, pbp_parts_dir: Path, game_id_str: str) -> Path:
    matches = sorted(pbp_parts_dir.glob(f"*{game_id_str}*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No pbp_events part file found for game_id={game_id_str} in {pbp_parts_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple pbp_events part files matched for game_id={game_id_str}: {matches[:3]}")
    return matches[0]


def _infer_missing_schedule_rows(
    *,
    pbp_parts_dir: Path,
    missing_game_ids_int: list[int],
    tricode_to_team_id: dict[str, int],
    season_label: str,
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict] = []
    used_pbp_paths: list[Path] = []
    for gid in missing_game_ids_int:
        game_id_str = str(int(gid)).zfill(10)
        pbp_path = _find_pbp_part_for_game(pbp_parts_dir=pbp_parts_dir, game_id_str=game_id_str)
        used_pbp_paths.append(pbp_path)

        inferred = _infer_away_home_tricodes_from_scores(pbp_path) or _infer_away_home_tricodes_from_filename(pbp_path)
        if inferred is None:
            raise RuntimeError(f"Could not infer away/home tricodes for {pbp_path}")
        away_tri, home_tri = inferred

        if away_tri not in tricode_to_team_id or home_tri not in tricode_to_team_id:
            raise ValueError(
                f"Missing team_id mapping for tricodes away={away_tri} home={home_tri} "
                f"(known={len(tricode_to_team_id)} tricodes)"
            )

        rows.append(
            {
                "game_id": int(gid),
                "season": str(season_label),
                "home_team_id": int(tricode_to_team_id[home_tri]),
                "away_team_id": int(tricode_to_team_id[away_tri]),
            }
        )
    return pd.DataFrame(rows), used_pbp_paths


@app.command()
def run(
    run_id: str = typer.Option(..., "--run-id", help="rot_v1 run id (used as output directory name)."),
    pbp_bundle: Path = typer.Option(
        DEFAULT_PBP_BUNDLE,
        "--pbp-bundle",
        help="Path to pbp_v1 bundle dir or LATEST_PUBLISHED pointer file.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite outputs if present."),
    resume: bool = typer.Option(False, "--resume", help="Resume a partially written bundle."),
    limit_games: int | None = typer.Option(None, "--limit-games", help="Optional limit for smoke builds."),
) -> None:
    if overwrite and resume:
        console.print("[red]--overwrite and --resume are mutually exclusive[/red]")
        raise typer.Exit(2)

    pbp_dir = _resolve_bundle_dir(pbp_bundle)
    stints_path = pbp_dir / "stints.parquet"
    player_stints_path = pbp_dir / "player_stints.parquet"
    if not stints_path.exists():
        console.print(f"[red]Missing input[/red] {stints_path}")
        raise typer.Exit(1)
    if not player_stints_path.exists():
        console.print(f"[red]Missing input[/red] {player_stints_path}")
        raise typer.Exit(1)

    out_dir = DEFAULT_ROT_ARTIFACTS_ROOT / run_id
    if out_dir.exists() and not (overwrite or resume):
        console.print("[red]Output bundle exists; pass --overwrite or --resume[/red]")
        raise typer.Exit(2)
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rotation_events_path = out_dir / "rotation_events.parquet"
    rotation_labels_path = out_dir / "rotation_labels.parquet"
    qa_report_path = out_dir / "qa_report.json"
    qa_failures_path = out_dir / "qa_failures.parquet"
    input_hashes_path = out_dir / "input_hashes.json"
    manifest_path = out_dir / "manifest.json"
    published_marker = out_dir / "PUBLISHED"

    # Build datasets (or load in resume mode).
    if resume and rotation_events_path.exists() and rotation_labels_path.exists():
        rotation_events = pd.read_parquet(rotation_events_path)
        rotation_labels = pd.read_parquet(rotation_labels_path)
    else:
        stints = pd.read_parquet(stints_path)
        player_stints = pd.read_parquet(player_stints_path)

        if limit_games is not None:
            unique_games = sorted(stints["game_id"].astype("string").unique().tolist())
            keep_games = set(unique_games[: int(limit_games)])
            stints = stints[stints["game_id"].astype("string").isin(keep_games)].copy()
            player_stints = player_stints[player_stints["game_id"].astype("string").isin(keep_games)].copy()

        game_id_series = stints["game_id"].astype("string")
        game_ids_int = set(pd.to_numeric(game_id_series, errors="coerce").astype(int).tolist())
        sample_game_id = str(game_id_series.iloc[0]) if len(game_id_series) else "unknown"
        schedule_paths = _schedule_paths_for_bundle(schedule_root=DEFAULT_SCHEDULE_ROOT, sample_game_id=sample_game_id)
        schedule_all = _load_schedule_all(schedule_paths=schedule_paths)
        tricode_to_team_id = _build_tricode_to_team_id(schedule_all)
        schedule = schedule_all[schedule_all["game_id"].isin(list(game_ids_int))][
            ["game_id", "season", "home_team_id", "away_team_id"]
        ].copy()
        found = set(schedule["game_id"].astype(int).tolist())
        missing = sorted(game_ids_int - found)
        if missing:
            pbp_parts_dir = pbp_dir / "_parts" / "pbp_events"
            if not pbp_parts_dir.exists():
                raise FileNotFoundError(f"Missing pbp parts directory: {pbp_parts_dir}")
            season_values = schedule_all["season"].dropna().astype("string").unique().tolist()
            season_label = season_values[0] if len(season_values) == 1 else "unknown"
            inferred_rows, _ = _infer_missing_schedule_rows(
                pbp_parts_dir=pbp_parts_dir,
                missing_game_ids_int=missing,
                tricode_to_team_id=tricode_to_team_id,
                season_label=season_label,
            )
            schedule = pd.concat([schedule, inferred_rows], ignore_index=True)
            schedule = schedule.drop_duplicates(subset=["game_id"], keep="last")

        dataset = build_rotation_dataset(stints=stints, player_stints=player_stints, schedule=schedule)
        rotation_events = dataset.rotation_events
        rotation_labels = dataset.rotation_labels

        _atomic_write_parquet(rotation_events, rotation_events_path)
        _atomic_write_parquet(rotation_labels, rotation_labels_path)

    season_values = rotation_events["season_id"].dropna().astype("string").unique().tolist() if len(rotation_events) else []
    season_id = season_values[0] if len(season_values) == 1 else "unknown"

    # Input hashes (best-effort; keep deterministic by file content).
    if not (resume and input_hashes_path.exists()):
        files: dict[str, str] = {
            str(stints_path): sha256_file(stints_path),
            str(player_stints_path): sha256_file(player_stints_path),
            str(pbp_dir / "pbp_events.parquet"): sha256_file(pbp_dir / "pbp_events.parquet")
            if (pbp_dir / "pbp_events.parquet").exists()
            else "missing",
        }
        stints_game_ids_int = set(pd.to_numeric(rotation_events["game_id"], errors="coerce").astype(int).tolist())
        sample_game_id = str(rotation_events["game_id"].iloc[0]) if len(rotation_events) else "unknown"
        schedule_paths = _schedule_paths_for_bundle(schedule_root=DEFAULT_SCHEDULE_ROOT, sample_game_id=sample_game_id)
        for p in schedule_paths:
            files[str(p)] = sha256_file(p)
        # Hash pbp parts used to infer home/away for games missing in schedule.
        schedule_all = _load_schedule_all(schedule_paths=schedule_paths)
        found = set(schedule_all["game_id"].astype(int).tolist())
        missing = sorted(stints_game_ids_int - found)
        if missing:
            pbp_parts_dir = pbp_dir / "_parts" / "pbp_events"
            for gid in missing:
                game_id_str = str(int(gid)).zfill(10)
                part_path = _find_pbp_part_for_game(pbp_parts_dir=pbp_parts_dir, game_id_str=game_id_str)
                files[str(part_path)] = sha256_file(part_path)
        write_json(input_hashes_path, {"files": files})

    # QA gates.
    if not (resume and qa_report_path.exists() and qa_failures_path.exists()):
        qa = run_qa_gates(
            rotation_events,
            season_id=season_id,
            run_id=run_id,
            schema_version=ROT_V1_SCHEMA_VERSION,
            tolerance_sec=0,
        )
        write_json(qa_report_path, qa.report)
        _atomic_write_parquet(qa.failures, qa_failures_path)
    else:
        qa = None

    # Manifest + publish markers.
    repo_root = Path(__file__).resolve().parents[2]
    manifest = build_manifest(repo_root=repo_root, season_id=season_id, run_id=run_id, input_hashes_path=input_hashes_path)
    write_json(manifest_path, manifest)
    published_marker.write_text("published\n", encoding="utf-8")
    write_latest_published_run_id(DEFAULT_ROT_ARTIFACTS_ROOT, run_id)

    console.print(f"rotation_events: {rotation_events_path} ({len(rotation_events):,} rows)")
    console.print(f"rotation_labels: {rotation_labels_path} ({len(rotation_labels):,} rows)")
    console.print(f"qa_report: {qa_report_path}")
    console.print(f"qa_failures: {qa_failures_path}")
    console.print(f"input_hashes: {input_hashes_path}")
    console.print(f"manifest: {manifest_path}")
    console.print(f"updated LATEST_PUBLISHED in {DEFAULT_ROT_ARTIFACTS_ROOT}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
