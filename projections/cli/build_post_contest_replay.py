from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from projections.post_contest.replay_service import (
    replay_output_dir,
    run_post_contest_replay,
    save_actual_field_library,
    write_resolved_entries_parquet,
)

app = typer.Typer(add_completion=False, help="Build exact post-contest replay artifacts and run flashback sim.")


@app.command()
def main(
    contest_id: str = typer.Option(..., help="DraftKings contest id."),
    date: str = typer.Option(..., "--date", help="Contest date in YYYY-MM-DD format."),
    user_pattern: str = typer.Option(..., help="Substring match against EntryName for the user's entries."),
    draft_group_id: Optional[int] = typer.Option(None, help="DraftKings draft group id override."),
    run_id: Optional[str] = typer.Option(None, help="Optional projections run id."),
    entry_fee: Optional[float] = typer.Option(None, help="Override entry fee when inventory metadata is missing."),
    archetype: str = typer.Option("medium", help="Contest sim payout archetype."),
    worlds_source: str = typer.Option("gtv2", help="World source: gtv2|sim_v2|auto."),
    ownership_mode: str = typer.Option("field_only", help="Contest sim ownership mode."),
    data_root: Optional[Path] = typer.Option(None, help="Override PROJECTIONS_DATA_ROOT."),
    output_dir: Optional[Path] = typer.Option(None, help="Override artifact output directory."),
    save_normalized: bool = typer.Option(True, "--save-normalized/--no-save-normalized", help="Write normalized resolved entries parquet."),
    save_field_library: bool = typer.Option(True, "--save-field-library/--no-save-field-library", help="Write exact opponent field library JSON."),
) -> None:
    replay_run = run_post_contest_replay(
        contest_id=contest_id,
        game_date=date,
        user_pattern=user_pattern,
        draft_group_id=draft_group_id,
        run_id=run_id,
        entry_fee=entry_fee,
        archetype=archetype,
        worlds_source=worlds_source,
        ownership_mode=ownership_mode,
        data_root=data_root,
    )

    out_dir = output_dir or replay_output_dir(
        game_date=replay_run.prepared.meta.game_date,
        contest_id=replay_run.prepared.meta.contest_id,
        user_pattern=user_pattern,
        data_root=data_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    if save_normalized:
        normalized_path = write_resolved_entries_parquet(replay_run.prepared, data_root=data_root)
        artifacts["normalized_entries_path"] = str(normalized_path)
    if save_field_library:
        field_library_path = save_actual_field_library(replay_run.prepared, data_root=data_root)
        artifacts["field_library_path"] = str(field_library_path)

    summary_path = out_dir / "summary.json"
    payload = replay_run.to_dict()
    payload["artifacts"] = artifacts
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    typer.echo(str(summary_path))


if __name__ == "__main__":
    app()
