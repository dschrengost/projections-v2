from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from projections.post_contest.replay_analytics_service import build_post_contest_replay_analytics

app = typer.Typer(add_completion=False, help="Build post-contest replay analytics artifacts.")


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
    output_dir: Optional[Path] = typer.Option(None, help="Override analytics artifact output directory."),
    modeled_field_version: str = typer.Option("v1_calibrated", help="Generated field library version for field calibration."),
    include_modeled_field: bool = typer.Option(True, "--include-modeled-field/--skip-modeled-field", help="Build/load generated field library for field calibration."),
    candidate_manifest_path: Optional[Path] = typer.Option(None, help="Optional explicit export manifest path for candidate-regret analysis."),
) -> None:
    bundle = build_post_contest_replay_analytics(
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
        output_dir=output_dir,
        modeled_field_version=modeled_field_version,
        include_modeled_field=include_modeled_field,
        candidate_manifest_path=candidate_manifest_path,
    )
    typer.echo(str(bundle.summary_path))


if __name__ == "__main__":
    app()
