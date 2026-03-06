from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from projections.post_contest.replay_calibration_service import build_replay_calibration_artifacts

app = typer.Typer(add_completion=False, help="Aggregate replay analytics into calibration artifacts.")


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(None, help="Override PROJECTIONS_DATA_ROOT."),
    output_dir: Optional[Path] = typer.Option(None, help="Override calibration artifact output directory."),
) -> None:
    bundle = build_replay_calibration_artifacts(
        data_root=data_root,
        output_dir=output_dir,
    )
    typer.echo(str(bundle.summary_path))


if __name__ == "__main__":
    app()
