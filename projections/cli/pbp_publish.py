"""Publish an immutable PBP v1 bundle by writing a manifest and updating pointers."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from projections.pbp.publish import build_manifest, write_json, write_latest_published_run_id, read_json

console = Console()
app = typer.Typer(help="Publish a PBP v1 run bundle (writes manifest + updates LATEST_PUBLISHED).")


@app.command()
def run(
    bundle_dir: Path = typer.Argument(..., help="Run bundle dir (e.g. /.../artifacts/pbp_v1/<run_id>/)."),
    artifacts_root: Path = typer.Option(
        Path("/home/daniel/projections-data/artifacts/pbp_v1"),
        "--artifacts-root",
        help="Artifacts root containing all run bundles.",
    ),
    force: bool = typer.Option(False, "--force", help="Publish even if QA has failures."),
) -> None:
    qa_report_path = bundle_dir / "qa_report.json"
    input_hashes_path = bundle_dir / "input_hashes.json"
    manifest_path = bundle_dir / "manifest.json"
    published_marker = bundle_dir / "PUBLISHED"

    if not qa_report_path.exists():
        console.print(f"[red]Missing QA report[/red] {qa_report_path}")
        raise typer.Exit(1)
    if not input_hashes_path.exists():
        console.print(f"[red]Missing input hashes[/red] {input_hashes_path}")
        raise typer.Exit(1)

    qa = read_json(qa_report_path)
    games_failed = int(qa.get("totals", {}).get("games_failed", 0))
    if games_failed and not force:
        console.print(f"[red]Refusing to publish: QA failures present[/red] games_failed={games_failed}")
        raise typer.Exit(2)

    season_id = qa.get("season_id", "unknown")
    run_id = bundle_dir.name

    repo_root = Path(__file__).resolve().parents[2]
    manifest = build_manifest(
        repo_root=repo_root,
        season_id=season_id,
        run_id=run_id,
        input_hashes_path=input_hashes_path,
    )
    write_json(manifest_path, manifest)
    published_marker.write_text("published\n", encoding="utf-8")
    write_latest_published_run_id(artifacts_root, run_id)

    console.print(f"manifest: {manifest_path}")
    console.print(f"updated LATEST_PUBLISHED in {artifacts_root}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

